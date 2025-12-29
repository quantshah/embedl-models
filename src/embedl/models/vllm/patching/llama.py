# Copyright (C) 2025 Embedl AB

"""
vLLM-compatible FlashHead Llama model.
"""

import os

import torch
import torch.nn as nn
from vllm.model_executor.models.llama import (
    LlamaForCausalLM as _LlamaForCausalLM,
)
from transformers import PretrainedConfig
from embedl.models.flash_head import FlashHead, get_flash_head_parameters
import torch
from vllm.model_executor.models.llama import (
    LlamaForCausalLM as _LlamaForCausalLM,
)
import torch
from typing import Optional
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)


class FlashHeadLlamaConfig(PretrainedConfig):
    """Configuration for FlashHead models."""

    model_type = "flash_head_llama"

    def __init__(
        self,
        model_or_dir: str = None,
        flash_head_cache_dir: str = "flash_head_assets",
        flash_head_special_token_ids: list[int] = None,
        n_clusters: int = None,
        n_probes: int = None,
        creation_time: float = None,
        enforce_equal_cluster_sizes: bool = True,
        **kwargs,
    ):
        self.model_or_dir = model_or_dir
        self.flash_head_cache_dir = flash_head_cache_dir
        self.flash_head_special_token_ids = flash_head_special_token_ids
        self.n_clusters = n_clusters
        self.n_probes = n_probes
        self.creation_time = creation_time
        self.enforce_equal_cluster_sizes = enforce_equal_cluster_sizes
        super().__init__(**kwargs)


class FlashHeadLlamaForCausalLM(_LlamaForCausalLM):
    """Llama model with FlashHead for efficient inference."""

    def __init__(self, *, vllm_config, prefix: str = "", **kwargs):
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

        config = vllm_config.model_config.hf_config

        # Check if FlashHead should be enabled
        model_path = vllm_config.model_config.model
        self.flash_head_enabled = False

        if hasattr(config, "flash_head_cache_dir"):
            from embedl.models.flash_head import (
                FlashHead,
                get_flash_head_parameters,
            )

            cache_dir = config.flash_head_cache_dir

            flash_params = get_flash_head_parameters(
                lm_head=self.lm_head,
                cache_dir=cache_dir,
                model_or_dir=model_path,
                n_clusters=config.n_clusters,
            )

            # Replace lm_head with FlashHead
            self.lm_head = FlashHead(
                lm_head=self.lm_head,
                n_probes=config.n_probes,
                **flash_params,
            )
            self.flash_head_enabled = True
            print("[FlashHeadLlamaForCausalLM] FlashHead enabled")

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        """
        Override logits computation to use FlashHead when appropriate.

        This is the key integration point. When batch_size=1 and seq_len=1,
        we use FlashHead to directly get the next token instead of computing
        full logits.
        """
        if not self.flash_head_enabled:
            # Use standard logits computation
            return super().compute_logits(hidden_states, sampling_metadata)

        # Handle different hidden_states shapes
        if hidden_states.ndim == 2:
            batch_size, hidden_dim = hidden_states.shape
            seq_len = 1
        else:
            batch_size, seq_len, hidden_dim = hidden_states.shape

        # FlashHead optimization only for single-token generation
        if batch_size == 1 and seq_len == 1:
            # Determine sampling parameters
            do_sample = False
            temperature = 1.0

            if sampling_metadata is not None and hasattr(
                sampling_metadata, "seq_groups"
            ):
                # Extract sampling parameters from metadata
                for seq_group in sampling_metadata.seq_groups:
                    if hasattr(seq_group, "sampling_params"):
                        params = seq_group.sampling_params
                        do_sample = params.temperature > 0
                        temperature = params.temperature if do_sample else 1.0
                        break

            # Ensure hidden_states has correct shape for FlashHead
            if hidden_states.ndim == 2:
                hidden_states = hidden_states.unsqueeze(
                    1
                )  # (B, hidden) -> (B, 1, hidden)

            # Use FlashHead to get next token directly
            next_token = self.lm_head.get_next_token(
                hidden_states=hidden_states,
                do_sample=do_sample,
                temperature=temperature,
                use_identical_tiebreak=False,
            )

            # Return token ID tensor - LogitsProcessor will convert to fake logits
            # Shape: (1, 1) with dtype int64
            return next_token

        else:
            # For batch_size > 1 or seq_len > 1, use original lm_head
            print(
                f"[FlashHeadLlamaForCausalLM] Using standard logits (batch={batch_size}, seq_len={seq_len})"
            )
            raise NotImplemented


# Register the model with vLLM
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


def load_weights(self, weights):
    """Load weights for FlashHead model."""
    params_dict = dict(self.named_parameters())

    for name, loaded_weight in weights:
        # Skip FlashHead-specific buffers as they're loaded from cache
        if (
            "vocab_maps_tensor" in name
            or "centroids" in name
            or "cluster_linear" in name
        ):
            print(
                f"[DEBUG] Skipping weight named {name} with shape {weights.shape}"
            )

            continue

        if name in params_dict:
            param = params_dict[name]
            default_weight_loader(param, loaded_weight)


FlashHeadLlamaForCausalLM.load_weights = load_weights


class FlashHeadLogitsProcessor(LogitsProcessor):
    """Custom logits processor that uses FlashHead for optimized token generation.

    Since FlashHead returns tokens directly instead of logits, this processor
    creates a "fake" logits tensor that vLLM's sampler can process correctly.
    """

    @classmethod
    def validate_params(cls, params: SamplingParams):
        """Validate sampling parameters for this processor.

        Args:
            params: SamplingParams to validate

        Raises:
            ValueError: If parameters are invalid
        """
        use_flash_head = params.extra_args and params.extra_args.get(
            "use_flash_head"
        )
        if use_flash_head is not None and not isinstance(use_flash_head, bool):
            raise ValueError(
                f"use_flash_head value {use_flash_head} is not bool"
            )

        scale = params.extra_args and params.extra_args.get("logits_scale")
        if scale is not None and not isinstance(scale, (int, float)):
            raise ValueError(f"logits_scale value {scale} is not numeric")

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device = "mps",
        is_pin_memory: bool = True,
    ):
        """Initialize the FlashHead logits processor.

        Args:
            vllm_config: vLLM configuration
            device: Device to run on
            is_pin_memory: Whether to pin memory
        """
        self.device = device
        self.is_pin_memory = is_pin_memory
        self.vllm_config = vllm_config

        # Store per-request configuration
        # Maps request_id -> dict of config options
        self.req_config: dict[int, dict] = {}

        # Reference to the model's FlashHead (will be set by model)
        self.flash_head_instance = None

        print("[FlashHeadLogitsProcessor] Initialized")

    def set_flash_head(self, flash_head):
        """Set the FlashHead instance from the model.

        Args:
            flash_head: FlashHead module instance
        """
        self.flash_head_instance = flash_head
        print(
            f"[FlashHeadLogitsProcessor] FlashHead instance set: {flash_head is not None}"
        )

    def is_argmax_invariant(self) -> bool:
        """Whether this processor affects greedy sampling.

        Returns:
            False, as this processor fundamentally changes token selection
        """
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        """Update internal state based on batch changes.

        Args:
            batch_update: Information about added/removed/moved requests
        """
        if not batch_update:
            return

        # Process added requests
        for index, params, _, _ in batch_update.added:
            if params is None:
                continue

            self.validate_params(params)

            # Extract custom parameters from extra_args
            config = {}
            if params.extra_args:
                config["use_flash_head"] = params.extra_args.get(
                    "use_flash_head", True
                )
                config["logits_scale"] = params.extra_args.get(
                    "logits_scale", 1.0
                )
                config["do_sample"] = params.temperature > 0
                config["temperature"] = params.temperature
            else:
                # Default configuration
                config["use_flash_head"] = True
                config["logits_scale"] = 1.0
                config["do_sample"] = params.temperature > 0
                config["temperature"] = params.temperature

            self.req_config[index] = config

        if self.req_config:
            # Process removed requests
            for index in batch_update.removed:
                self.req_config.pop(index, None)

            # Process moved requests
            for adx, bdx, direct in batch_update.moved:
                a_val = self.req_config.pop(adx, None)
                b_val = self.req_config.pop(bdx, None)

                if a_val is not None:
                    self.req_config[bdx] = a_val

                if direct == MoveDirectionality.SWAP and b_val is not None:
                    self.req_config[adx] = b_val

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply logits processing or use FlashHead for token selection.

        This is where the magic happens. If we detect that FlashHead was used
        (indicated by specific logits tensor characteristics), we handle it.
        Otherwise, we apply standard logits transformations.

        Args:
            logits: Logits tensor of shape (batch_size, vocab_size)

        Returns:
            Modified logits tensor
        """
        if not self.req_config:
            return logits

        # Check if this is a FlashHead output (single token ID)
        # FlashHead returns shape (1, 1) with token ID
        if logits.shape == (1, 1) and logits.dtype == torch.int64:
            # This is a token ID from FlashHead, convert to one-hot logits
            token_id = logits[0, 0].item()

            # Get vocab size from model config
            vocab_size = self.vllm_config.model_config.hf_config.vocab_size

            # Create one-hot logits: very high value for selected token, -inf for others
            fake_logits = torch.full(
                (1, vocab_size),
                float("-inf"),
                dtype=torch.float32,
                device=logits.device,
            )
            fake_logits[0, token_id] = 100.0  # High logit for selected token

            print(
                f"[FlashHeadLogitsProcessor] Converted token {token_id} to fake logits"
            )
            return fake_logits

        # Standard logits processing for non-FlashHead outputs
        for req_idx, config in self.req_config.items():
            if req_idx >= logits.shape[0]:
                continue

            # Apply scaling if needed
            scale = config.get("logits_scale", 1.0)
            if scale != 1.0:
                logits[req_idx] = logits[req_idx] * scale

        return logits

    def can_use_flash_head(self, batch_size: int, seq_len: int) -> bool:
        """Check if FlashHead can be used for this batch.

        Args:
            batch_size: Current batch size
            seq_len: Sequence length

        Returns:
            True if FlashHead is available and conditions are met
        """
        if self.flash_head_instance is None:
            return False

        # FlashHead optimization works best for single-token generation
        if batch_size > 1 or seq_len > 1:
            return False

        # Check if any active request wants to use FlashHead
        return any(
            config.get("use_flash_head", True)
            for config in self.req_config.values()
        )


def create_fake_logits_from_token(
    token_id: int,
    vocab_size: int,
    device: torch.device,
    batch_size: int = 1,
) -> torch.Tensor:
    """Create a fake logits tensor from a single token ID.

    This creates a logits tensor where the selected token has a very high
    logit value and all others have -inf, ensuring the sampler selects
    the correct token.

    Args:
        token_id: The token ID to encode
        vocab_size: Total vocabulary size
        device: Device for the tensor
        batch_size: Batch size (default 1)

    Returns:
        Fake logits tensor of shape (batch_size, vocab_size)
    """
    fake_logits = torch.full(
        (batch_size, vocab_size),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    fake_logits[:, token_id] = 100.0
    return fake_logits
