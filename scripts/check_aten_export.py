# Copyright (C) 2025 Embedl AB
"""Check whether popular HuggingFace models export to an ATen graph.

This script exercises the HuggingFace ``transformers.exporters`` pipeline
(https://huggingface.co/docs/transformers/main/en/exporters). The
``DynamoExporter`` wraps ``torch.export`` and produces an ``ExportedProgram``
whose ``graph_module`` is a graph of **ATen** operators -- i.e. the "ATen
graph" we verify here.

Everything runs on the PyTorch **CPU** backend.

For every model we:

1. Load the real HuggingFace model (real weights -- these are all small
   enough to fit on a CPU box) with ``attn_implementation="sdpa"`` (falling
   back to ``eager``; flash/flex attention is not exportable).
2. Build a representative sample input with the model's own processor.
3. Run an eager forward pass to get a reference output.
4. Export to an ATen ``ExportedProgram`` via ``DynamoExporter``.
5. Run a forward pass **on the exported ATen graph** and check it matches
   the eager reference.

The default model set covers the categories asked for:

* Vision backbones / segmentation:  SAM, DINOv2, DINO (ViT)
* Autonomous-driving perception:     DETR, Grounding DINO, OWLv2 (detection)
                                     Depth-Anything-V2, DPT (monocular depth)
* Robotics VLAs:                     GR00T, SmolVLA  (see notes -- these are
                                     not transformers ``PreTrainedModel``s)

Note on BEV / robotics: true BEV models (BEVFormer, BEVFusion, PETR) and
robot policies (pi0, GR00T, SmolVLA) are not packaged as transformers
``PreTrainedModel``s, so the *transformers* exporter cannot load them -- they
would need raw ``torch.export`` on the underlying ``nn.Module``. We still list
the robotics VLAs so the report records exactly why they are out of scope.

Usage::

    python scripts/check_aten_export.py                    # curated set
    python scripts/check_aten_export.py --only sam dinov2
    python scripts/check_aten_export.py --trending --top 10  # trending LLMs
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from dataclasses import dataclass, asdict
from typing import Any, Callable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HF_API = "https://huggingface.co/api/models"


# ---------------------------------------------------------------------------
# Curated model registry
# ---------------------------------------------------------------------------
# Each spec:  key -> dict(id, category, auto, inputs, note)
#   auto    : transformers auto-class used to load the model
#   inputs  : how to build a sample input ("image", "grounding_dino",
#             "owlv2", "sam")
SPECS: dict[str, dict[str, str]] = {
    # --- vision backbones / segmentation ------------------------------------
    "sam": dict(id="facebook/sam-vit-base", category="vision-segmentation",
                auto="AutoModelForMaskGeneration", inputs="sam"),
    "dinov2": dict(id="facebook/dinov2-base", category="vision-backbone",
                   auto="AutoModel", inputs="image"),
    "dino": dict(id="facebook/dino-vitb16", category="vision-backbone",
                 auto="AutoModel", inputs="image"),
    # --- autonomous-driving perception: detection ---------------------------
    "detr": dict(id="facebook/detr-resnet-50", category="av-detection",
                 auto="AutoModelForObjectDetection", inputs="image"),
    "grounding_dino": dict(id="IDEA-Research/grounding-dino-tiny",
                           category="av-detection",
                           auto="AutoModelForZeroShotObjectDetection",
                           inputs="grounding_dino"),
    "owlv2": dict(id="google/owlv2-base-patch16-ensemble",
                  category="av-detection",
                  auto="AutoModelForZeroShotObjectDetection", inputs="owlv2"),
    # --- autonomous-driving perception: monocular depth ---------------------
    "depth_anything": dict(id="depth-anything/Depth-Anything-V2-Small-hf",
                           category="av-depth",
                           auto="AutoModelForDepthEstimation", inputs="image"),
    "dpt": dict(id="Intel/dpt-hybrid-midas", category="av-depth",
                auto="AutoModelForDepthEstimation", inputs="image"),
    # --- DETR-family object detectors (RF-DETR and relatives) ---------------
    "rf_detr": dict(id="Roboflow/rf-detr-nano", category="detr-family",
                    auto="AutoModelForObjectDetection", inputs="image",
                    note="Roboflow RF-DETR (real-time, DINOv2 backbone)"),
    "rtdetr": dict(id="PekingU/rtdetr_r50vd", category="detr-family",
                   auto="AutoModelForObjectDetection", inputs="image",
                   note="RT-DETR real-time detector"),
    "rtdetrv2": dict(id="PekingU/rtdetr_v2_r18vd", category="detr-family",
                     auto="AutoModelForObjectDetection", inputs="image",
                     note="RT-DETRv2 (r18 backbone)"),
    "deformable_detr": dict(id="SenseTime/deformable-detr",
                            category="detr-family",
                            auto="AutoModelForObjectDetection", inputs="image",
                            note="Deformable DETR (multi-scale deform attn)"),
    "conditional_detr": dict(id="microsoft/conditional-detr-resnet-50",
                             category="detr-family",
                             auto="AutoModelForObjectDetection", inputs="image",
                             note="Conditional DETR"),
    "deta": dict(id="jozhang97/deta-resnet-50", category="detr-family",
                 auto="AutoModelForObjectDetection", inputs="image",
                 note="DETA (deformable, NMS-free two-stage)"),
    "yolos": dict(id="hustvl/yolos-tiny", category="detr-family",
                  auto="AutoModelForObjectDetection", inputs="image",
                  note="YOLOS (ViT-based DETR)"),
    "table_transformer": dict(id="microsoft/table-transformer-detection",
                              category="detr-family",
                              auto="AutoModelForObjectDetection", inputs="image",
                              note="Table Transformer (DETR variant)"),
    # --- robotics VLAs (expected out-of-scope for the transformers exporter) -
    "gr00t": dict(id="nvidia/GR00T-N1.5-3B", category="robotics-vla",
                  auto="AutoModel", inputs="image",
                  note="NVIDIA GR00T robot foundation model"),
    "smolvla": dict(id="lerobot/smolvla_base", category="robotics-vla",
                    auto="AutoModel", inputs="image",
                    note="LeRobot SmolVLA policy"),
}


@dataclass
class Result:
    key: str
    model_id: str
    category: str
    architecture: str | None = None
    loaded: bool = False
    inputs_built: bool = False
    eager_forward: bool = False
    aten_exported: bool = False
    aten_forward: bool = False
    outputs_match: bool = False
    num_aten_ops: int | None = None
    graph_file: str | None = None
    error: str | None = None
    error_stage: str | None = None


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------
def _random_image(h: int = 480, w: int = 640):
    from PIL import Image
    rng = np.random.RandomState(0)
    return Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8))


def _build_inputs(kind: str, model_id: str) -> dict[str, torch.Tensor]:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    img = _random_image()
    if kind == "image":
        enc = processor(images=img, return_tensors="pt")
    elif kind == "sam":
        enc = processor(images=img, input_points=[[[320, 240]]],
                        return_tensors="pt")
    elif kind == "grounding_dino":
        enc = processor(images=img, text="a car. a pedestrian.",
                        return_tensors="pt")
    elif kind == "owlv2":
        enc = processor(text=[["a photo of a car", "a photo of a pedestrian"]],
                        images=img, return_tensors="pt")
    else:
        raise ValueError(f"unknown input kind {kind}")
    return dict(enc)


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------
def _first_tensor(obj: Any) -> torch.Tensor | None:
    if isinstance(obj, torch.Tensor):
        return obj
    if hasattr(obj, "to_tuple"):
        obj = obj.to_tuple()
    if isinstance(obj, (list, tuple)):
        for o in obj:
            t = _first_tensor(o)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for o in obj.values():
            t = _first_tensor(o)
            if t is not None:
                return t
    return None


def load_curated_model(spec: dict[str, str]):
    """Load a curated model (real weights, CPU/float32) and its sample inputs.

    Returns ``(model, inputs_dict)``.  Reused by both the ATen-export check and
    the embedl-deploy transform/quantize check.
    """
    import transformers

    auto_cls = getattr(transformers, spec["auto"])
    try:
        model = auto_cls.from_pretrained(
            spec["id"], trust_remote_code=True,
            torch_dtype=torch.float32, attn_implementation="sdpa")
    except (ValueError, TypeError):
        model = auto_cls.from_pretrained(
            spec["id"], trust_remote_code=True,
            torch_dtype=torch.float32, attn_implementation="eager")
    model = model.eval().to("cpu")

    inputs = _build_inputs(spec["inputs"], spec["id"])
    inputs = {k: (v.to(torch.float32) if torch.is_floating_point(v) else v)
              for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    return model, inputs


def load_tiny_llm(model_id: str):
    """Build a downsized random variant of a (possibly huge) LLM + inputs.

    Returns ``(model, inputs_dict)``.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    _shrink_llm_config(cfg)
    model = None
    for attn in ("sdpa", "eager"):
        try:
            model = AutoModelForCausalLM.from_config(
                cfg, trust_remote_code=True, attn_implementation=attn)
            break
        except ValueError:
            continue
    if model is None:
        raise RuntimeError("could not instantiate model with sdpa or eager")
    model = model.to(torch.float32).eval()
    vocab = min(getattr(cfg, "vocab_size", 512), 512)
    ids = torch.randint(1, vocab, (1, 8))
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    return model, inputs


def check_model(key: str, spec: dict[str, str], out_dir: str) -> Result:
    from transformers.exporters import DynamoExporter, DynamoConfig

    res = Result(key=key, model_id=spec["id"], category=spec["category"])

    # ---- Stage 1+2: load model (real weights, CPU, float32) + inputs -----
    try:
        model, inputs = load_curated_model(spec)
        res.loaded = True
        res.inputs_built = True
        res.architecture = type(model).__name__
    except Exception as e:  # noqa: BLE001
        res.error_stage = "load"
        res.error = f"{type(e).__name__}: {e}"
        return res

    # ---- Stage 3: eager reference forward --------------------------------
    try:
        with torch.no_grad():
            ref = model(**inputs)
        ref_t = _first_tensor(ref)
        res.eager_forward = True
    except Exception as e:  # noqa: BLE001
        res.error_stage = "eager_forward"
        res.error = f"{type(e).__name__}: {e}"
        return res

    # ---- Stage 4: export to an ATen graph --------------------------------
    try:
        exporter = DynamoExporter()
        exported = exporter.export(model, dict(inputs),
                                   config=DynamoConfig(dynamic=False))
        res.aten_exported = True
        gm = exported.graph_module
        res.num_aten_ops = sum(1 for n in gm.graph.nodes
                               if n.op == "call_function")
        os.makedirs(out_dir, exist_ok=True)
        graph_path = os.path.join(out_dir, key + ".aten_graph.txt")
        with open(graph_path, "w") as fh:
            fh.write(gm.print_readable(print_output=False))
        res.graph_file = graph_path
    except Exception as e:  # noqa: BLE001
        res.error_stage = "export"
        res.error = f"{type(e).__name__}: {e}"
        return res

    # ---- Stage 5: forward on the exported ATen graph ---------------------
    try:
        with torch.no_grad():
            out = exported.module()(**inputs)
        out_t = _first_tensor(out)
        res.aten_forward = True
        if ref_t is not None and out_t is not None and ref_t.shape == out_t.shape:
            res.outputs_match = torch.allclose(ref_t, out_t, atol=1e-3, rtol=1e-3)
    except Exception as e:  # noqa: BLE001
        res.error_stage = "aten_forward"
        res.error = f"{type(e).__name__}: {e}"
        return res

    return res


# ---------------------------------------------------------------------------
# Trending-LLM mode (kept from the original exercise)
# ---------------------------------------------------------------------------
def run_trending(top: int, out_dir: str) -> list[Result]:
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.exporters import DynamoExporter, DynamoConfig

    url = f"{HF_API}?sort=trendingScore&limit=100&full=false"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    picked = [m for m in data if m.get("pipeline_tag") == "text-generation"
              and "gguf" not in m.get("id", "").lower()][:top]

    results = []
    for i, m in enumerate(picked):
        mid = m["id"]
        print(f"\n[{i + 1}/{len(picked)}] {mid}")
        res = Result(key=mid, model_id=mid, category="trending-llm")
        try:
            cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)
            res.architecture = (cfg.architectures or [None])[0]
            _shrink_llm_config(cfg)
            for attn in ("sdpa", "eager"):
                try:
                    model = AutoModelForCausalLM.from_config(
                        cfg, trust_remote_code=True, attn_implementation=attn)
                    break
                except ValueError:
                    continue
            model = model.to(torch.float32).eval()
            res.loaded = True
            vocab = min(getattr(cfg, "vocab_size", 512), 512)
            ids = torch.randint(1, vocab, (1, 8))
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            res.inputs_built = True
            with torch.no_grad():
                ref = model(**inputs)
            ref_t = _first_tensor(ref)
            res.eager_forward = True
            exported = DynamoExporter().export(model, dict(inputs),
                                               config=DynamoConfig(dynamic=False))
            res.aten_exported = True
            gm = exported.graph_module
            res.num_aten_ops = sum(1 for n in gm.graph.nodes
                                   if n.op == "call_function")
            os.makedirs(out_dir, exist_ok=True)
            gp = os.path.join(out_dir, mid.replace("/", "__") + ".aten_graph.txt")
            with open(gp, "w") as fh:
                fh.write(gm.print_readable(print_output=False))
            res.graph_file = gp
            with torch.no_grad():
                out = exported.module()(**inputs)
            out_t = _first_tensor(out)
            res.aten_forward = True
            res.outputs_match = torch.allclose(ref_t, out_t, atol=1e-3, rtol=1e-3)
        except Exception as e:  # noqa: BLE001
            res.error = f"{type(e).__name__}: {e}"
        results.append(res)
        _print_line(res)
    return results


def _shrink_llm_config(cfg: Any) -> None:
    if getattr(cfg, "vocab_size", None):
        cfg.vocab_size = min(cfg.vocab_size, 512)
        for tok in ("pad_token_id", "bos_token_id", "eos_token_id"):
            v = getattr(cfg, tok, None)
            if isinstance(v, int) and v >= cfg.vocab_size:
                setattr(cfg, tok, 0)
    if getattr(cfg, "num_attention_heads", None):
        cfg.num_attention_heads = min(cfg.num_attention_heads, 4)
        if getattr(cfg, "num_key_value_heads", None):
            cfg.num_key_value_heads = min(cfg.num_key_value_heads, 2)
        hd = min(getattr(cfg, "head_dim", None) or 16, 16)
        if getattr(cfg, "head_dim", None):
            cfg.head_dim = hd
        if getattr(cfg, "hidden_size", None):
            cfg.hidden_size = cfg.num_attention_heads * hd
    for attr, val in {"num_hidden_layers": 2, "intermediate_size": 128,
                      "num_experts": 4, "num_local_experts": 4,
                      "n_routed_experts": 4, "moe_intermediate_size": 64,
                      "num_experts_per_tok": 2, "first_k_dense_replace": 0}.items():
        if getattr(cfg, attr, None) is not None:
            setattr(cfg, attr, min(getattr(cfg, attr), val))
    if getattr(cfg, "quantization_config", None) is not None:
        cfg.quantization_config = None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_line(res: Result) -> None:
    status = "MATCH" if res.outputs_match else (
        "EXPORTED" if res.aten_exported else f"FAIL@{res.error_stage or 'run'}")
    print(f"    arch={res.architecture} cat={res.category} -> {status}")
    if res.aten_exported:
        print(f"    ATen ops: {res.num_aten_ops}  graph: {res.graph_file}")
    if res.error:
        print(f"    error: {res.error.splitlines()[0][:200]}")


def _summary(results: list[Result]) -> None:
    print("\n" + "=" * 104)
    print("SUMMARY  (load->inputs->eager->export->aten-forward->match)")
    print("=" * 104)
    hdr = (f"{'model':40} {'category':18} {'arch':22} "
           f"{'ld':3}{'in':3}{'eg':3}{'ex':3}{'fw':3}{'mt':3}{'ops':>6}")
    print(hdr)
    print("-" * len(hdr))

    def f(b: bool) -> str:
        return " y " if b else " . "

    for r in results:
        print(f"{r.model_id[:40]:40} {r.category[:18]:18} "
              f"{str(r.architecture)[:22]:22} "
              f"{f(r.loaded)}{f(r.inputs_built)}{f(r.eager_forward)}"
              f"{f(r.aten_exported)}{f(r.aten_forward)}{f(r.outputs_match)}"
              f"{(r.num_aten_ops or ''):>6}")
    print("-" * len(hdr))
    n_exp = sum(r.aten_exported for r in results)
    n_ok = sum(r.outputs_match for r in results)
    print(f"{n_exp}/{len(results)} exported to an ATen graph; "
          f"{n_ok}/{len(results)} ran a matching forward on the ATen graph.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of registry keys to run")
    ap.add_argument("--trending", action="store_true",
                    help="run trending text-generation LLMs instead")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out-dir", default="aten_graphs")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)

    if args.trending:
        results = run_trending(args.top, args.out_dir)
    else:
        keys = args.only or list(SPECS)
        results = []
        for i, key in enumerate(keys):
            spec = SPECS[key]
            print(f"\n[{i + 1}/{len(keys)}] {key}: {spec['id']}")
            res = check_model(key, spec, args.out_dir)
            results.append(res)
            _print_line(res)

    _summary(results)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
        print(f"\nWrote JSON results to {args.json_out}")


if __name__ == "__main__":
    main()
