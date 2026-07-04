# Copyright (C) 2025 Embedl AB
"""Run ``embedl-deploy`` transform + quantize on popular HuggingFace models.

Companion to ``check_aten_export.py``.  Where that script verifies the models
export to an ATen graph via HuggingFace's ``transformers.exporters``, this one
feeds the *same* models through the Embedl Deploy API
(https://docs.embedl.com/embedl-deploy/latest/guide/quantization.html):

* ``embedl_deploy.transform(model, args)`` traces the model with
  ``torch.export`` and returns a fused **ATen** ``GraphModule``.
* ``embedl_deploy.quantize.quantize(gm, args, config, forward_loop=...)`` runs
  INT8 post-training quantization (weight + activation) with a calibration
  forward loop, returning a fake-quantized ``GraphModule``.

For every model we check, on CPU:

1. transform          -> fused ATen graph runs a forward
2. quantize (INT8)    -> quantized graph runs a forward
3. report the max abs / relative deviation of the quantized output vs fp32

The model set mirrors the request: the vision / autonomous-driving perception
models plus a few of the trending top-10 LLMs (built as tiny random variants).

Usage::

    python scripts/check_embedl_quantize.py
    python scripts/check_embedl_quantize.py --vision-only
    python scripts/check_embedl_quantize.py --precision int4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass, asdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_aten_export import (  # noqa: E402
    SPECS, load_curated_model, load_tiny_llm, _first_tensor,
)

from embedl_deploy import transform  # noqa: E402
from embedl_deploy.quantize import (  # noqa: E402
    QuantConfig, TensorQuantConfig, Precision, quantize,
)

# Vision / AV-perception models (real weights) that export to ATen.
VISION_KEYS = ["sam", "dinov2", "dino", "detr", "grounding_dino",
               "owlv2", "depth_anything", "dpt"]

# A few of the trending top-10 LLMs (built as tiny random variants), covering
# the distinct architectures that exported cleanly.
TRENDING_LLMS = [
    "Qwen/Qwen3-0.6B",                             # clean Qwen3 baseline
    "empero-ai/Qwythos-9B-Claude-Mythos-5-1M",     # Qwen3_5 (dense)
    "Qwen/Qwen-AgentWorld-35B-A3B",                # Qwen3_5Moe (MoE)
]


@dataclass
class QResult:
    name: str
    kind: str
    architecture: str | None = None
    loaded: bool = False
    fp_forward: bool = False
    transformed: bool = False
    transform_forward: bool = False
    transform_ops: int | None = None
    quantized: bool = False
    quant_forward: bool = False
    fq_nodes: int | None = None
    max_abs_diff: float | None = None
    rel_diff: float | None = None
    error: str | None = None
    error_stage: str | None = None


class PositionalWrapper(nn.Module):
    """Adapt a HF model (keyword inputs, structured output) to a flat
    positional-in / single-tensor-out module that torch.export likes."""

    def __init__(self, model: nn.Module, keys: list[str]):
        super().__init__()
        self.model = model
        self.keys = list(keys)

    def forward(self, *args):
        out = self.model(**dict(zip(self.keys, args)))
        return _first_tensor(out)


def _precision(name: str) -> Precision:
    return {"int8": Precision.INT8, "int4": Precision.INT4,
            "int16": Precision.INT16}[name]


def run_one(name: str, kind: str, model: nn.Module,
            inputs: dict, precision: str) -> QResult:
    res = QResult(name=name, kind=kind, architecture=type(model).__name__,
                  loaded=True)
    keys = list(inputs.keys())
    args = tuple(inputs[k] for k in keys)
    wrapper = PositionalWrapper(model, keys).eval()

    # ---- fp32 reference --------------------------------------------------
    try:
        with torch.no_grad():
            ref = wrapper(*args)
        res.fp_forward = ref is not None
    except Exception as e:  # noqa: BLE001
        res.error_stage, res.error = "fp_forward", f"{type(e).__name__}: {e}"
        return res

    # ---- transform: fuse into an ATen GraphModule ------------------------
    try:
        tr = transform(wrapper, args)
        gm = tr.model
        res.transformed = True
        res.transform_ops = sum(1 for n in gm.graph.nodes
                                if n.op == "call_function")
        with torch.no_grad():
            t_out = gm(*args)
        res.transform_forward = t_out is not None
    except Exception as e:  # noqa: BLE001
        res.error_stage, res.error = "transform", f"{type(e).__name__}: {e}"
        return res

    # ---- quantize (PTQ, INT8 weight + activation) ------------------------
    try:
        prec = _precision(precision)
        config = QuantConfig(
            activation=TensorQuantConfig(prec, symmetric=True, per_channel=False),
            weight=TensorQuantConfig(prec, symmetric=True, per_channel=True),
        )

        def forward_loop(m):
            with torch.no_grad():
                for _ in range(4):
                    m(*args)

        qgm = quantize(gm, args, config,
                       forward_loop=forward_loop, freeze_weights=True)
        res.quantized = True
        res.fq_nodes = sum(1 for n in qgm.graph.nodes
                           if "fake_quant" in str(n.target).lower()
                           or "quant" in str(n.target).lower())
        with torch.no_grad():
            q_out = qgm(*args)
        res.quant_forward = q_out is not None
        if isinstance(ref, torch.Tensor) and isinstance(q_out, torch.Tensor) \
                and ref.shape == q_out.shape:
            diff = (ref.float() - q_out.float()).abs()
            res.max_abs_diff = float(diff.max())
            denom = float(ref.float().abs().max()) + 1e-8
            res.rel_diff = float(diff.max()) / denom
    except Exception as e:  # noqa: BLE001
        res.error_stage, res.error = "quantize", f"{type(e).__name__}: {e}"
        return res

    return res


def _print_line(r: QResult) -> None:
    tags = []
    tags.append("transform=y" if res_ok(r.transform_forward) else "transform=.")
    tags.append("quant=y" if r.quant_forward else "quant=.")
    print(f"    arch={r.architecture} -> {' '.join(tags)}"
          + (f"  fp/int-maxdiff={r.max_abs_diff:.4g} rel={r.rel_diff:.3g}"
             if r.max_abs_diff is not None else ""))
    if r.error:
        print(f"    error@{r.error_stage}: {r.error.splitlines()[0][:200]}")


def res_ok(b) -> bool:
    return bool(b)


def _summary(results: list[QResult]) -> None:
    print("\n" + "=" * 104)
    print("EMBEDL-DEPLOY  transform + quantize  (fp32 -> ATen graph -> INT PTQ)")
    print("=" * 104)
    hdr = (f"{'model':40} {'kind':6} {'arch':24} "
           f"{'tf':3}{'tf-fwd':7}{'quant':6}{'q-fwd':6}{'rel-diff':>10}")
    print(hdr)
    print("-" * len(hdr))

    def f(b) -> str:
        return " y " if b else " . "

    for r in results:
        rel = f"{r.rel_diff:.3g}" if r.rel_diff is not None else ""
        print(f"{r.name[:40]:40} {r.kind[:6]:6} {str(r.architecture)[:24]:24} "
              f"{f(r.transformed)}{f(r.transform_forward):^7}"
              f"{f(r.quantized):^6}{f(r.quant_forward):^6}{rel:>10}")
    print("-" * len(hdr))
    nt = sum(r.transform_forward for r in results)
    nq = sum(r.quant_forward for r in results)
    print(f"{nt}/{len(results)} transformed graphs ran a forward; "
          f"{nq}/{len(results)} quantized graphs ran a forward.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vision-only", action="store_true")
    ap.add_argument("--llm-only", action="store_true")
    ap.add_argument("--keys", nargs="*", default=None,
                    help="explicit SPECS keys to run (image-input models)")
    ap.add_argument("--precision", default="int8",
                    choices=["int8", "int4", "int16"])
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    results: list[QResult] = []

    if args.keys:
        vision, llms = args.keys, []
    else:
        vision = [] if args.llm_only else VISION_KEYS
        llms = [] if args.vision_only else TRENDING_LLMS

    n_total = len(vision) + len(llms)
    idx = 0
    for key in vision:
        idx += 1
        spec = SPECS[key]
        print(f"\n[{idx}/{n_total}] {key}: {spec['id']}")
        try:
            model, inputs = load_curated_model(spec)
        except Exception as e:  # noqa: BLE001
            r = QResult(name=spec["id"], kind="vision")
            r.error_stage, r.error = "load", f"{type(e).__name__}: {e}"
            results.append(r)
            print(f"    FAIL@load: {r.error.splitlines()[0][:160]}")
            continue
        r = run_one(spec["id"], "vision", model, inputs, args.precision)
        results.append(r)
        _print_line(r)
        del model, inputs
        gc.collect()

    for mid in llms:
        idx += 1
        print(f"\n[{idx}/{n_total}] llm: {mid}")
        try:
            model, inputs = load_tiny_llm(mid)
        except Exception as e:  # noqa: BLE001
            r = QResult(name=mid, kind="llm")
            r.error_stage, r.error = "load", f"{type(e).__name__}: {e}"
            results.append(r)
            print(f"    FAIL@load: {r.error.splitlines()[0][:160]}")
            continue
        r = run_one(mid, "llm", model, inputs, args.precision)
        results.append(r)
        _print_line(r)
        del model, inputs
        gc.collect()

    _summary(results)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
        print(f"\nWrote JSON results to {args.json_out}")


if __name__ == "__main__":
    main()
