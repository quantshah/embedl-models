# Copyright (C) 2025 Embedl AB
"""Diagnose (and work around) the Embedl-Deploy `transform()` failure on RT-DETR.

`embedl_deploy.transform()` / `quantize()` fail on RT-DETR and RT-DETRv2 with::

    KeyError: add_89

while the same models export cleanly to a HuggingFace ATen graph and run a
matching forward. This script pins down the cause and demonstrates a safe
work-around.

Root cause (embedl-deploy 0.8.0, recomposition phase)
-----------------------------------------------------
`transform()` -> `prepare_graph()` -> `export_trace()` first exports the model
to an aten graph, then applies `RECOMPOSITION_PATTERNS` via
`apply_transformation_plan()`. Patterns are applied sequentially inside one
`ReplaceSession`. Two things combine:

1. `apply_transformation_plan()` only guards against *tree-node* overlap
   between enabled matches (`tree_match.get_tree_nodes()`); it does **not**
   consider the *input* nodes a match consumes.
2. `AtenConv2dPattern` (registry index 4) is applied before
   `AtenActivationPattern` (index 6). A residual/bias `add` node
   (`add_89`, `aten.add.Tensor`) that feeds a `sigmoid` is detached and erased
   during the earlier pattern's application **without being recorded in the
   `ReplaceSession` replaced-node map**.

When `AtenActivationPattern` then recomposes that `sigmoid` (`nn.Sigmoid`),
`replace_tree()` -> `_insert_module()` computes
`max(args, key=_graph_order(gm))` over its input args. `add_89` is now absent
from both the graph and the remap, so `_graph_order`'s dict lookup raises
`KeyError: add_89` instead of the stale input being remapped or the match
being skipped.

RT-DETR triggers this because its decoder applies `sigmoid` to
reference-point / query-selection tensors that share an `add` producer with a
conv/linear that recomposition rewrites. Plain DETR, RF-DETR, YOLOS, Deformable
and Conditional DETR do not have that exact shared-input shape, so they pass.

A second, independent latent failure surfaces once the first is bypassed:
`AtenLinearPattern._make_linear` asserts `isinstance(args[1], fx.Node)` on some
RT-DETR linears (weight arg not a plain `get_attr`).

Work-around
-----------
Both failures raise *before* mutating the graph (during replacement insertion /
graft construction), so a single failing recomposition match can be skipped
safely -- recomposition is an optimisation-enabling hint, not a correctness
requirement. `--fix` installs a shim that wraps `Pattern.replace` to skip an
individually-failing match instead of aborting the whole transform; RT-DETR
then transforms, INT8-quantizes, and runs a forward like the other detectors.

Usage::

    python scripts/rtdetr_embedl_diagnosis.py            # reproduce + diagnose
    python scripts/rtdetr_embedl_diagnosis.py --fix      # apply shim, show pass
    python scripts/rtdetr_embedl_diagnosis.py --model rtdetr --fix
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch  # noqa: E402
from torch import fx  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_aten_export import SPECS, load_curated_model  # noqa: E402
from check_embedl_quantize import PositionalWrapper, run_one  # noqa: E402


def diagnose(key: str) -> None:
    """Reproduce the failure and print which pattern/node causes it."""
    import embedl_deploy._internal.core.tree.replace as R
    import embedl_deploy._internal.core.tree.state as S
    import embedl_deploy._internal.core.patterns.main as PM
    from embedl_deploy import transform

    cur = {"pat": None}
    orig_replace = PM.Pattern.replace.__func__

    def traced_replace(cls, pm):
        cur["pat"] = pm.pattern.__name__
        return orig_replace(cls, pm)

    PM.Pattern.replace = classmethod(traced_replace)

    orig_insert = R._insert_module

    def traced_insert(gm, module, args):
        try:
            return orig_insert(gm, module, args)
        except KeyError:
            in_graph = set(gm.graph.nodes)
            rmap = S.get_replaced_nodes()
            print("\n  >>> transform() aborted in recomposition phase")
            print(f"      failing pattern : {cur['pat']}")
            print(f"      recomposing into: {type(module).__name__}")
            for a in args:
                print(f"      input arg       : {a.name} "
                      f"(op={a.op}, target={getattr(a, 'target', None)})")
                print(f"                        in_graph={a in in_graph}  "
                      f"in_replaced_map={a in rmap}  users={list(a.users)}")
            raise

    R._insert_module = traced_insert

    model, inputs = load_curated_model(SPECS[key])
    keys = list(inputs.keys())
    args = tuple(inputs[k] for k in keys)
    wrapper = PositionalWrapper(model, keys).eval()
    print(f"[{key}] {SPECS[key]['id']} — calling embedl_deploy.transform() ...")
    try:
        transform(wrapper, args)
        print("  (no error — nothing to diagnose)")
    except Exception as e:  # noqa: BLE001
        print(f"\n  {type(e).__name__}: {e}")
        print("\n  Root cause: a recomposition match references an input node "
              "that an\n  earlier pattern erased without recording it in the "
              "ReplaceSession\n  remap. apply_transformation_plan()'s overlap "
              "guard only checks each\n  match's tree nodes, not its shared "
              "input nodes, and _insert_module()'s\n  "
              "max(args, key=_graph_order(gm)) assumes every input is still "
              "present.")


def install_resilience_shim() -> dict[str, int]:
    """Wrap Pattern.replace so an individually-failing match is skipped.

    Both RT-DETR recomposition failures raise before mutating the graph, so
    skipping the match leaves the graph valid. Returns a dict counting how many
    matches of each pattern were skipped.
    """
    import embedl_deploy._internal.core.patterns.main as PM

    skipped: dict[str, int] = {}
    orig = PM.Pattern.replace.__func__

    def safe_replace(cls, pm):
        try:
            return orig(cls, pm)
        except Exception:  # noqa: BLE001
            skipped[pm.pattern.__name__] = skipped.get(pm.pattern.__name__, 0) + 1
            return []

    PM.Pattern.replace = classmethod(safe_replace)
    return skipped


def run_fixed(key: str) -> None:
    skipped = install_resilience_shim()
    model, inputs = load_curated_model(SPECS[key])
    r = run_one(SPECS[key]["id"], "detr", model, inputs, "int8")
    print(f"[{key}] with resilience shim:")
    print(f"    transform : {r.transformed}  (fused ops: {r.transform_ops})")
    print(f"    quantize  : {r.quantized}  (fake-quant nodes: {r.fq_nodes})")
    print(f"    quant fwd : {r.quant_forward}  rel-diff: {r.rel_diff}")
    print(f"    skipped recomposition matches: {skipped}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="rtdetrv2",
                    choices=["rtdetr", "rtdetrv2"])
    ap.add_argument("--fix", action="store_true",
                    help="install the resilience shim and show it pass")
    args = ap.parse_args()
    torch.manual_seed(0)
    if args.fix:
        run_fixed(args.model)
    else:
        diagnose(args.model)


if __name__ == "__main__":
    main()
