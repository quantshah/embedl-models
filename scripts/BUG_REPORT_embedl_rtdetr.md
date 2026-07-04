# Bug report: `transform()`/`quantize()` crash with `KeyError` on RT-DETR (shared `add` → `sigmoid`)

## Summary

`embedl_deploy.transform()` (and therefore `quantize()`) aborts with a bare
`KeyError: add_<n>` while recomposing a model whose graph contains an
`aten.add.Tensor` node that is (a) matched by `AtenAddPattern` **and**
(b) the input to a `sigmoid` matched by `AtenActivationPattern`. The recomposition
engine erases/replaces that shared `add` while applying one match, but the other
match still references the stale node, and `_insert_module()` raises instead of
remapping it.

This is hit by **RT-DETR / RT-DETRv2** (`RTDetrForObjectDetection`,
`RtDetrV2ForObjectDetection`), whose decoder refines reference points with the
"inverse-sigmoid" update `sigmoid(delta + log(r / (1 - r)))` — exactly an
`add(linear, log) → sigmoid`. The same models export cleanly to a HuggingFace
`torch.export` ATen graph and run a correct forward, so the model graph is sound;
the failure is entirely inside Embedl-Deploy's recomposition phase.

## Affected versions

| embedl-deploy | Result |
|---|---|
| **0.8.0** | Reproduced consistently (both RT-DETR models) |
| **0.7.0** | Reproduced (`KeyError: add_89` on `rtdetr_v2_r18vd`) |

The two defective code paths (below) are **identical** in 0.7.0 and 0.8.0
(`get_transformation_plan` overlap resolution and `tree/replace.py::_insert_module`),
so every release that ships this recomposition engine is affected.

## Environment where it reproduced

- `embedl-deploy` / `embedl-deploy-tensorrt` 0.8.0 and 0.7.0
- `torch==2.12.x` **CPU build** (`2.12.0+cpu` / `2.12.1+cpu`)
- `transformers` (main), CPU only (`CUDA_VISIBLE_DEVICES=""`)

> ⚠️ **Intermittency.** The crash is order-sensitive (see "Why it is
> intermittent"). It reproduced on every run in the CPU-wheel environment above,
> but stopped reproducing after the torch wheel was swapped to a CUDA build
> (`2.12.1+cu130`) in the same venv — the graph/allocation change shifts the
> match application order out of the failing window. Reproduce on a CPU wheel,
> and run the repro several times / across fresh processes.

## Reproduction

### Reliable: RT-DETR from the Hub (recorded failure)

```python
import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch, torch.nn as nn
from transformers import AutoModelForObjectDetection
from embedl_deploy import transform

model = AutoModelForObjectDetection.from_pretrained(
    "PekingU/rtdetr_v2_r18vd", torch_dtype=torch.float32,
    attn_implementation="sdpa").eval()

class Wrap(nn.Module):          # flat positional in / single tensor out
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, pixel_values): return self.m(pixel_values=pixel_values).logits

pixel_values = torch.randn(1, 3, 640, 640)
transform(Wrap(model), (pixel_values,))     # -> KeyError: add_89
```

Recorded result (embedl-deploy 0.8.0, torch CPU):

```
PekingU/rtdetr_r50vd      -> transform FAILED: KeyError: add_174
PekingU/rtdetr_v2_r18vd   -> transform FAILED: KeyError: add_89
```

### Minimal architecture (the offending topology)

The whole trigger is the reference-point-refinement head — an
`add(linear, log(...))` whose **sole consumer is a `sigmoid`**:

```python
import torch, torch.nn as nn
from embedl_deploy import transform

class DetrRefHead(nn.Module):
    """RT-DETR-style reference-point refinement: sigmoid(delta + inverse_sigmoid(ref))."""
    def __init__(self):
        super().__init__()
        self.bbox = nn.Sequential(nn.Linear(16, 16), nn.ReLU(),
                                  nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
    def forward(self, hidden, ref):
        delta = self.bbox(hidden)
        r = ref.clamp(1e-4, 1 - 1e-4)
        inv = torch.log(r / (1 - r))            # -> aten.log
        return torch.sigmoid(delta + inv)       # add(linear, log) -> sigmoid

m = DetrRefHead().eval()
args = (torch.randn(1, 6, 16), torch.rand(1, 6, 4))
for _ in range(50):                             # order-sensitive: loop to hit it
    transform(m, args)
```

This module contains the exact offending pair of matches
(`AtenAddPattern` on the `add`, `AtenActivationPattern` on the `sigmoid` whose
input is that `add`). Whether it crashes depends on the match application order
(see below); the full RT-DETR graph hits the failing order reliably in the
affected environment, a tiny graph only occasionally.

## Observed traceback

```
File ".../embedl_deploy/_internal/core/plan.py", line 306, in apply_transformation_plan
    pm.pattern.replace(pm)
File ".../embedl_deploy/_internal/core/patterns/main.py", line 132, in replace
    replacement_nodes = replace_tree(...)
File ".../embedl_deploy/_internal/core/tree/replace.py", line 189, in _insert_replacements
    nodes.extend(item(graph_module, prev_args))
File ".../embedl_deploy/_internal/core/tree/replace.py", line 77, in _insert
    node = _insert_module(graph_module, module, prev_args[:arg_count])
File ".../embedl_deploy/_internal/core/tree/replace.py", line 49, in _insert_module
    latest = max(args, key=_graph_order(graph_module))
KeyError: add_89
```

Instrumented at the failure point (embedl 0.8.0):

```
failing pattern : AtenActivationPattern      # recomposing an aten.sigmoid -> nn.Sigmoid
recomposing into: Sigmoid
input arg       : add_89 (op=call_function, target=aten.add.Tensor)
                  in_graph=False   in_replaced_map=False   users=[]
# add_89 == aten.add.Tensor(linear_57, log_3); its only user was sigmoid_4
# two enabled matches referenced it:
#   AtenAddPattern       tree=['add_89']   inputs=['linear_57','log_3']
#   AtenActivationPattern tree=['sigmoid_4'] inputs=['add_89']
```

## Root cause

Two independent defects in the recomposition engine combine:

1. **Overlap resolution ignores input nodes.**
   `get_transformation_plan()` (`_internal/core/plan.py`) resolves overlapping
   matches only on their **tree nodes**:

   ```python
   overlap = set(tree_nodes) & consumed.keys()   # tree nodes only
   ```

   A node that is a **tree node of one match** (`add_89` in `AtenAddPattern`) and
   an **input node of another** (`add_89` feeding `AtenActivationPattern`'s
   `sigmoid`) is not considered an overlap, so both matches stay `apply=True`.
   When the first is applied it erases/replaces `add_89`, invalidating the
   second match's input.

2. **`_insert_module()` assumes every input is still present.**
   `_internal/core/tree/replace.py::_insert_module`:

   ```python
   latest = max(args, key=_graph_order(graph_module))   # line 49
   ```

   `_graph_order` builds `{node: index for node in gm.graph.nodes}` and returns
   `dict.__getitem__`, so a stale input (`add_89`, no longer in the graph and not
   found in the `ReplaceSession` replaced-node map) raises a raw `KeyError`
   instead of being remapped or the match being skipped.

`replace_tree()` *does* remap stale inputs through `get_replaced_nodes()`, but in
the failing case the erased `add_89` is absent from that map
(`in_replaced_map=False`), so the remap is a no-op and the stale node reaches
`_insert_module()`.

## Why it is intermittent

Match application order comes from `pattern_matches` (registry order) →
`_matches_to_dict()` (keyed by `tree_nodes[-1].name`, insertion order). Match
*discovery* and the `set[fx.Node]`/`dict[fx.Node, …]` bookkeeping are keyed by
`fx.Node` objects, which hash by `id()`. Across processes / allocations / torch
builds the effective ordering shifts, so the window where the shared `add` is
erased *before* the dependent `sigmoid` is recomposed only opens on some runs.
On the CPU-wheel environment it opened every run for RT-DETR; after a torch
CUDA-wheel swap in the same venv it stopped. A correctness bug should not depend
on `id()` ordering — this intermittency is itself a symptom.

## Suggested fix (any one closes the crash; 1+2 are the clean fix)

1. In `get_transformation_plan()`, also record each match's **input nodes** when
   resolving overlaps (or when a match's input is another enabled match's tree
   node, order them so the consumer is recomposed first / mark one `apply=False`).
2. In `_insert_module()` (and the `fx.Node` branch of `_insert_replacements`),
   resolve `args` through `get_replaced_nodes()` and, if a node is still absent
   from the graph, raise a descriptive error rather than a bare `KeyError` from
   `_graph_order` — or make `_graph_order` tolerant and skip the match.
3. Make recomposition **degrade gracefully**: recomposition is an
   optimisation-enabling hint, so a single failing match should be skipped, not
   abort the whole `transform()`. (Verified workaround, below.)

## Verified workaround

Both RT-DETR recomposition failures (`AtenActivationPattern` `KeyError` and a
secondary `AtenLinearPattern._make_linear` assertion) raise **before** mutating
the graph, so skipping the individual failing match is safe:

```python
import embedl_deploy._internal.core.patterns.main as PM
_orig = PM.Pattern.replace.__func__
def _safe(cls, pm):
    try: return _orig(cls, pm)
    except Exception: return []      # skip this recomposition match
PM.Pattern.replace = classmethod(_safe)
```

With this shim, RT-DETR and RT-DETRv2 fully `transform()` + INT8-`quantize()` +
run a forward, in line with the other DETR-family detectors (RT-DETRv2: 518 fused
ops; RT-DETR: 826). Only `AtenActivationPattern` / `AtenLinearPattern` matches are
skipped; every other recomposition applies normally.

See `scripts/rtdetr_embedl_diagnosis.py` in this repo (`--fix` applies the shim).
