<!-- Copyright (C) 2025 Embedl AB -->

# ATen export & Embedl-Deploy quantization check

Two CPU-only checks against popular HuggingFace models:

1. **`check_aten_export.py`** — can the model be exported to an **ATen graph**
   via HuggingFace's [`transformers.exporters`](https://huggingface.co/docs/transformers/main/en/exporters)
   (`DynamoExporter` → `torch.export` → `ExportedProgram`), and does a forward
   run on the exported graph and match eager?
2. **`check_embedl_quantize.py`** — do
   [Embedl Deploy](https://docs.embedl.com/embedl-deploy/latest/guide/quantization.html)
   `transform()` and `quantize()` run on the same models, producing a fused
   ATen graph and an INT8-quantized graph that each run a forward?

Everything runs on **PyTorch CPU** (`torch==2.12`, `transformers` main,
`embedl-deploy` + `embedl-deploy-tensorrt` 0.8.0).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.12.0"
pip install "git+https://github.com/huggingface/transformers.git" \
            accelerate huggingface_hub pillow timm scipy \
            embedl-deploy embedl-deploy-tensorrt

python scripts/check_aten_export.py              # ATen export sweep
python scripts/check_embedl_quantize.py          # transform + quantize sweep
python scripts/check_aten_export.py --trending   # bonus: trending top-10 LLMs
```

## 1. ATen export (`transformers.exporters`, CPU)

Real weights, `attn_implementation="sdpa"` (flash/flex attention is not
exportable). `ex` = exported to an ATen graph, `fwd` = forward runs on the
exported graph, `match` = matches eager within 1e-3.

| Model | Category | Architecture | ATen ops | ex | fwd | match |
|-------|----------|--------------|---------:|:--:|:---:|:-----:|
| facebook/sam-vit-base | vision-seg | SamModel | 1112 | ✅ | ✅ | ✅ |
| facebook/dinov2-base | vision-backbone | Dinov2Model | 311 | ✅ | ✅ | ✅ |
| facebook/dino-vitb16 | vision-backbone | ViTModel | 307 | ✅ | ✅ | ✅ |
| facebook/detr-resnet-50 | av-detection | DetrForObjectDetection | 1365 | ✅ | ✅ | ✅ |
| IDEA-Research/grounding-dino-tiny | av-detection | GroundingDinoForObjectDetection | 4555 | ✅ | ✅ | ✅ |
| google/owlv2-base-patch16-ensemble | av-detection | Owlv2ForObjectDetection | 668 | ✅ | ✅ | ✅ |
| depth-anything/Depth-Anything-V2-Small-hf | av-depth | DepthAnythingForDepthEstimation | 454 | ✅ | ✅ | ✅ |
| Intel/dpt-hybrid-midas | av-depth | DPTForDepthEstimation | 771 | ✅ | ✅ | ✅ |
| nvidia/GR00T-N1.5-3B | robotics-vla | — | — | ❌ | — | — |
| lerobot/smolvla_base | robotics-vla | — | — | ❌ | — | — |

**8/10 export to an ATen graph and run a matching forward.** The exported
graphs are pure `torch.ops.aten.*` operators — see `aten_graphs/*.aten_graph.txt`
(a sample, `dinov2.aten_graph.txt`, is committed).

Notes:
- **GR00T / SmolVLA** are *not* transformers `PreTrainedModel`s (GR00T has
  `library=None` and an unregistered `model_type=gr00t_n1_5`; SmolVLA is a
  `lerobot` policy with no `model_type`). The transformers exporter cannot load
  them — they would need raw `torch.export` on the underlying `nn.Module`.
- **BEV models** (BEVFormer / BEVFusion / PETR) live in `mmdetection3d`, not
  transformers, so they are outside this exporter too. The autonomous-driving
  perception models that *are* in transformers — detectors (DETR, Grounding
  DINO, OWLv2) and monocular-depth (Depth-Anything-V2, DPT) — all export.

## 2. Embedl-Deploy transform + quantize (INT8 PTQ, CPU)

`transform()` fuses the model into an ATen `GraphModule`; `quantize()` inserts
INT8 fake-quant (per-channel weights, per-tensor activations) and calibrates
with a short forward loop. `tf` = transformed graph runs a forward, `q` =
quantized graph runs a forward. `rel-diff` is the max relative deviation of the
INT8 output vs fp32 — **calibration used a single random-noise input, so this
measures functional correctness of the pipeline, not real-data accuracy.**

| Model | fused ops | fq nodes | tf | q | rel-diff |
|-------|----------:|---------:|:--:|:-:|---------:|
| facebook/sam-vit-base | — | — | ❌ | ❌ | — |
| facebook/dinov2-base | 49 | 85 | ✅ | ✅ | 0.86 |
| facebook/dino-vitb16 | 70 | 85 | ✅ | ✅ | 0.20 |
| facebook/detr-resnet-50 | 604 | 142 | ✅ | ✅ | 0.91 |
| IDEA-Research/grounding-dino-tiny | 2284 | 172 | ✅ | ✅ | nan* |
| google/owlv2-base-patch16-ensemble | 237 | 123 | ✅ | ✅ | 0.60 |
| depth-anything/Depth-Anything-V2-Small-hf | 80 | 55 | ✅ | ✅ | 0.13 |
| Intel/dpt-hybrid-midas | 314 | 115 | ✅ | ✅ | 0.055 |
| Qwen/Qwen3-0.6B (LLM, tiny cfg) | — | — | ✅ | ✅ | 0.011 |

**7/8 vision/AV models + the Qwen3 LLM baseline** run through both
`transform()` and `quantize()` and execute a forward on the quantized graph.

Notes:
- **SAM** is the one `transform()` failure: Embedl-Deploy's TensorRT MHA-fusion
  pass (`MHAInProjection`) assumes a rank-3 `(batch, seq, dim)` attention input,
  but SAM's windowed attention feeds a different rank →
  `ValueError: too many values to unpack (expected 3)`. A genuine
  Embedl-Deploy limitation on SAM's attention shape, worth reporting upstream.
- **`grounding-dino-tiny` rel-diff is `nan`**: the fp32 reference output itself
  contains `nan` on the random text+image probe (empty detections). The graph
  transforms, quantizes, and runs — only the numeric comparison is undefined.
- **Trending Qwen3.5 / Qwen3.5-MoE LLMs** (`empero-ai/Qwythos-9B…`,
  `Qwen/Qwen-AgentWorld-35B-A3B`) **OOM during `quantize()`** on this 15 GB CPU
  box — their linear-attention graphs exceed memory during the quant passes.
  This is a resource limit here, not an API defect; `Qwen/Qwen3-0.6B` (a
  standard decoder) quantizes cleanly with the lowest error of the set (0.011).

## Bonus: trending top-10 LLMs (ATen export)

Run with `--trending`. On this CPU box the literal top-10 trending models are
9B–35B (several MoE / NVFP4-quantized), so they are built as **downsized random
variants of their real architectures** (identical op graph, tiny weights) to
test exportability without downloading hundreds of GB. Of the 10, the loadable
non-blocked architectures (`Qwen3.5`, `Qwen3.5-MoE`) export to ATen and run a
matching forward; `DeepseekV4` lacks an `sdpa`/exportable attention path and
`GlmMoeDsa` requires model-side fixes.
