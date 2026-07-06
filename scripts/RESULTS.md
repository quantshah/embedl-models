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

## 3. DETR-family object detectors (RF-DETR and relatives)

Prompted by "more models like RF-DETR". RF-DETR itself is on the Hub as a
transformers model (`Roboflow/rf-detr-*`, `RfDetrForObjectDetection`), so the
*actual* RF-DETR is tested alongside its relatives. Run with:

```bash
python scripts/check_aten_export.py --only rf_detr rtdetr rtdetrv2 \
    deformable_detr conditional_detr deta yolos table_transformer
python scripts/check_embedl_quantize.py --keys rf_detr rtdetr rtdetrv2 \
    deformable_detr conditional_detr yolos table_transformer
```

### ATen export (`transformers.exporters`)

| Model | Architecture | ATen ops | ex+fwd+match |
|-------|--------------|---------:|:------------:|
| Roboflow/rf-detr-nano | RfDetrForObjectDetection | 753 | ✅ |
| PekingU/rtdetr_r50vd | RTDetrForObjectDetection | 1746 | ✅ |
| PekingU/rtdetr_v2_r18vd | RTDetrV2ForObjectDetection | 1028 | ✅ |
| SenseTime/deformable-detr | DeformableDetrForObjectDetection | 2291 | ✅ |
| microsoft/conditional-detr-resnet-50 | ConditionalDetrForObjectDetection | 1502 | ✅ |
| hustvl/yolos-tiny | YolosForObjectDetection | 307 | ✅ |
| microsoft/table-transformer-detection | TableTransformerForObjectDetection | 1175 | ✅ |
| jozhang97/deta-resnet-50 | DetaForObjectDetection | — | ❌ load |

**7/8 export to an ATen graph and forward-match** — including RF-DETR itself and
Deformable DETR (whose multi-scale deformable attention traces fine). **DETA**
fails to load: `model_type=deta` was removed from current transformers
(`does not recognize this architecture`).

### Embedl-Deploy transform + quantize (INT8 PTQ)

| Model | fused ops | fq nodes | tf | q | rel-diff |
|-------|----------:|---------:|:--:|:-:|---------:|
| Roboflow/rf-detr-nano | 278 | 106 | ✅ | ✅ | 0.55 |
| PekingU/rtdetr_r50vd | — | — | ❌ | ❌ | — |
| PekingU/rtdetr_v2_r18vd | — | — | ❌ | ❌ | — |
| SenseTime/deformable-detr | 1278 | 152 | ✅ | ✅ | 0.65 |
| microsoft/conditional-detr-resnet-50 | 664 | 142 | ✅ | ✅ | 0.56 |
| hustvl/yolos-tiny | 71 | 40 | ✅ | ✅ | 0.060 |
| microsoft/table-transformer-detection | 504 | 106 | ✅ | ✅ | 0.19 |

**5/7 transform + quantize cleanly**, including **RF-DETR**. Finding:
**RT-DETR and RT-DETRv2 both fail `transform()` with `KeyError: add_<n>`** — a
node-reference bug in an Embedl-Deploy fusion pass on the RT-DETR graph (they
export fine to a HuggingFace ATen graph, so the graph itself is sound). Worth
reporting upstream to Embedl-Deploy. (rel-diff again reflects random-noise
calibration, not real-data accuracy.)

### Deep dive: why RT-DETR / RT-DETRv2 fail Embedl-Deploy `transform()`

`scripts/rtdetr_embedl_diagnosis.py` reproduces and explains the
`KeyError: add_89`. Both RT-DETR models export to a valid HuggingFace ATen graph
and run a matching forward, so the model graph is sound — the failure is inside
Embedl-Deploy's **recomposition phase** (`transform()` → `prepare_graph()` →
`export_trace()` → `apply_transformation_plan()`).

**Root cause (embedl-deploy 0.8.0).** Recomposition patterns are applied
sequentially in one `ReplaceSession`, and `AtenConv2dPattern` (registry index 4)
runs before `AtenActivationPattern` (index 6):

- A residual/bias `add` node (`add_89`, `aten.add.Tensor`) that feeds a
  `sigmoid` is detached and erased while an earlier pattern is applied, **without
  being recorded in the `ReplaceSession` replaced-node map**.
- When `AtenActivationPattern` recomposes that `sigmoid` into `nn.Sigmoid`,
  `replace_tree()` → `_insert_module()` computes
  `max(args, key=_graph_order(gm))` over its inputs. `add_89` is now in neither
  the graph nor the remap (verified: `in_graph=False`, `in_replaced_map=False`,
  `users=[]`), so the `_graph_order` dict lookup raises `KeyError: add_89`.

Two contributing embedl-deploy issues:
1. `apply_transformation_plan()`'s overlap guard checks only each match's
   **tree nodes** (`get_tree_nodes()`), not the **input nodes** shared between
   matches — so a node can be erased by one match while still referenced as
   another match's input.
2. `_insert_module()`'s `max(args, key=_graph_order(gm))` assumes every input
   arg is still present, turning a stale reference into a `KeyError` instead of
   remapping or skipping.

RT-DETR hits this because its decoder applies `sigmoid` to reference-point /
query-selection tensors that **share an `add` producer** with a conv/linear that
recomposition rewrites; DETR, RF-DETR, YOLOS, Deformable and Conditional DETR
lack that exact shared-input shape. A second, independent latent failure exists:
`AtenLinearPattern._make_linear` asserts `isinstance(args[1], fx.Node)` on some
RT-DETR linears.

**Work-around (verified).** Both failures raise *before* mutating the graph, so a
single failing recomposition match can be skipped safely. `--fix` wraps
`Pattern.replace` to skip an individually-failing match instead of aborting the
whole transform:

```
python scripts/rtdetr_embedl_diagnosis.py            # reproduce + diagnose
python scripts/rtdetr_embedl_diagnosis.py --fix      # RT-DETR then passes
```

With the shim, both models transform + INT8-quantize + run a forward
(RT-DETRv2: 518 fused ops, rel-diff 0.64; RT-DETR: 826 ops, rel-diff 0.63 —
in line with Deformable/Conditional DETR). Skipped matches:
`AtenActivationPattern` and `AtenLinearPattern` only. This is a good candidate
fix to report upstream: recomposition is an optimisation hint, so one failing
match should degrade gracefully rather than abort `transform()`.

**Version coverage.** The crash was confirmed on both **embedl-deploy 0.8.0**
(consistent) and **0.7.0** (`KeyError: add_89` on `rtdetr_v2_r18vd`) — the two
defective code paths (`get_transformation_plan` overlap resolution and
`tree/replace.py::_insert_module`) are identical across releases. The
manifestation is **order-sensitive**: it reproduced on every run against a torch
**CPU wheel**, but stopped after the same venv's torch was swapped to a CUDA
wheel (`2.12.1+cu130`) — the graph/allocation change shifts the match
application order out of the failing window (a correctness bug that depends on
`id()`-hashed node ordering). Full write-up — root cause, minimal architecture,
suggested fixes — in **`scripts/BUG_REPORT_embedl_rtdetr.md`**.

## 4. Widened coverage: AV segmentation, speech, text

Extends the sweep beyond vision/detection into autonomous-driving scene
segmentation, speech, and text. Run with:

```bash
python scripts/check_aten_export.py --only segformer mask2former swinv2 \
    whisper wav2vec2 wavlm bert minilm roberta modernbert gpt2 t5
python scripts/check_embedl_quantize.py --keys segformer mask2former swinv2 \
    whisper wav2vec2 wavlm bert minilm roberta modernbert gpt2 t5
```

| Model | Arch | Category | ATen ops | export | embedl tf+quant |
|-------|------|----------|---------:|:------:|:---------------:|
| nvidia/segformer-b0-finetuned-ade-512-512 | SegformerForSemanticSegmentation | AV segmentation | 328 | ✅ | ✅ (0.42) |
| facebook/mask2former-swin-tiny-coco-instance | Mask2FormerModel | AV segmentation | 2734 | ✅ | ✅ (0.15) |
| microsoft/swinv2-tiny-patch4-window8-256 | Swinv2ForImageClassification | vision backbone | 1063 | ✅ | ✅ (0.16) |
| openai/whisper-tiny | WhisperForConditionalGeneration | speech (ASR) | 374 | ✅ | ✅ (0.27) |
| facebook/wav2vec2-base-960h | Wav2Vec2ForCTC | speech (ASR) | 353 | ✅ | ✅ (0.57) |
| microsoft/wavlm-base | WavLMModel | speech | 708 | ✅ | ⚠️ transform ✅, quant ✗ |
| google-bert/bert-base-uncased | BertModel | text | 320 | ✅ | ✅ (1.0) |
| sentence-transformers/all-MiniLM-L6-v2 | BertModel | text embedding | 182 | ✅ | ✅ (1.0) |
| FacebookAI/roberta-base | RobertaModel | text | 331 | ✅ | ✅ (0.54) |
| answerdotai/ModernBERT-base | ModernBertModel | text | 1269 | ✅ | ✅ (0.19) |
| openai-community/gpt2 | GPT2LMHeadModel | text (LM) | 597 | ✅ | ✅ (0.062) |
| google-t5/t5-small | T5ForConditionalGeneration | text (seq2seq) | 993 | ✅ | ✅ (0.080) |
| qualcomm/BEVFusion | — | AV / BEV | — | ❌ load | — |

**12/13 export to an ATen graph; 11/12 loadable also pass Embedl transform +
INT8 quantize.** Both encoder-decoder speech/text models (Whisper, T5) export
and quantize via a single traced forward (`input_features`/`input_ids` +
`decoder_input_ids`).

Notes:
- **BEV / 3D detection is not on the Hub as transformers models.** Every
  BEVFormer / BEVFusion / PETR / DETR3D checkpoint is a raw `mmdetection3d` /
  `pytorch` artifact (`library=None`, no `model_type`) — `qualcomm/BEVFusion`
  fails to load for exactly this reason, like GR00T / SmolVLA. Exporting them
  would need raw `torch.export` on the `mmdet3d` `nn.Module` (custom voxel /
  deformable-attention CUDA ops — out of scope for a CPU transformers-exporter
  sweep). The AV perception that *is* in transformers — detection (DETR family),
  monocular depth (Depth-Anything, DPT), and now semantic/panoptic segmentation
  (SegFormer, Mask2Former) — all export and (mostly) quantize.
- **WavLM** exports and `transform()`s but `quantize()` fails with a shape
  mismatch (`768 vs 64`) in its gated relative-position attention — an
  Embedl-Deploy quantizer limitation on that attention variant (Wav2Vec2, same
  family without gated rel-pos, quantizes fine).
- Text encoders show large `rel-diff` (BERT/MiniLM ≈ 1.0) purely because INT8
  activation calibration used a single random sentence; GPT2 / T5 (0.06–0.08)
  and ModernBERT (0.19) fare much better on the same crude calibration.

## 5. Classic backbones, DINOv3, SAM3 (incl. random-weights-from-config)

Classic classifiers/backbones use real weights; **gated / unreleased models
(DINOv3, SAM3) are instantiated from their transformers config class with random
weights and no Hub download**, so no auth token is needed
(`load_random_model()` — set `weights="random"` + `config_class` on the spec).

```bash
python scripts/check_aten_export.py --only vit resnet convnext convnextv2 dinov3 sam3
python scripts/check_embedl_quantize.py --keys vit resnet convnext convnextv2 dinov3 sam3
```

| Model | Arch | Weights | ATen ops | export | embedl tf+quant |
|-------|------|---------|---------:|:------:|:---------------:|
| google/vit-base-patch16-224 | ViTForImageClassification | real | 306 | ✅ | ✅ (0.30) |
| microsoft/resnet-50 | ResNetForImageClassification | real | 175 | ✅ | ✅ (0.080) |
| facebook/convnext-tiny-224 | ConvNextForImageClassification | real | 181 | ✅ | ✅ (0.12) |
| facebook/convnextv2-tiny-1k-224 | ConvNextV2ForImageClassification | real | 307 | ✅ | ✅ (0.51) |
| facebook/dinov3-vitb16-pretrain-lvd1689m | DINOv3ViTModel | random (gated) | 126 | ✅ | ✅ (0.077) |
| facebook/sam3 | Sam3Model | random (gated) | 4330 | ✅ | ✅ (0.88) |

**6/6 export to an ATen graph and 6/6 pass Embedl transform + INT8 quantize** —
including **DINOv3** and **SAM3** built from config with random weights (SAM3 is
text-promptable, so it takes `pixel_values` + `input_ids`).

Notes:
- **DINOv3** is gated on the Hub (`gated=manual`); building `DINOv3ViTConfig()`
  directly avoids the download entirely. The random-weight graph is
  architecturally identical to the real model, which is what `torch.export`
  traces.
- **SAM-3D** (`facebook/sam-3d-objects`, `facebook/sam-3d-body-*`) is **not** a
  transformers model — it ships under custom `sam-3d-objects` / `sam-3d-body`
  libraries with no transformers config class, so it cannot be built from config
  (unlike SAM3, which is `Sam3Model` in transformers). Out of scope for this
  exporter, same as GR00T / SmolVLA / BEV.

## Bonus: trending top-10 LLMs (ATen export)

Run with `--trending`. On this CPU box the literal top-10 trending models are
9B–35B (several MoE / NVFP4-quantized), so they are built as **downsized random
variants of their real architectures** (identical op graph, tiny weights) to
test exportability without downloading hundreds of GB. Of the 10, the loadable
non-blocked architectures (`Qwen3.5`, `Qwen3.5-MoE`) export to ATen and run a
matching forward; `DeepseekV4` lacks an `sdpa`/exportable attention path and
`GlmMoeDsa` requires model-side fixes.
