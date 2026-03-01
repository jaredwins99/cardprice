# DINOv2 Inference Optimization: ONNX Runtime & TensorRT Research

**Date:** 2026-02-28
**Current setup:** PyTorch 2.10.0+cu128, RTX 4070 SUPER, DINOv2 ViT-B/14
**Current latency:** ~100-200ms per card on GPU (PyTorch eager mode)
**Goal:** Reduce per-card inference time for embedding extraction

---

## 1. Current Bottleneck Analysis

Our `dino_matcher.py` uses `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")` with
vanilla PyTorch inference. The 100-200ms per card includes:
- Image preprocessing (resize to 224x224, normalize) — negligible
- Model forward pass — bulk of the time
- CPU copy + L2 normalization — negligible

The ViT-B/14 model has ~86M parameters, 768-dim output, 12 transformer layers.

---

## 2. ONNX Export Feasibility

### Can DINOv2 be exported to ONNX?

**Yes, but with workarounds.** Direct `torch.onnx.export()` fails due to:

1. **`MemEffAttention` incompatibility** — The default DINOv2 uses memory-efficient attention
   (`xformers`), which is not traceable by ONNX. Must replace with standard `Attention`.
2. **Device mismatch** — The `mask_token` lives on CPU while other tensors are on CUDA.
   Export must be done on CPU.
3. **Bicubic interpolation** — Position encoding interpolation uses `bicubic` mode which
   ONNX doesn't support well. Must patch to `bilinear`.
4. **Dynamic tensor ops** — Scale factors in `interpolate()` must be converted with `.item()`
   to static Python floats.

### Working export approach

```python
import torch

# Load model on CPU for export
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model.to('cpu')
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model, dummy, 'dinov2_vitb14.onnx',
    opset_version=17,
    input_names=['input'],
    output_names=['embedding'],
    dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
)
```

**Caveats:**
- May need to patch `dinov2/layers/attention.py` to remove xformers dependency
- Pre-exported ONNX models exist on HuggingFace: `sefaburak/dinov2-small-onnx`
  (small only; we use base, so we'd need to export ourselves)
- Reference repo: [sefaburakokcu/dinov2_onnx](https://github.com/sefaburakokcu/dinov2_onnx)

### Sources
- [DINOv2 ONNX export issue #19](https://github.com/facebookresearch/dinov2/issues/19)
- [dinov2_onnx inference repo](https://github.com/sefaburakokcu/dinov2_onnx)
- [sefaburak/dinov2-small-onnx on HuggingFace](https://huggingface.co/sefaburak/dinov2-small-onnx)

---

## 3. ONNX Runtime

### Current status on our system

**Not installed.** `import onnxruntime` raises `ModuleNotFoundError`.

### Installation

```bash
# GPU-accelerated ONNX Runtime (includes CUDA EP)
pip install onnxruntime-gpu

# Or CPU-only (still faster than PyTorch eager for inference)
pip install onnxruntime
```

`onnxruntime-gpu` ships with its own CUDA/cuDNN, so it should work with our CUDA 12.8
setup. The `CUDAExecutionProvider` will be available automatically.

### Expected speedup with ONNX Runtime (CUDAExecutionProvider)

| Optimization         | Expected latency | Speedup vs baseline |
|---------------------|------------------|---------------------|
| PyTorch eager (now) | 100-200ms        | 1x                  |
| ONNX Runtime + CUDA | 40-80ms          | ~2-3x               |
| + FP16 graph opt    | 25-50ms          | ~3-5x               |

ONNX Runtime gains come from:
- Graph-level optimizations (constant folding, operator fusion)
- Elimination of Python overhead
- Better CUDA kernel selection
- Optional FP16 mode (RTX 4070 SUPER has excellent FP16 throughput)

### Inference code pattern

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(
    'dinov2_vitb14.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# Enable FP16
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Run inference
input_np = preprocessed_image.numpy()  # (1, 3, 224, 224) float32
embedding = sess.run(None, {'input': input_np})[0]  # (1, 768)
```

---

## 4. TensorRT

### Expected speedup

TensorRT provides the deepest optimization on NVIDIA hardware:

| Optimization          | Expected latency | Speedup vs baseline |
|----------------------|------------------|---------------------|
| PyTorch eager (now)  | 100-200ms        | 1x                  |
| TensorRT FP32       | 40-70ms          | ~2-3x               |
| TensorRT FP16       | 15-35ms          | ~4-8x               |
| TensorRT INT8 (cal) | 10-25ms          | ~5-10x              |

FP16 is the sweet spot — RTX 4070 SUPER (Ada Lovelace, compute 8.9) has dedicated
FP16 tensor cores with 2x throughput over FP32.

**Important caveat:** INT8 quantization of DINOv2 has been reported to show minimal
speedup over FP16 in practice ([GitHub issue #489](https://github.com/facebookresearch/dinov2/issues/489)).
This is common with ViT architectures where the attention layers don't benefit as much
from INT8 as CNN layers do.

### Known issues

1. **FMHA fusion failure** — TensorRT 10.8 has a [known bug](https://github.com/NVIDIA/tensorrt/issues/4537)
   where Fused Multi-Head Attention layers are not generated from DINOv2's ONNX graph.
   The MHA-related layers aren't being properly identified during PyTorch ONNX export.
   Fixed in TensorRT >= 10.6 with specific workarounds.

2. **Performance parity issue** — Some users report that DINOv2 TensorRT FP16 shows
   no improvement over FP32 on certain GPU/TRT version combinations
   ([NVIDIA forum](https://forums.developer.nvidia.com/t/dinov2-tensorrt-model-performance-issue/312251)).
   This is GPU/driver dependent.

### Deployment options

**Option A: ONNX Runtime with TensorRT EP** (recommended for us)
```bash
pip install onnxruntime-gpu  # includes TensorRT EP
```
```python
sess = ort.InferenceSession(
    'dinov2_vitb14.onnx',
    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider']
)
```
This auto-compiles TensorRT engines from the ONNX graph. First run is slow (engine
compilation), subsequent runs use cached engines.

**Option B: Standalone TensorRT** (maximum performance, more setup)
```bash
pip install tensorrt
trtexec --onnx=dinov2_vitb14.onnx --saveEngine=dinov2_vitb14.engine --fp16
```

### Sources
- [Accelerating Vision AI with TensorRT: DINOv2 in Practice](https://medium.com/@testth02/accelerating-vision-ai-inference-with-tensorrt-yolov8-and-dinov2-optimization-in-practice-287acd4c73e1)
- [TensorRT FMHA fusion issue #4537](https://github.com/NVIDIA/tensorrt/issues/4537)
- [DINOv2 INT8 not faster than FP16 issue #489](https://github.com/facebookresearch/dinov2/issues/489)
- [DINOv2 TensorRT performance issue](https://forums.developer.nvidia.com/t/dinov2-tensorrt-model-performance-issue/312251)

---

## 5. ViT Architecture Gotchas with ONNX

1. **Attention mechanism** — `MemEffAttention` (xformers) is not ONNX-traceable. Must
   use standard scaled dot-product attention. When loading from torch.hub, pass the
   right arguments or monkey-patch the attention layers before export.

2. **Dynamic axes** — The ONNX converter can fail to propagate dynamic axis info
   through the transformer encoder graph
   ([PyTorch issue #110801](https://github.com/pytorch/pytorch/issues/110801)).
   If batch-dynamic export fails, fall back to static batch=1 (fine for our use case).

3. **Position interpolation** — DINOv2 interpolates position embeddings for non-standard
   resolutions. The `bicubic` mode must be replaced with `bilinear` for ONNX compat.
   Since we always use 224x224 (the native resolution), this interpolation is a no-op
   and can be patched out entirely.

4. **Opset version** — Use opset >= 14 for full LayerNormalization support. Opset 17
   is recommended for best ViT compatibility.

5. **torch.compile alternative** — PyTorch 2.x `torch.compile()` with `mode="reduce-overhead"`
   can provide 1.5-2x speedup with zero export hassle. Worth benchmarking first as a
   low-effort baseline before going full ONNX.

### Sources
- [Debugging ViT and TensorRT compilation](https://ohadravid.github.io/posts/2025-01-debugging-vit-and-tensorrt/)
- [ViT ONNX export on Medium](https://medium.com/@romanbessouat/fine-tune-a-vision-transformers-model-and-export-it-to-onnx-format-245173b69549)
- [PyTorch dynamic axes bug #110801](https://github.com/pytorch/pytorch/issues/110801)

---

## 6. Other Quick Wins (No Export Required)

Before investing in ONNX/TRT, these PyTorch-native optimizations are worth trying:

| Technique                        | Expected speedup | Effort |
|----------------------------------|------------------|--------|
| `torch.compile(model)`          | 1.5-2x           | 1 line |
| `torch.cuda.amp.autocast()`     | 1.3-1.8x         | 2 lines|
| Batch multiple cards together    | 3-5x throughput  | Medium |
| Use ViT-S/14 instead of ViT-B  | 2-3x             | 1 line |
| `torch.backends.cudnn.benchmark`| 1.1-1.2x         | 1 line |

**Batching** is probably the single biggest win if processing binder pages with multiple
cards — our `binder_scanner.py` segments multiple cards, and we currently embed them
one at a time. Processing 9 cards in a single batch would be ~5x faster total throughput.

**ViT-S/14** (384-dim, ~22M params) is 3-4x faster than ViT-B/14 and may be
sufficient for card identification given we're matching against a reference index.

---

## 7. Recommended Action Plan

### Phase 1: Quick wins (30 min, no dependencies)
1. Add `torch.compile(model, mode="reduce-overhead")` in `_load_model()`
2. Add `torch.cuda.amp.autocast()` context in `extract_embedding()`
3. Add batch embedding support for binder page processing
4. Benchmark — expect **2-3x improvement** (50-100ms -> 30-60ms per card)

### Phase 2: ONNX Runtime (2-3 hours)
1. `pip install onnxruntime-gpu`
2. Export DINOv2 to ONNX (handle the attention/interpolation patches)
3. Add ONNX inference path in `dino_matcher.py` with auto-fallback to PyTorch
4. Benchmark — expect **3-5x improvement** (100-200ms -> 25-50ms per card)

### Phase 3: TensorRT (optional, 4-6 hours)
1. Only pursue if Phase 2 doesn't meet latency targets
2. Use ONNX Runtime TensorRT EP (easier) or standalone trtexec
3. FP16 precision — skip INT8 (marginal gain for ViTs, calibration complexity)
4. Benchmark — expect **5-8x improvement** (100-200ms -> 15-30ms per card)

### Phase 4: Model size reduction (if needed)
1. Switch from ViT-B/14 (768-dim) to ViT-S/14 (384-dim)
2. Rebuild FAISS index with 384-dim vectors
3. Test accuracy impact on card identification
4. Combined with ONNX: potentially **<10ms per card**

---

## 8. Summary

| Approach                    | Latency estimate | Speedup | Effort   | Risk  |
|-----------------------------|------------------|---------|----------|-------|
| Current (PyTorch eager)     | 100-200ms        | 1x      | —        | —     |
| torch.compile + AMP         | 40-80ms          | 2-3x    | Trivial  | None  |
| ONNX Runtime (CUDA EP)      | 25-50ms          | 3-5x    | Medium   | Low   |
| ONNX Runtime (TensorRT EP)  | 15-35ms          | 4-8x    | Medium   | Med   |
| Standalone TensorRT FP16    | 15-30ms          | 5-8x    | High     | Med   |
| ViT-S + ONNX RT             | 8-15ms           | 10-15x  | Medium   | Low   |

**Recommendation:** Start with Phase 1 (torch.compile + batching) for immediate
gains with zero risk, then move to Phase 2 (ONNX Runtime) if more speed is needed.
TensorRT is overkill for our use case unless we need real-time (<16ms) inference.
