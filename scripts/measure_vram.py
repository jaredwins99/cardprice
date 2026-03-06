#!/usr/bin/env python3
"""Measure actual GPU VRAM usage for all ML models in the cardprice pipeline.

Loads each model in FP32 and FP16, measuring torch.cuda.memory_allocated()
before and after each load.  Prints a summary table and headroom estimate.

Usage:
    python scripts/measure_vram.py
"""

import gc
import sys

import torch


def fmt_mb(bytes_val: int) -> str:
    """Format bytes as MB string."""
    return f"{bytes_val / (1024 ** 2):.1f} MB"


def clear_gpu():
    """Force-clear all GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def measure_dinov2(*, fp16: bool) -> dict:
    """Load DINOv2 ViT-B/14 and measure VRAM."""
    clear_gpu()
    before = torch.cuda.memory_allocated()

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.to("cuda")
    model.eval()
    if fp16:
        model.half()

    # Run a dummy forward pass to capture any extra buffers
    dummy = torch.randn(1, 3, 224, 224, device="cuda")
    if fp16:
        dummy = dummy.half()
    with torch.no_grad():
        _ = model(dummy)

    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()

    # Cleanup
    del model, dummy
    clear_gpu()

    return {
        "allocated": after - before,
        "peak": peak - before,
    }


def measure_clip(*, fp16: bool) -> dict:
    """Load CLIP ViT-Large/14 and measure VRAM."""
    from transformers import CLIPModel, CLIPProcessor

    clear_gpu()
    before = torch.cuda.memory_allocated()

    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    model.to("cuda")
    if fp16:
        model.half()

    # Dummy forward pass (image side)
    from PIL import Image
    import numpy as np

    dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    inputs = processor(images=dummy_img, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    if fp16:
        inputs = {
            k: v.half() if v.is_floating_point() else v
            for k, v in inputs.items()
        }
    with torch.no_grad():
        _ = model.get_image_features(**inputs)

    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()

    del model, processor, inputs, dummy_img
    clear_gpu()

    return {
        "allocated": after - before,
        "peak": peak - before,
    }


def measure_paddleocr() -> dict | None:
    """Check if PaddleOCR uses GPU VRAM."""
    try:
        import paddle
    except ImportError:
        return None

    clear_gpu()
    before = torch.cuda.memory_allocated()

    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=True)

        # Dummy inference
        import numpy as np
        dummy = np.zeros((100, 300, 3), dtype=np.uint8)
        ocr.ocr(dummy, cls=True)
    except Exception as e:
        print(f"  PaddleOCR measurement failed: {e}")
        return None

    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()

    del ocr
    clear_gpu()

    # PaddleOCR uses PaddlePaddle's own GPU allocator, not PyTorch's.
    # torch.cuda.memory_allocated() won't see it. Report what we can.
    return {
        "allocated": after - before,
        "peak": peak - before,
        "note": "PaddlePaddle uses its own allocator; torch cannot measure it. "
                "Use nvidia-smi for true VRAM.",
    }


def measure_easyocr() -> dict | None:
    """Check if EasyOCR uses GPU VRAM."""
    try:
        import easyocr
    except ImportError:
        return None

    clear_gpu()
    before = torch.cuda.memory_allocated()

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    # Dummy inference to materialize all buffers
    import numpy as np
    dummy = np.zeros((100, 300, 3), dtype=np.uint8)
    reader.readtext(dummy)

    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()

    del reader
    clear_gpu()

    return {
        "allocated": after - before,
        "peak": peak - before,
    }


def main():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available. This script requires a GPU.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_mem
    print(f"GPU: {device_name}")
    print(f"Total VRAM: {fmt_mb(total_vram)}")
    print()

    results = {}

    # ── DINOv2 ──────────────────────────────────────────────────────
    print("Loading DINOv2 ViT-B/14 (FP32)...")
    dino_fp32 = measure_dinov2(fp16=False)
    print(f"  Allocated: {fmt_mb(dino_fp32['allocated'])}, Peak: {fmt_mb(dino_fp32['peak'])}")

    print("Loading DINOv2 ViT-B/14 (FP16)...")
    dino_fp16 = measure_dinov2(fp16=True)
    print(f"  Allocated: {fmt_mb(dino_fp16['allocated'])}, Peak: {fmt_mb(dino_fp16['peak'])}")
    results["DINOv2 ViT-B/14"] = (dino_fp32, dino_fp16)

    # ── CLIP ────────────────────────────────────────────────────────
    print("Loading CLIP ViT-L/14 (FP32)...")
    clip_fp32 = measure_clip(fp16=False)
    print(f"  Allocated: {fmt_mb(clip_fp32['allocated'])}, Peak: {fmt_mb(clip_fp32['peak'])}")

    print("Loading CLIP ViT-L/14 (FP16)...")
    clip_fp16 = measure_clip(fp16=True)
    print(f"  Allocated: {fmt_mb(clip_fp16['allocated'])}, Peak: {fmt_mb(clip_fp16['peak'])}")
    results["CLIP ViT-L/14"] = (clip_fp32, clip_fp16)

    # ── EasyOCR ─────────────────────────────────────────────────────
    print("Loading EasyOCR...")
    easyocr_result = measure_easyocr()
    if easyocr_result:
        print(f"  Allocated: {fmt_mb(easyocr_result['allocated'])}, Peak: {fmt_mb(easyocr_result['peak'])}")
        results["EasyOCR"] = (easyocr_result, easyocr_result)  # no FP16 toggle
    else:
        print("  EasyOCR not installed, skipping.")

    # ── PaddleOCR ───────────────────────────────────────────────────
    print("Loading PaddleOCR...")
    paddle_result = measure_paddleocr()
    if paddle_result:
        note = paddle_result.get("note", "")
        print(f"  Allocated (torch view): {fmt_mb(paddle_result['allocated'])}, "
              f"Peak: {fmt_mb(paddle_result['peak'])}")
        if note:
            print(f"  Note: {note}")
        results["PaddleOCR"] = (paddle_result, paddle_result)
    else:
        print("  PaddleOCR/PaddlePaddle not installed, skipping.")

    # ── Summary table ───────────────────────────────────────────────
    print()
    print("=" * 78)
    print(f"{'Model':<22} {'FP32 Alloc':>12} {'FP32 Peak':>12} "
          f"{'FP16 Alloc':>12} {'FP16 Peak':>12} {'Savings':>10}")
    print("-" * 78)

    total_fp16_peak = 0

    for name, (fp32, fp16) in results.items():
        fp32_alloc = fmt_mb(fp32["allocated"])
        fp32_peak = fmt_mb(fp32["peak"])
        fp16_alloc = fmt_mb(fp16["allocated"])
        fp16_peak = fmt_mb(fp16["peak"])

        if fp32["allocated"] > 0:
            savings_pct = (1 - fp16["allocated"] / fp32["allocated"]) * 100
            savings_str = f"{savings_pct:.0f}%"
        else:
            savings_str = "N/A"

        # For OCR models that don't have separate FP16 measurements
        if name in ("EasyOCR", "PaddleOCR"):
            fp32_alloc = fp16_alloc  # same measurement
            fp32_peak = fp16_peak
            savings_str = "N/A"

        print(f"{name:<22} {fp32_alloc:>12} {fp32_peak:>12} "
              f"{fp16_alloc:>12} {fp16_peak:>12} {savings_str:>10}")

        total_fp16_peak += fp16["peak"]

    print("-" * 78)

    # ── Headroom estimate (all models loaded simultaneously in FP16) ─
    print()
    print("Headroom estimate (all models loaded simultaneously, FP16):")
    print(f"  Total peak VRAM:  {fmt_mb(total_fp16_peak)}")
    print(f"  GPU capacity:     {fmt_mb(total_vram)}")
    remaining = total_vram - total_fp16_peak
    print(f"  Remaining:        {fmt_mb(remaining)}")
    print(f"  Utilization:      {total_fp16_peak / total_vram * 100:.1f}%")


if __name__ == "__main__":
    main()
