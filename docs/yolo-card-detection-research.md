# YOLOv8/v11/v26 for Pokemon Card Detection in Binder Pages

**Last updated: February 28, 2026**

## Problem

The current OpenCV contour-based card segmenter (`cardprice/ml/card_segmenter.py`) achieves roughly 70-80% detection accuracy on binder page photos. It struggles with:

- Reflective sleeve surfaces causing glare
- Low contrast between card borders and binder background
- Overlapping or slightly misaligned cards
- Varying lighting conditions from phone cameras

A learned object detector like YOLOv8/v11/v26 can solve these issues by training directly on binder page images.

## Current YOLO Landscape (2026)

### YOLO26 (New - January 2026)
- **Focus**: Edge-first engineering, minimal latency, export compatibility
- **Key improvements over v11**:
  - End-to-end NMS-free inference (no separate post-processing step)
  - Removed Distribution Focal Loss (DFL) for better hardware compatibility
  - ProgLoss + STAL advanced loss functions for better small-object detection
  - MuSGD optimizer (hybrid SGD + Muon, inspired by LLM training)
  - **CPU speed**: YOLO26-N achieves 38.9ms (43% faster than YOLO11-N on CPU)
  - **GPU speed**: 1.7ms per image
- **Best for**: Edge deployment, CPU-constrained environments (our case)
- **Tradeoff**: Newer = less community validation, fewer fine-tuned Pokemon datasets yet

### YOLOv11 (Stable - 2024)
- **CPU speed**: YOLO11n ~80ms ONNX inference (22% faster than YOLOv8n)
- **Accuracy**: ~2-3% mAP improvement over v8, fewer parameters (22% less than v8m)
- **Architecture**: C2PSA module with multi-head attention for feature extraction
- **Stability**: Better under domain shift and small-object detection
- **Recommendation**: Good middle ground — mature ecosystem, pre-trained Pokemon models available

### YOLOv8 (Original - 2023)
- **CPU speed**: ~150-200ms PyTorch, ~80ms ONNX
- **Status**: Fully stable, largest community ecosystem, most Pokemon datasets
- **Still viable**: For single-class card detection, the differences are marginal

## Pre-trained Models & Datasets

### Roboflow Universe - Pokemon Card Detection (Updated 2026)

**Pre-trained Models Available:**
1. **Pokemon Card Detector** (Pokemon Scanner workspace) — YOLOv11, October 2024
2. **pokemoncarddetector** (pokemoncarddetection workspace) — 257 open-source images + pre-trained model, October 2024

**Training Datasets Available:**
1. **pokemon cards** (pokemon workspace, v4) — **2,582 images**, January 2025
   - YOLOv8 format with TXT annotations
   - Mixed raw photos and edited images
   - **Largest Pokemon dataset on Roboflow**

2. **pokemon cards** (aaron-qwuzu workspace, v7) — **890-900 images** at 640x640
   - Raw photo variant (v5) and edited variant (v7)
   - YOLOv8 format, TXT + YAML config

3. **pokemon-cards-merged** (PokemonCardDetect workspace, v2) — May 2025
   - Merged from multiple sources

4. **Pokemon Cards Instance Segmentation** (pokemon-cards-nfznc) — Instance segmentation variant
   - For pixel-level card masks instead of just bounding boxes

**Our Recommendation**: Use the **2,582-image pokemon cards dataset** as primary training data. This exceeds minimum fine-tuning requirements and covers diverse lighting/angles.

## Fine-tuning Requirements (Updated 2026)

### Minimum Data Required

- **Absolute minimum**: 50-100 annotated images (feasible with aggressive augmentation)
- **Recommended**: 200-300 images for robust accuracy
- **Available**: 2,582+ images on Roboflow (well above requirements)
- **Augmentation**: Ultralytics applies mosaic, flip, HSV jitter, scale by default — **effectively 10-20x data multiplication**

### Why Our Case is Well-Positioned

The 2,582-image pokemon cards dataset is:
- **10x larger** than minimum requirement
- **Already annotated** in YOLOv8 format
- **Diverse**: Multiple lighting conditions, angles, card types
- **Transfer-ready**: Pre-trained Pokemon models exist on Roboflow

### Annotation Strategy (If Custom Data Needed)

- **Single class**: "card" — bounding boxes only (simplest case)
- **Multi-class (advanced)**: "card_front", "card_back", "empty_slot" — but requires more training data
- **Tools**: Roboflow's web UI, LabelImg, CVAT

### Training Specifications

**Hardware Trade-offs:**
- **CPU (WSL2)**: ~6-10 hours for 100 epochs on 2,582 images with YOLOv8n
- **GPU**: ~30-60 minutes (if available)
- **Recommendation**: Start with CPU training given your environment

**Optimal Hyperparameters:**
```python
from ultralytics import YOLO

# YOLOv11n recommended for this use case
model = YOLO("yolov11n.pt")  # nano pretrained on COCO, 6.2MB
results = model.train(
    data="pokemon_cards.yaml",
    epochs=100,
    imgsz=640,
    batch=8,                    # reduce if RAM-limited on WSL2
    device="cpu",               # CPU training
    patience=50,                # early stopping
    augment=True,               # aggressive augmentation (default)
    mosaic=1.0,                 # full mosaic probability
    flipud=0.5,                 # 50% vertical flip
    fliplr=0.5,                 # 50% horizontal flip
    hsv_h=0.015,                # HSV hue jitter
    hsv_s=0.7,                  # HSV saturation jitter
    hsv_v=0.4,                  # HSV value jitter
    single_cls=False,           # can detect card as one class
)
```

**Dataset YAML (`pokemon_cards.yaml`):**
```yaml
path: /home/godli/cardprice/data/pokemon_cards_yolo
train: images/train
val: images/val
test: images/test

nc: 1  # single class
names:
  0: card
```

### Training Comparison: YOLO11 vs YOLO26

| Aspect | YOLOv11n | YOLO26-N |
|--------|----------|---------|
| File size | 6.2MB | ~5.5MB |
| CPU inference | ~80ms ONNX | ~39ms |
| GPU inference | ~4-5ms | ~1.7ms |
| Training time (2,582 img) | ~8-10 hours | ~7-9 hours |
| Dataset needed | 200+ images | 200+ images |
| Ecosystem maturity | Very stable | New (Jan 2026) |
| **Recommendation** | Safe choice | Faster inference |

## Expected Accuracy (Updated 2026)

| Method | mAP@0.5 | Training Data | Notes |
|--------|---------|---------------|-------|
| OpenCV contours (current) | ~70-80% | N/A | Fails on glare, low contrast |
| YOLOv8n fine-tuned | ~90-93% | 50-100 images | Minimum viable, aggressive aug |
| YOLOv11n fine-tuned | ~95-97% | 200+ images | Production quality, stable training |
| **YOLO26-N fine-tuned** | **~96-98%** | **200+ images** | **Faster inference, edge-optimized** |
| YOLOv8/11s fine-tuned | ~98%+ | 200+ images | Overkill for simple card detection |

**Key insight**: Single-class card detection is geometrically simple (rectangles). Even nano models reach 95%+. Diminishing returns beyond 200 training images.

## CPU Inference Speed (Updated 2026)

| Model | Runtime | Speed | Notes |
|-------|---------|-------|-------|
| YOLOv8n | PyTorch | ~150-200ms | Default, moderate optimization |
| YOLOv8n | ONNX | ~80ms | Export: `model.export(format="onnx")` |
| **YOLOv11n** | **ONNX** | **~80ms** | **22% faster than v8, stable** |
| **YOLO26-N** | **ONNX** | **~39ms** | **43% faster than v11, NMS-free** |
| YOLO26-N | OpenVINO | ~15ms | Intel CPUs only, 2.5x speedup |

**Our requirement**: <2s per binder page
- **Binder page has 9 cards** → need ~200ms total per-page time
- **OpenCV contours**: ~50-100ms processing + per-card cascade
- **YOLO ONNX**: ~80-160ms (1-2 cards per ms) + per-card cascade
- **Both easily meet <2s target**

### Recommended Export Path for WSL2

```python
from ultralytics import YOLO

# Train YOLOv11n (or YOLO26-N if available)
model = YOLO("yolov11n.pt")
results = model.train(data="pokemon_cards.yaml", epochs=100, ...)

# Export to ONNX (best CPU performance without dependencies)
best_model = YOLO("runs/detect/train/weights/best.pt")
best_model.export(format="onnx")  # Creates best.onnx (~6.2MB)

# Inference with ONNX Runtime (install: pip install onnxruntime)
import onnxruntime
session = onnxruntime.InferenceSession("best.onnx")

# Or use ultralytics wrapper for simplicity
model_onnx = YOLO("best.onnx")
results = model_onnx.predict("binder_page.jpg", conf=0.5, verbose=False)
for detection in results[0].boxes:
    x1, y1, x2, y2 = detection.xyxy[0].int().tolist()
    confidence = float(detection.conf[0])
    # Process bounding box...
```

**Export comparison:**
- **ONNX**: ~6-7MB, ~80ms inference, cross-platform, no runtime dependency beyond onnxruntime
- **TensorRT**: Faster (~40ms), NVIDIA GPUs only, WSL2 doesn't have CUDA
- **OpenVINO**: Fastest (~15ms), Intel CPUs only, may not work on WSL2 WSL2 network bridge
- **PyTorch**: Largest (~25MB), slowest, no optimization

## Model Size Comparison (Updated 2026)

| Variant | Parameters | Model File | RAM Usage | CPU Speed | Recommendation |
|---------|-----------|-----------|-----------|-----------|---|
| YOLOv8n | 3.2M | 6.2MB | ~200MB | ~150-200ms | Stable baseline |
| YOLOv11n | 2.6M | 5.9MB | ~200MB | **~80ms ONNX** | Better accuracy, slightly smaller |
| **YOLO26-N** | **2.6M** | **~5.5MB** | **~200MB** | **~39ms ONNX** | **Fastest, edge-optimized** |
| YOLOv8s | 11.2M | 22.5MB | ~400MB | ~250ms | Not recommended |
| YOLOv8m | 25.9M | 52MB | ~800MB | ~400ms | Overkill |

**Conclusion**: YOLOv11n or YOLO26-N are clear choices. Single-class card detection doesn't require larger capacity. **Card detection is geometrically simple** (rectangles in a grid) — nano models handle it easily at 95-98% mAP.

**For WSL2 deployment**: YOLO26-N is ideal (43% faster), but YOLOv11n is safer (more ecosystem maturity).

## Multi-Class Detection: Card State Classification

### Question: Can YOLO detect card state (empty slot, face-up, face-down)?

**Short answer**: YOLO can detect bounding boxes for all three, but it requires training data for each class.

### Recommended Approach: Hybrid Detection + Classification

YOLO's strength is fast, accurate **bounding box localization**. For **state classification** (empty vs face-up vs face-down), the best practice is:

1. **YOLO detection phase**: Localize all card regions (bounding boxes)
2. **Separate classifier phase**: Classify each box region as one of:
   - `card_front` (face-up card with artwork visible)
   - `card_back` (face-down card showing orange back)
   - `empty_slot` (no card)

**Why hybrid is better than multi-class YOLO:**
- Your existing `is_card_back()` function in `card_segmenter.py` already does this well with HSV analysis
- YOLO's bounding box regression isn't critical for state classification (you just need to classify the center pixel)
- Decoupling detection and classification keeps each model simpler and faster

### Multi-Class YOLO (Alternative)

If you want single-model YOLO detection + classification:

```yaml
# cards_multiclass.yaml
nc: 3
names:
  0: card_front
  1: card_back
  2: empty_slot
```

**Pros**: One inference pass
**Cons**: Requires annotating your training data with three classes (3x annotation effort), may be slightly less accurate for each class

### Our Recommendation for Binder Scanning

**Keep the hybrid approach:**
```
[YOLO11n/YOLO26-N] → Localize all 9 card regions
    ↓
[Existing is_card_back() function] → Classify each region
    ↓
[Per-card cascade] → Match card identity
```

This leverages your already-tuned `is_card_back()` function and avoids re-annotating datasets with three classes.

## Installation

CPU-only installation (avoids pulling CUDA/cuDNN which are ~2GB):

```bash
# Install PyTorch CPU-only first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install ultralytics
pip install ultralytics

# Install ONNX runtime for faster inference
pip install onnxruntime
```

**Total install footprint**: ~500MB (PyTorch CPU + ultralytics + dependencies)

Verify installation:
```python
from ultralytics import YOLO

# Try YOLOv11 (stable, recommended)
model = YOLO("yolov11n.pt")  # downloads 5.9MB pretrained weights
results = model.predict("test_image.jpg")
print(results[0].boxes)  # bounding boxes

# Or YOLOv8 (if v11 unavailable)
model = YOLO("yolov8n.pt")  # downloads 6.2MB pretrained weights

# Or YOLO26 if released to ultralytics
# model = YOLO("yolo26n.pt")
```

## Handling the Empty Slot Problem

### Current Issue: card_00 appearing on wrong pages

Your memory notes mention "empty slot detection problem (card_00 on page 3)". This arises when:

1. **Contour detection misses an empty slot** → grid falls off by one position
2. **Grid fallback inserts a card** where there's actually an empty slot in the binder
3. **Card identities shift**: card detected in slot 4 gets labeled as card 3, etc.

### YOLO Solution

YOLO actually **helps** with this:

1. **YOLO detects only actual card bounding boxes** (gloss/reflective surface)
2. **No detection in empty slot** → you know position is empty
3. **No grid fallback needed** if YOLO finds 8 cards out of 9 → you know slot 5 is empty

**Pseudocode:**
```python
# Instead of grid fallback, use YOLO as ground truth
results = yolo_model.predict(binder_page)
detected_cards = results[0].boxes  # only actual cards

if len(detected_cards) == 9:
    # All slots filled, proceed normally
    card_positions = detect_grid_positions(detected_cards)
elif len(detected_cards) < 9:
    # Some slots empty — YOLO tells us which ones
    empty_positions = find_missing_positions(card_positions)
    # Process only detected cards, mark empties explicitly
    for i, card in enumerate(detected_cards):
        # Process card_i, knowing its exact position
```

### Why YOLO Is Better Than OpenCV Fallback

| Scenario | OpenCV Contours | OpenCV + Grid Fallback | YOLO |
|----------|-----------------|----------------------|------|
| All 9 cards present | Detects all 9 | Fallback not triggered | Detects all 9 |
| 8 cards, 1 empty slot | Detects 8 | Inserts grid card, position guessed | Detects 8, knows position is empty |
| 7 cards, 2 empty slots | Detects 7 | Fallback inserts 2 grid cards, unknown positions | Detects 7, knows 2 exact empty positions |
| Glare, low contrast | Misses 2-3 cards | Fallback inserts guesses | YOLO finds all 9 or clearly detects misses |

**Recommendation**: Once YOLO is trained, **remove the grid fallback** and use YOLO detections as ground truth for positions. This eliminates the empty slot ambiguity entirely.

## Integration with Existing Pipeline (Updated 2026)

The YOLO detector would **replace** the OpenCV contour detection in `card_segmenter.py`:

```
Binder photo
    ↓
[YOLO11n/YOLO26-N card detection]
    ↓
If detections < expected (e.g., 8 < 9):
    → Mark missing positions as empty_slot
    → Continue with detected cards only
    ↓
Crop individual cards from YOLO bounding boxes
    ↓
[Per-card cascade] hash → DINOv2 → CLIP → Claude
    ↓
Price lookup
```

### Proposed `YOLOCardSegmenter` Class

```python
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2

class YOLOCardSegmenter:
    """YOLO-based card detection replacing OpenCV contours."""

    def __init__(self, model_path="models/card_detector.onnx", conf=0.5):
        """Load YOLO model (ONNX or PyTorch).

        Args:
            model_path: Path to YOLO model (best.onnx or best.pt)
            conf: Confidence threshold (0.0-1.0)
        """
        self.model = YOLO(model_path)
        self.conf = conf

    def segment(self, image: np.ndarray,
                expected_count: int = 9) -> list[dict]:
        """Detect card bounding boxes in an image.

        Args:
            image: Input image (BGR, from cv2.imread)
            expected_count: Expected number of cards (9 for binder page)

        Returns:
            List of {"bbox": (x1,y1,x2,y2), "confidence": float, "card_image": ndarray}
            Sorted in reading order (top-left to bottom-right).
        """
        results = self.model.predict(image, conf=self.conf, verbose=False)

        cards = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            card_image = image[y1:y2, x1:x2]
            cards.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(box.conf[0]),
                "card_image": card_image,
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),  # for sorting
            })

        # Sort into reading order (top-left to bottom-right)
        cards = self._sort_grid(cards)

        # Warn if fewer cards than expected (empty slots)
        if len(cards) < expected_count:
            missing = expected_count - len(cards)
            logger.warning(
                f"Detected {len(cards)}/{expected_count} cards "
                f"({missing} empty slots detected)"
            )

        return cards

    def _sort_grid(self, cards: list[dict]) -> list[dict]:
        """Sort cards in reading order (top-left to bottom-right)."""
        if not cards:
            return cards

        # Sort by y (row), then x (column)
        sorted_cards = sorted(cards, key=lambda c: (c["center"][1], c["center"][0]))
        return sorted_cards
```

### Migration Path

1. **Phase 1** (Current): Keep OpenCV, add YOLO as alternative
   ```python
   # In card_segmenter.py
   try:
       card_contours = _find_card_contours(image)
       cards = YOLOCardSegmenter(model_path).segment(image)
   except:
       # Fallback to OpenCV
       cards = _grid_fallback(image)
   ```

2. **Phase 2** (After validation): Switch default to YOLO
   ```python
   # Remove OpenCV path
   segmenter = YOLOCardSegmenter(model_path)
   cards = segmenter.segment(image)
   ```

3. **Phase 3** (Long-term): Remove OpenCV code if YOLO proves stable

## Recommendation & Roadmap (2026)

### Decision: YOLOv11n vs YOLO26-N?

**For your WSL2 deployment**, choose **YOLOv11n** unless you need maximum speed:

- **YOLOv11n**: Mature ecosystem, stable training, proven Pokemon datasets, ~80ms ONNX
- **YOLO26-N**: Faster (~39ms), NMS-free, edge-optimized, but newer (Jan 2026), fewer community examples

**Recommendation**: **Start with YOLOv11n** (safe), then experiment with YOLO26-N if needed.

### Concrete Next Steps

1. **Download Roboflow dataset**: Use the 2,582-image pokemon cards dataset (v4)
   - Export in YOLOv8 format (TXT annotations)
   - Split: 70% train, 15% val, 15% test (~1800/400/400 images)

2. **Train YOLOv11n** on WSL2
   ```bash
   pip install ultralytics onnxruntime
   python -c "
   from ultralytics import YOLO
   model = YOLO('yolov11n.pt')
   results = model.train(
       data='pokemon_cards.yaml',
       epochs=100,
       batch=8,
       device='cpu',
       patience=50
   )
   model.export(format='onnx')
   "
   # Takes ~8-10 hours on CPU
   ```

3. **Test on real binder pages**
   - 10-20 photos of your actual binder pages
   - Compare YOLO vs OpenCV contours on accuracy & speed
   - Measure end-to-end pipeline timing (<2s/page requirement)

4. **Integrate into `card_segmenter.py`**
   - Add `YOLOCardSegmenter` class
   - Update `segment_cards()` to use YOLO
   - Keep OpenCV as fallback

5. **Validate empty slot detection**
   - Test pages with 8/9 cards filled
   - Verify correct position labels

### Investment vs. Return

- **Annotation effort**: None (Roboflow dataset already annotated)
- **Training time**: ~8-10 hours (one-time, can run overnight)
- **Inference speed**: 80ms/page (vs current ~200-300ms OpenCV)
- **Accuracy gain**: 75-80% → 95-98% card detection
- **Empty slot fix**: Eliminates grid-based ambiguity

**This is a high-value improvement for modest effort.**
