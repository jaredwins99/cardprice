# DINOv2 Fine-Tuning for Pokemon Card Matching: Research Notes

**Date:** 2026-02-28
**Status:** Research only -- no implementation decisions made

---

## 1. Can DINOv2 ViT-B/14 Be Fine-Tuned with LoRA on a Small Dataset?

**Yes.** The HuggingFace `facebook/dinov2-base` model is fully compatible with the
`peft` library (HuggingFace PEFT). LoRA adapters can be injected into the
self-attention projection layers.

### Target Modules for LoRA

DINOv2 ViT-B/14 uses standard ViT attention blocks. The relevant linear layers
for LoRA injection are:

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                          # rank (8-16 recommended for small data)
    lora_alpha=16,                # scaling factor
    target_modules=["query", "value"],  # Q and V projections in attention
    lora_dropout=0.1,
    bias="none",
    modules_to_save=None,         # no classification head for retrieval
)
```

Additional targets that can be included: `"key"`, `"output.dense"`,
`"mlp.fc1"`, `"mlp.fc2"`. Research suggests Q+V is the sweet spot for
efficiency vs. performance.

### Model Specs

| Property         | Value          |
| ---------------- | -------------- |
| Architecture     | ViT-B/14       |
| Parameters       | 86M            |
| Hidden size      | 768            |
| Patch size       | 14x14          |
| Input resolution | 224x224        |
| Embedding dim    | 768 (CLS token)|
| Attention heads  | 12             |
| Layers           | 12             |

With LoRA rank=8 on Q+V (24 injected matrices), trainable parameters drop to
roughly **0.3-0.6M** (< 1% of total), while the 86M backbone stays frozen.

---

## 2. Minimum Training Data Requirements

### What We Have

- **20,026 reference card images** (clean, high-quality, one per card)
- **18 phone photo segments** (from binder scans, real-world conditions)
- Effectively a **one-shot retrieval** problem: 1 reference per class, query
  images come from phone cameras

### Data Requirements by Approach

| Approach                    | Min Training Data              | Notes                                   |
| --------------------------- | ------------------------------ | --------------------------------------- |
| Linear probe on frozen features | 1-5 per class (we have 1)  | Already what we do. Limited ceiling.    |
| LoRA fine-tuning (retrieval)    | 50-200 augmented pairs     | Synthetic pairs from augmentation       |
| Full fine-tuning            | 1000+ per class                | Impractical for us. Don't do this.      |
| Learned similarity head     | 100-500 training pairs         | Most practical for our situation        |

### Key Insight

We don't need per-class labels. For metric learning / retrieval fine-tuning,
we need **pairs** (anchor, positive) or **triplets** (anchor, positive,
negative). We can generate these synthetically from our 20k reference images
using augmentation (see Section 4).

**Bottom line:** 20k reference images + synthetic augmentation is sufficient
for LoRA fine-tuning or a learned similarity head. The 18 real phone photos
are useful for validation only.

---

## 3. Linear Probing vs Full Fine-Tuning vs LoRA

### Comparison

| Method              | Trainable Params | Risk of Overfitting | Performance (small data) | Training Time |
| ------------------- | ---------------- | ------------------- | ------------------------ | ------------- |
| Frozen + cosine     | 0                | None                | Baseline (current)       | 0             |
| Linear probe        | ~768 x N         | Low                 | Marginal improvement     | Minutes       |
| LoRA (r=8, Q+V)     | ~0.4M            | Low-Medium          | Significant improvement  | 1-3 hours     |
| LoRA (r=16, all attn)| ~1.5M           | Medium              | Best for retrieval       | 2-5 hours     |
| Full fine-tuning    | 86M              | **Very High**       | **Worse than frozen**    | Hours-days    |

### Research Consensus

Multiple papers (agriculture domain, medical imaging, PCB inspection) report
that **full fine-tuning of DINOv2 on small datasets degrades performance**
compared to frozen features. One practitioner reported accuracy dropping from
75% (linear probe) to <40% (full fine-tuning) even on CIFAR-10.

**LoRA is the clear winner for domain adaptation with limited data.** It
preserves the general visual features learned during self-supervised
pretraining while adapting the attention patterns to the target domain.

### Recommendation for Pokemon Cards

1. **Immediate:** Try a learned similarity head (Section 7) -- lowest risk,
   fastest to implement
2. **If needed:** LoRA with r=8 on Q+V projections using retrieval loss
3. **Avoid:** Full fine-tuning entirely

---

## 4. Synthetic Training Pairs from Reference Images

### The Domain Gap Problem

Our reference images are clean digital scans (uniform lighting, perfect
alignment, no glare). Phone photos introduce:

- Perspective distortion and rotation
- Motion blur and defocus
- Glare and specular highlights
- Color temperature shifts (warm indoor, cool fluorescent)
- Vignetting and lens distortion
- Partial occlusion (sleeve edges, binder rings)
- Background clutter at card edges
- JPEG compression artifacts
- Variable resolution and crop

### Augmentation Pipeline to Simulate Phone Photos

```python
import albumentations as A

phone_camera_augmentation = A.Compose([
    # Geometric
    A.Perspective(scale=(0.02, 0.08), p=0.7),
    A.Rotate(limit=15, border_mode=0, p=0.5),
    A.RandomResizedCrop(224, 224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),

    # Optical
    A.MotionBlur(blur_limit=(3, 7), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(var_limit=(5, 30), p=0.3),

    # Lighting
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05, p=0.5),
    A.RandomShadow(p=0.2),
    A.RandomSunFlare(flare_roi=(0, 0, 1, 1), src_radius=100, p=0.1),

    # Compression
    A.ImageCompression(quality_lower=50, quality_upper=95, p=0.4),

    # Normalize for DINOv2
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Training Pair Generation Strategy

For each training batch:

1. Sample N reference images as anchors
2. Apply `phone_camera_augmentation` to create positive pairs
3. Negatives come from other cards in the batch (in-batch negatives)

This gives us effectively **unlimited training pairs** from 20k references.

### Validation Set

The 18 real phone segments + their known ground truth card IDs form the
validation set. This is small but sufficient to detect overfitting and
measure real-world improvement.

---

## 5. Training Time Estimates on RTX 4070 SUPER

### Hardware Specs

- GPU: RTX 4070 SUPER
- VRAM: 12 GB
- CUDA cores: 7168
- Memory bandwidth: 504 GB/s
- Compute: ~35 TFLOPS FP16

### DINOv2 ViT-B/14 Memory Usage

| Configuration                | Batch Size | VRAM Usage | Notes                   |
| ---------------------------- | ---------- | ---------- | ----------------------- |
| Frozen backbone (inference)  | 32         | ~2.5 GB    | Current setup           |
| LoRA r=8, Q+V, FP16          | 16         | ~5-6 GB    | Comfortable fit         |
| LoRA r=16, all attn, FP16    | 8          | ~7-8 GB    | Tight but workable      |
| Full fine-tune, FP16         | 4          | ~10-11 GB  | Barely fits, avoid      |
| LoRA r=8 + gradient ckpt     | 32         | ~6-7 GB    | Best throughput option  |

### Estimated Training Times

Assuming 20k reference images, generating 2 augmented views per image per
epoch (40k pairs/epoch):

| Approach                      | Epochs | Time/Epoch | Total    |
| ----------------------------- | ------ | ---------- | -------- |
| Learned similarity head only  | 20-50  | ~2 min     | 40-100 min |
| LoRA r=8, Q+V                 | 10-20  | ~8 min     | 1.5-3 hours |
| LoRA r=16, all attention      | 10-20  | ~12 min    | 2-4 hours |

**These are practical training times.** A full experiment cycle (train +
validate + iterate) could be done in an afternoon.

### Memory Optimization Tips

- Use `torch.cuda.amp` (automatic mixed precision) -- halves memory
- Gradient checkpointing reduces memory at ~15% speed cost
- Gradient accumulation allows effective batch sizes > physical batch size
- Keep FAISS index building separate (CPU-only, after training)

---

## 6. Retrieval-Specific Loss Functions

### Why Classification Loss Doesn't Fit

Classification loss (cross-entropy with 20k classes) is impractical:
- 20k output neurons = huge classifier head
- Adding new cards requires retraining the head
- We want embeddings, not class predictions

### Retrieval Losses Ranked by Suitability

#### Tier 1: Best for Our Use Case

**CosFace / ArcFace (margin-based classification)**
- Works with small batch sizes (< 256) -- critical for 12GB VRAM
- Uses 1 image per class per batch
- Adds angular margin penalty to cosine similarity
- State-of-the-art for retrieval with limited GPU resources
- **Caveat:** Still needs a classification head (20k classes), but the head
  is lightweight and the loss drives strong embeddings
- From Berton et al. (2025): "When GPU memory is limited (batch size < 256),
  classification losses like CosFace and ArcFace perform much better than
  contrastive losses"

**InfoNCE / NT-Xent (normalized temperature-scaled cross-entropy)**
- Contrastive loss used by CLIP, SimCLR
- Anchor + positive vs. all other batch elements as negatives
- Works well with moderate batch sizes (64-256)
- Clean implementation, well understood
- Natural fit for our augmentation-based pair generation

#### Tier 2: Viable Alternatives

**Triplet Loss (with online hard mining)**
- Classic retrieval loss: d(anchor, positive) < d(anchor, negative) + margin
- Requires careful mining strategy (hard negatives matter hugely)
- Semi-hard mining recommended over hardest-negative mining
- Slower convergence than InfoNCE

**Multi-Similarity Loss**
- Considers all positive/negative pairs in a batch simultaneously
- Better gradient signal than triplet loss
- Needs batch size >= 64 for effectiveness

#### Tier 3: Not Recommended

**Contrastive Loss (vanilla)**
- Only considers pairs, not relative ranking
- Suboptimal compared to triplet and InfoNCE variants

**Vanilla Cross-Entropy (20k classes)**
- Too many classes, inflexible to new cards

### Recommended Loss: InfoNCE + LoRA

```python
import torch
import torch.nn.functional as F

def info_nce_loss(anchors, positives, temperature=0.07):
    """InfoNCE loss for retrieval fine-tuning.

    anchors:   (B, 768) L2-normalized embeddings of clean reference images
    positives: (B, 768) L2-normalized embeddings of augmented versions
    """
    # Cosine similarity matrix
    logits = torch.mm(anchors, positives.T) / temperature  # (B, B)

    # Positive pairs are on the diagonal
    labels = torch.arange(logits.shape[0], device=logits.device)

    # Symmetric loss
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)

    return (loss_a + loss_b) / 2
```

### Alternative: ArcFace for Maximum Performance

If willing to maintain a 20k-class head:

```python
from pytorch_metric_learning.losses import ArcFaceLoss

# ArcFace with 768-dim embeddings, 20k classes
loss_fn = ArcFaceLoss(
    num_classes=20026,
    embedding_size=768,
    margin=28.6,      # angular margin in degrees
    scale=64,
)
# Learning rate for ArcFace head: 1.0 (much higher than backbone lr of 1e-6)
```

---

## 7. DINOv2 + Adapters for Retrieval

### Adapter Architectures

Beyond LoRA, several adapter methods work with DINOv2:

#### 7a. Bottleneck Adapters

Insert small MLP bottlenecks after each transformer block:

```
Input -> [Frozen Attention] -> [Adapter: Linear(768,64) -> ReLU -> Linear(64,768)] -> Output
```

- ~0.1M parameters per adapter, ~1.2M total for 12 layers
- Similar performance to LoRA, slightly more parameters
- Implementation: HuggingFace `adapters` library or manual

#### 7b. Visual Prompt Tuning (VPT)

Prepend learnable tokens to the input sequence:

- VPT-Shallow: add N learnable tokens to the first layer only
- VPT-Deep: add N learnable tokens to every layer
- Typical N=10-50, adding 7.6k-38k parameters (very few)
- Less effective than LoRA for retrieval tasks per recent benchmarks

#### 7c. Projection Head Adapter (Most Practical)

Add a learnable projection after the frozen CLS token:

```python
class RetrievalAdapter(torch.nn.Module):
    """Lightweight adapter: frozen DINOv2 -> learned projection."""

    def __init__(self, dinov2_model, proj_dim=256):
        super().__init__()
        self.backbone = dinov2_model
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, proj_dim),
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)  # (B, 768)
        projected = self.projector(features)  # (B, proj_dim)
        return F.normalize(projected, dim=-1)
```

- Only ~460k trainable parameters
- Backbone stays completely frozen (fastest training, lowest VRAM)
- Can use lower-dimensional embeddings (256 vs 768) for faster FAISS search
- Trained with InfoNCE or triplet loss on augmented pairs
- **This is the lowest-risk, fastest-to-implement option**

### Adapter Comparison for Our Use Case

| Method                | Params   | VRAM   | Training Time | Risk     | Expected Gain |
| --------------------- | -------- | ------ | ------------- | -------- | ------------- |
| Projection head       | ~460k    | ~3 GB  | 30-60 min     | Very low | Moderate      |
| LoRA r=8 Q+V          | ~400k    | ~6 GB  | 1.5-3 hrs     | Low      | Significant   |
| LoRA r=16 all attn    | ~1.5M    | ~8 GB  | 2-4 hrs       | Medium   | High          |
| Bottleneck adapters   | ~1.2M    | ~7 GB  | 2-3 hrs       | Low      | Significant   |
| VPT-Deep (50 tokens)  | ~38k     | ~4 GB  | 1-2 hrs       | Low      | Low-Moderate  |

---

## 8. Learned Similarity Metric Instead of Cosine

### The Problem with Cosine Similarity

Our current pipeline uses raw cosine similarity on DINOv2 CLS embeddings.
This treats all 768 dimensions equally, but some dimensions may encode
features irrelevant to card identity (e.g., lighting conditions, background
texture) while others encode critical features (card art, text, borders).

### Option A: Learned Mahalanobis Distance

Learn a PSD matrix W such that `d(a,b) = (a-b)^T W (a-b)`.
Equivalent to learning a linear projection `L` where `W = L^T L`.

```python
class MahalanobisMetric(torch.nn.Module):
    def __init__(self, dim=768, proj_dim=256):
        super().__init__()
        self.L = torch.nn.Linear(dim, proj_dim, bias=False)

    def forward(self, a, b):
        a_proj = self.L(a)  # (B, proj_dim)
        b_proj = self.L(b)  # (B, proj_dim)
        return F.cosine_similarity(a_proj, b_proj, dim=-1)
```

- Effectively the same as the projection head adapter (Section 7c)
- ~196k parameters for 768->256

### Option B: MLP Similarity Scorer

Learn a non-linear similarity function:

```python
class MLPSimilarity(torch.nn.Module):
    """Score similarity between two embeddings with a small MLP."""

    def __init__(self, dim=768):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim * 3, 512),  # concat + element-wise product
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, a, b):
        combined = torch.cat([a, b, a * b], dim=-1)  # (B, 768*3)
        return self.net(combined).squeeze(-1)          # (B,)
```

- ~1.2M parameters
- Can capture non-linear similarity patterns
- **Problem:** Requires O(N) forward passes per query (can't use FAISS)
- Only viable as a **re-ranker** on top-k FAISS results

### Option C: Two-Stage Pipeline (Best of Both Worlds)

```
Query Image
    |
    v
[Frozen DINOv2] -> 768-dim embedding
    |
    v
[Learned Projection] -> 256-dim embedding  (trained with InfoNCE)
    |
    v
[FAISS Index Search] -> top-20 candidates   (fast, sub-millisecond)
    |
    v
[MLP Re-ranker] -> final top-5             (accurate, 20 forward passes)
```

This is the architecture used by production retrieval systems. The projection
handles the domain gap (phone vs. reference), and the re-ranker handles
fine-grained discrimination (similar cards from the same set).

### Recommendation

Start with **Option A (learned projection)** -- it's simple, compatible
with FAISS, and addresses the core domain gap problem. Add the MLP
re-ranker only if the projection alone doesn't achieve sufficient accuracy
on real phone photos.

---

## 9. HuggingFace PEFT Compatibility

### Confirmed: facebook/dinov2-base supports PEFT

The `facebook/dinov2-base` model from HuggingFace Transformers is a standard
`Dinov2Model` class that inherits from `PreTrainedModel`, making it fully
compatible with the PEFT library.

```python
from transformers import Dinov2Model
from peft import get_peft_model, LoraConfig

# Load base model
model = Dinov2Model.from_pretrained("facebook/dinov2-base")

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: trainable params: ~400k || all params: ~86.6M || trainable%: ~0.46%
```

### Alternative: torch.hub DINOv2

Our current code uses `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`,
which loads the model directly from Meta's repo. This model is a raw `nn.Module`
and is NOT directly compatible with HuggingFace PEFT.

Options:
1. **Switch to HuggingFace** `Dinov2Model.from_pretrained("facebook/dinov2-base")`
   for PEFT compatibility (recommended)
2. **Manual LoRA injection** on the torch.hub model using `loralib` or custom code
3. **Use the projection head approach** (Section 7c) which works with any model

### Migration Concern

The HuggingFace model and torch.hub model should produce identical embeddings
(same weights), but the output format differs:

```python
# torch.hub: returns CLS token directly
embedding = model(tensor)  # (B, 768)

# HuggingFace: returns a BaseModelOutputWithPooling
output = model(tensor)
embedding = output.last_hidden_state[:, 0]  # CLS token, (B, 768)
# or
embedding = output.pooler_output  # (B, 768) if pooler is configured
```

The existing FAISS index would remain valid as long as we ensure the
embeddings are numerically identical.

---

## 10. Implementation Roadmap (If Pursued)

### Phase A: Learned Projection Head (1-2 days)

1. Keep DINOv2 frozen (torch.hub version, no migration needed)
2. Add `RetrievalAdapter` (Section 7c) on top
3. Build augmentation pipeline (Section 4)
4. Train with InfoNCE loss for 30-50 epochs
5. Rebuild FAISS index with projected 256-dim embeddings
6. Validate on 18 real phone segments
7. Compare cosine similarities vs. current baseline

### Phase B: LoRA Fine-Tuning (2-3 days)

1. Migrate to HuggingFace `Dinov2Model` (or use manual LoRA)
2. Apply LoRA config (r=8, Q+V)
3. Train with InfoNCE loss, augmented pairs
4. Merge LoRA weights for inference (no overhead at query time)
5. Rebuild FAISS index with fine-tuned embeddings
6. Validate and compare vs. Phase A

### Phase C: Re-Ranker (1-2 days, only if needed)

1. Train MLP re-ranker on top-k FAISS results
2. Use hard negatives from FAISS (top-20 results that are wrong)
3. Integrate into `MatchPipeline.match()` after FAISS search

---

## 11. Key Risks and Mitigations

| Risk | Mitigation |
| ---- | ---------- |
| Overfitting on synthetic augmentations (model learns augmentation artifacts, not domain gap) | Validate on real phone photos; use diverse augmentation; early stopping |
| Breaking existing embeddings (FAISS index invalidated) | Keep old index as fallback; A/B test old vs. new pipeline |
| Catastrophic forgetting (LoRA damages general features) | LoRA rank <= 16 to limit capacity; monitor loss on held-out reference pairs |
| Training data too homogeneous (all reference images are similar style) | Include negatives from visually similar cards (same set, same Pokemon) |
| 12 GB VRAM insufficient | Use gradient checkpointing, FP16, small batch + gradient accumulation |

---

## 12. Summary of Recommendations

1. **Start with the projection head adapter** (Section 7c). Lowest risk,
   fastest to implement (1 day), no model migration, compatible with existing
   FAISS pipeline. Expected to close a significant portion of the phone-to-
   reference domain gap.

2. **Use InfoNCE loss** with in-batch negatives. Works at batch size 16-32,
   clean implementation, proven for retrieval.

3. **Generate synthetic pairs** via aggressive augmentation of reference
   images to simulate phone camera conditions. 20k references are more than
   enough.

4. **If projection head isn't sufficient**, move to LoRA (r=8, Q+V) with
   the HuggingFace model. This adapts the attention patterns themselves,
   which is more powerful but requires model migration.

5. **Avoid full fine-tuning.** The research consensus is clear: full
   fine-tuning of DINOv2 on small datasets hurts performance.

6. **Training is fast.** Even the heaviest option (LoRA on all attention
   layers) takes 2-4 hours on the RTX 4070 SUPER.

---

## Sources

- [DINOv2 Paper](https://arxiv.org/abs/2304.07193) -- Oquab et al., Meta AI
- [facebook/dinov2-base on HuggingFace](https://huggingface.co/facebook/dinov2-base)
- [HuggingFace PEFT Library](https://github.com/huggingface/peft)
- [All You Need to Know About Training Image Retrieval Models](https://arxiv.org/abs/2503.13045) -- Berton et al., 2025
- [DINOv2 Fine-Tuning Tutorial](https://kili-technology.com/data-labeling/computer-vision/dinov2-fine-tuning-tutorial-maximizing-accuracy-for-computer-vision-tasks) -- Kili Technology
- [DINOv2 Fine-Tuning: Transfer Learning vs Full Fine-Tuning](https://debuggercafe.com/dinov2-for-image-classification-fine-tuning-vs-transfer-learning/) -- Debugger Cafe
- [DINOv2 Engineer's Deep Dive](https://www.lightly.ai/blog/dinov2) -- Lightly AI
- [dinov3-finetune (LoRA for DINOv2)](https://github.com/RobvanGastel/dinov3-finetune) -- Rob van Gastel
- [Foundation Vision Models in Agriculture: DINOv2, LoRA and Knowledge Distillation](https://www.sciencedirect.com/science/article/abs/pii/S0168169925010063) -- ScienceDirect, 2025
- [PCB Defect Inspection with Few-Shot Adaptation](https://www.mdpi.com/2313-433x/11/11/415) -- MDPI, 2025
- [DINOv2 High GPU Memory Usage (Issue #553)](https://github.com/facebookresearch/dinov2/issues/553)
- [Contrastive Loss Comparison: InfoNCE vs Triplet vs NT-Xent](https://aicompetence.org/contrastive-loss-infonce-vs-triplet-vs-nt-xent/)
