# Few-Shot Learning for Pokemon Card Identification

Research document covering approaches to improve card matching from phone photos
against a reference database of ~20k clean card images.

**Date**: 2026-02-28
**Current performance**: ~60% DINOv2 top-1 / ~80% CLIP similarity / hash only works for clean-to-clean
**Target**: >90% top-1 accuracy on phone photos
**Hardware**: RTX 4070 SUPER (12GB VRAM), CPU fallback required for inference

---

## Table of Contents

1. [Current Pipeline Analysis](#1-current-pipeline-analysis)
2. [Approach 1: Tip-Adapter (Training-Free, Recommended First)](#2-approach-1-tip-adapter)
3. [Approach 2: DINOv2 Linear Probe](#3-approach-2-dinov2-linear-probe)
4. [Approach 3: DINOv2 LoRA Fine-Tuning](#4-approach-3-dinov2-lora-fine-tuning)
5. [Approach 4: CLIP Fine-Tuning with Contrastive Loss](#5-approach-4-clip-fine-tuning)
6. [Approach 5: Siamese Networks](#6-approach-5-siamese-networks)
7. [Approach 6: Prototypical Networks](#7-approach-6-prototypical-networks)
8. [Approach 7: Triplet Loss with Hard Negative Mining](#8-approach-7-triplet-loss)
9. [Approach 8: SimCLR/MoCo Contrastive Pretraining](#9-approach-8-simclr-moco)
10. [Approach 9: Meta-Learning (MAML, Reptile)](#10-approach-9-meta-learning)
11. [Industry Reference: PokeScope Architecture](#11-industry-reference-pokescope)
12. [Recommended Implementation Order](#12-recommended-implementation-order)
13. [Sources](#13-sources)

---

## 1. Current Pipeline Analysis

### What We Have

The cascade pipeline (`cardprice/ml/__init__.py`) runs:

1. **Perceptual hash** (phash, distance < 5) -- instant, but only matches clean-to-clean
2. **DINOv2 ViT-B/14 + FAISS** (cosine sim > 0.65) -- 768-dim embeddings, ~1s
3. **CLIP ViT-L/14 image-to-image** (cosine sim > 0.75) -- 768-dim embeddings, ~2s
4. **Claude Haiku vision API** (fallback, costs money)

### The Domain Gap Problem

The core issue is **domain shift** between reference images (clean digital scans from
pokemontcg.io) and query images (phone photos through plastic sleeves). This manifests as:

- Lighting variation (glare, shadows, uneven illumination)
- Perspective distortion (angle, rotation)
- Sleeve reflections and scratches
- Background clutter from binder pages
- Color shift from phone camera processing
- Compression artifacts

DINOv2 and CLIP were trained on general internet images. Their embeddings capture
semantic content well but are not optimized for the specific invariances needed for
card matching (ignore glare, ignore sleeve texture, focus on card art and text).

### Available Training Data

- **20,026 clean reference images** in `data/card_images/` (one per card variant)
- **18 phone photos** from 2 binder page scans (segmented via `card_segmenter.py`)
- **Unlimited synthetic augmentation** possible from the reference images

The 18 real phone photos are precious -- they represent actual domain gap examples.
Any approach must work with this tiny real-world sample while leveraging the large
reference set.

---

## 2. Approach 1: Tip-Adapter (Training-Free, Recommended First)

### Overview

Tip-Adapter constructs a non-parametric adapter on top of frozen CLIP by building a
key-value cache from few-shot examples. It requires **zero training** -- just
forward passes to build the cache.

### How It Works for Card Matching

1. Encode all 20k reference images with CLIP image encoder -> cache keys (N x 768)
2. Create one-hot labels for each card_id -> cache values (N x 20k)
3. At query time, compute similarity between query embedding and cache keys
4. Weight the cache values by similarity to produce class logits
5. Combine with CLIP's zero-shot logits (text-based) via learned or fixed alpha

### Adaptation for Our Use Case

Since we have 20k classes each with exactly 1 reference image (1-shot), Tip-Adapter
becomes equivalent to nearest-neighbor search with a learned temperature scaling.
The key insight is that Tip-Adapter also incorporates CLIP's text understanding:

```
logits = alpha * clip_text_logits + (1 - alpha) * cache_logits
```

This combines visual similarity (cache) with semantic understanding (text), which
can help disambiguate cards that look similar but have different names/sets.

### Tip-Adapter-F (Fine-Tuned Variant)

With our 18 phone-photo examples, we can fine-tune the cache model's residual:
- Freeze CLIP backbone entirely
- Train only the adapter weights (a single linear layer)
- Training takes seconds on CPU for 18 examples
- Adds domain adaptation without catastrophic forgetting

### Implementation Estimate

- **Training time**: 0 (training-free) or ~1 minute for Tip-Adapter-F
- **Trainable parameters**: 0 (or ~1.5M for the fine-tuned variant)
- **Integration effort**: Low -- builds on existing CLIP embeddings
- **Expected improvement**: 5-10% over raw CLIP cosine similarity

### Key Advantage

This is the **lowest-risk, fastest-to-implement** approach. It reuses the existing
CLIP image index and adds a principled combination with text matching. Even if the
improvement is modest, it costs nothing to try.

### References

- [Tip-Adapter paper (ECCV 2022)](https://arxiv.org/abs/2207.09519)
- [GitHub implementation](https://github.com/gaopengcuhk/Tip-Adapter)
- [Proto-Adapter (training-free variant)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175357/)

---

## 3. Approach 2: DINOv2 Linear Probe

### Overview

Add a trained linear classifier on top of frozen DINOv2 features. The backbone
stays frozen; only a single linear layer (768 -> N_classes) is trained.

### Why This Might Work

DINOv2 ViT-B/14 produces 768-dim CLS embeddings that capture fine-grained visual
features. A linear probe can learn which **dimensions** of the embedding are most
relevant for card identity (vs. lighting, angle, etc.).

### The Challenge: 20k Classes, 1 Shot Each

Standard linear probing needs multiple examples per class. With 1-shot (clean refs
only), the linear probe degenerates to learned feature weighting + nearest neighbor.
This is still useful -- it can learn to downweight embedding dimensions that capture
lighting/angle variation and upweight dimensions that capture card art/text.

### Augmentation Strategy

Generate synthetic "phone photos" from clean references:

```python
augment = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.15, p=0.8),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.RandomRotation(degrees=5),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    # Simulate sleeve glare
    # (custom transform: random bright elliptical overlay)
])
```

For each of the 20k cards, generate 5-10 augmented versions. This gives 100-200k
training pairs. The 18 real phone photos serve as validation to tune augmentation
parameters.

### Implementation

```python
import torch
import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, embed_dim=768, num_classes=20000):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
```

### Training Estimate

- **Trainable parameters**: 768 * 20,000 = ~15.4M (just the linear layer)
- **Training time**: ~30 min on RTX 4070 SUPER (forward passes through frozen DINOv2 +
  linear layer backprop), or ~2-4 hours on CPU
- **Memory**: Embedding matrix is small; DINOv2 inference is the bottleneck
- **Expected improvement**: 5-15% over raw cosine similarity

### Practical Concern

With 20k classes and 1-shot, this is really a metric learning problem, not a
classification problem. The linear probe approach is best if we frame it as
learning a projection that maximizes separability in the embedding space, not
as a 20k-way softmax classifier.

Better formulation: **learn a linear projection W** such that
`sim(W @ query_emb, W @ ref_emb)` is maximized for correct pairs.
This reduces to a 768x768 matrix (590K parameters) trained with contrastive loss.

---

## 4. Approach 3: DINOv2 LoRA Fine-Tuning

### Overview

Low-Rank Adaptation (LoRA) injects small trainable matrices into the attention
layers of frozen DINOv2, allowing the model to adapt its representations to the
card matching domain without full fine-tuning.

### How LoRA Works

For each attention weight matrix W (Q, K, V projections), LoRA adds:

```
W' = W + alpha * (B @ A)
```

Where A is (r x d_in) and B is (d_out x r), with r << d_in, d_out.
Only A and B are trained; W stays frozen.

### Parameter Counts for DINOv2 ViT-B/14

- Full model: 86M parameters
- LoRA rank=4, applied to Q and V: ~200K trainable parameters
- LoRA rank=8, applied to Q and V: ~400K trainable parameters
- LoRA rank=16, applied to Q, K, V, and output: ~1.6M trainable parameters

### Training Strategy

1. Pre-extract augmented phone-photo-style images from all 20k refs
2. Train with contrastive loss: (augmented_query, clean_ref) should be close;
   (augmented_query, wrong_ref) should be far
3. Use the 18 real phone photos as validation
4. Early stopping when validation loss plateaus

### Key Research Findings

From recent 2025 literature:
- LoRA fine-tuned DINOv2 shows 1.7% improvement over standard fine-tuning while
  using 85M fewer parameters and training 25.8% faster
- Typical LoRA ranks r=4-32 produce 1-3M trainable parameters atop 86M frozen backbone
- LoRA excels at out-of-distribution generalization, which is exactly our problem
  (clean refs = in-distribution, phone photos = out-of-distribution)

### Implementation with PEFT

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["qkv"],  # DINOv2 uses fused QKV
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(dino_model, lora_config)
# model.print_trainable_parameters()
# trainable params: ~400K || all params: ~86M || trainable%: 0.47%
```

### Training Estimate

- **Trainable parameters**: 200K-1.6M (depending on rank)
- **Training time**: ~1-2 hours on RTX 4070 SUPER, ~8-12 hours on CPU (not recommended)
- **Memory**: ~4GB VRAM (frozen backbone + LoRA adapters + small batch)
- **Expected improvement**: 10-20% over raw DINOv2 cosine similarity

### Risk

The main risk is overfitting to synthetic augmentations. If our augmentations
don't accurately simulate real phone photo artifacts, the LoRA adapter may learn
the wrong invariances. The 18 real photos are critical for validation.

---

## 5. Approach 4: CLIP Fine-Tuning with Contrastive Loss

### Overview

Fine-tune CLIP's visual encoder (or a lightweight adapter on top) using contrastive
loss on (phone_photo, clean_reference) pairs. This directly addresses the domain gap.

### CLIP-Adapter Approach (Recommended over Full Fine-Tuning)

Instead of modifying CLIP's weights, CLIP-Adapter adds a small residual adapter:

```python
class CLIPAdapter(nn.Module):
    def __init__(self, embed_dim=768, reduction=4):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction),
            nn.ReLU(),
            nn.Linear(embed_dim // reduction, embed_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        adapted = self.adapter(x)
        return self.alpha * adapted + (1 - self.alpha) * x
```

### Supervised Contrastive Loss

From 2025 research (Fine-Tuning of CLIP in Few-Shot Scenarios via Supervised
Contrastive Learning), the approach uses a visual adapter at the end of CLIP's
visual encoder with supervised contrastive loss to alleviate overfitting:

```python
def supervised_contrastive_loss(features, labels, temperature=0.07):
    """SupCon loss: pull same-class features together, push different apart."""
    similarity = torch.matmul(features, features.T) / temperature
    # Mask positive pairs (same label)
    mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    # Remove self-similarity
    mask.fill_diagonal_(0)
    # Log-sum-exp trick for numerical stability
    logits_max, _ = similarity.max(dim=1, keepdim=True)
    logits = similarity - logits_max.detach()
    exp_logits = torch.exp(logits) * (1 - torch.eye(len(labels)).to(features.device))
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
    mean_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return -mean_log_prob.mean()
```

### Training Data Generation

For contrastive pairs:
- **Positive pair**: (augmented phone-style image of card X, clean reference of card X)
- **Hard negative**: (augmented image of card X, clean reference of card Y where Y is
  visually similar -- same Pokemon, same set, different number)

Hard negative mining is critical: two Charizard cards from different sets are much
harder to distinguish than a Charizard and a Pikachu.

### Implementation Estimate

- **Trainable parameters**: ~150K (adapter only) to ~300M (full ViT-L/14, not recommended)
- **Training time**: 30-60 min on RTX 4070 SUPER for adapter, not practical on CPU for full model
- **Memory**: ~6GB VRAM with adapter, frozen backbone
- **Expected improvement**: 10-15% with adapter, potentially 20%+ with full fine-tuning

### Key Insight from PokeScope

PokeScope (a commercial Pokemon card scanner with 50k+ users) uses CLIP fine-tuned
specifically on Pokemon cards combined with OCR for card number extraction. They report
95%+ accuracy. Their dataset includes 10,000+ manual photographs of cards in every
possible sleeve configuration. We cannot match their dataset size, but their architecture
validates that CLIP + domain-specific fine-tuning is the right direction.

---

## 6. Approach 5: Siamese Networks

### Overview

A Siamese network uses two identical encoder branches that share weights. Both the
query image and reference image are encoded, and a learned similarity function
determines if they match.

### Architecture

```
Query Photo  -->  [Encoder]  -->  embedding_q  --\
                                                   --> [Distance] --> match/no-match
Reference    -->  [Encoder]  -->  embedding_r  --/
```

The encoder can be DINOv2 or CLIP (frozen or fine-tuned). The distance function
can be learned (a small MLP on concatenated/subtracted embeddings) or fixed (cosine).

### One-Shot Card Matching

This is the classic one-shot learning setup: given a query image, compare it against
each of the 20k reference images and return the most similar.

For practical deployment, we do NOT run 20k forward passes. Instead:
1. Pre-compute reference embeddings (already done in our FAISS index)
2. At query time, encode the query once
3. Use FAISS for efficient nearest-neighbor search

So the Siamese network architecture is **already what we're doing**. The question
is how to train the encoder to produce better embeddings for our domain.

### Training with Binary Cross-Entropy

```python
# Positive pair: same card, one clean + one augmented
# Negative pair: different cards
loss = BCELoss(distance(enc(query), enc(ref)), label)
```

### Practical Assessment

A Siamese network with a frozen backbone + learned similarity head is equivalent
to the linear probe approach above. With a fine-tuned backbone, it's equivalent
to the LoRA/adapter approaches. Siamese framing is useful for understanding the
problem but doesn't add a unique technique beyond what LoRA/adapter provide.

**Verdict**: Siamese is the right *framing* for our problem, but the specific
*training technique* (LoRA, adapter, linear probe) matters more than the Siamese
label itself.

---

## 7. Approach 6: Prototypical Networks

### Overview

Prototypical Networks learn an embedding space where classification is performed by
computing distances to class prototypes (mean embeddings of each class).

### Direct Application

In our setup, each card has exactly 1 reference image, so the "prototype" for
each class is just the single reference embedding. This makes Prototypical Networks
degenerate to nearest-neighbor in embedding space -- again, what we already do.

### Where Prototypical Networks Add Value

If we had **multiple examples per class** (e.g., multiple photos of the same card
under different conditions), the prototype would be the mean embedding, which is
more robust than any single example.

### Synthetic Prototype Enrichment

We can create multiple augmented versions of each reference image and use the
mean embedding as the prototype:

```python
def compute_prototype(ref_image, augment_fn, n_augments=10, encoder=dino_model):
    embeddings = [encoder(ref_image)]  # original clean reference
    for _ in range(n_augments):
        aug = augment_fn(ref_image)
        embeddings.append(encoder(aug))
    prototype = torch.stack(embeddings).mean(dim=0)
    prototype = prototype / prototype.norm()  # re-normalize
    return prototype
```

This creates a "domain-averaged" prototype that's more robust to the augmentation
distribution (and by proxy, to real phone photo variations).

### Implementation Estimate

- **Training time**: ~2-4 hours to re-encode all 20k cards x 10 augmentations
  (200k forward passes through DINOv2)
- **Trainable parameters**: 0 (just re-encoding with augmentations)
- **Expected improvement**: 3-8% (modest, but zero-risk)
- **Storage**: 20k prototypes x 768 dims x 4 bytes = ~60MB (same as current index)

### Episodic Training (Advanced)

If we want to train the encoder, we use episodic training:
1. Sample N classes, K examples per class (K-shot)
2. Compute prototypes from support set
3. Classify query images by distance to prototypes
4. Backpropagate through the encoder

This requires multiple examples per class, so we'd use augmented images as the
support set and different augmentations as the query set.

---

## 8. Approach 7: Triplet Loss with Hard Negative Mining

### Overview

Triplet loss directly optimizes the embedding space to place matching images closer
together and non-matching images farther apart:

```
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

### Hard Negative Mining

The key to effective triplet training is mining **hard negatives** -- negative
examples that are close to the anchor in embedding space. For Pokemon cards:

- **Hard negatives**: Same Pokemon, different set (e.g., two different Pikachu cards)
- **Semi-hard negatives**: Same type/rarity, different Pokemon
- **Easy negatives**: Completely different cards (Trainer vs. Pokemon)

### PyTorch Metric Learning Implementation

```python
from pytorch_metric_learning import losses, miners

loss_fn = losses.TripletMarginLoss(margin=0.2)
miner = miners.TripletMarginMiner(
    margin=0.2,
    type_of_triplets="semihard"  # start with semihard, move to hard
)

# In training loop:
embeddings = encoder(batch_images)
hard_pairs = miner(embeddings, labels)
loss = loss_fn(embeddings, labels, hard_pairs)
```

### Mining Strategy for Card Matching

Pre-compute similarity matrix between all 20k reference embeddings. For each card,
identify its k-nearest neighbors. These are the hard negatives.

Cards in the same set with the same Pokemon name are the hardest negatives and should
be oversampled during training.

### Training Estimate

- **Trainable parameters**: Depends on what's being trained (adapter: ~150K, LoRA: ~400K)
- **Training time**: ~1-2 hours on RTX 4070 SUPER
- **Memory**: ~4-6GB VRAM
- **Expected improvement**: 10-15% when combined with LoRA or adapter fine-tuning

### Comparison with Contrastive Loss

Triplet loss is more sample-efficient than contrastive loss (InfoNCE) because it
directly specifies the desired relationship between anchor, positive, and negative.
However, it's sensitive to margin selection and can be slow to converge without
good mining.

For our use case, **supervised contrastive loss** (SupCon) may be better than
pure triplet loss because it handles multiple positives/negatives per anchor
within a batch, making better use of each forward pass.

---

## 9. Approach 8: SimCLR/MoCo Contrastive Pretraining

### Overview

SimCLR and MoCo are self-supervised contrastive learning methods that learn visual
representations by treating augmented views of the same image as positive pairs.

### SimCLR

- Creates two augmented views of each image
- Encodes both through the same network
- Pulls augmented views of the same image together, pushes different images apart
- **Requires large batch sizes** (4096+ for best results) due to in-batch negatives

### MoCo (Momentum Contrast)

- Maintains a momentum-updated encoder and a queue of negative examples
- **Does NOT require large batch sizes** (256 is sufficient)
- The queue provides a large pool of negatives without increasing batch size
- More suitable for our hardware constraints

### Application to Card Matching

We can pretrain (or fine-tune) a vision encoder using MoCo on our 20k reference images:

1. Each reference image generates augmented pairs (simulating phone photos)
2. MoCo learns to be invariant to the augmentations
3. The resulting encoder should be more robust to real phone photo variations

### Critical Limitation: Small Dataset

SimCLR and MoCo were designed for large datasets (ImageNet: 1.28M images).
With only 20k images, self-supervised pretraining from scratch is unlikely to learn
good representations. However, **fine-tuning** a pretrained model (DINOv2 or CLIP)
using MoCo-style contrastive objectives on our card images is promising.

This is essentially what LoRA + contrastive loss (Approach 3 + 7) achieves, but
with the MoCo queue mechanism for efficient negative sampling.

### Research Finding

From the literature: "Self-supervised pretraining with contrastive methods deals with
small datasets by pre-training on large datasets and fine-tuning on small ones." This
validates using pretrained DINOv2/CLIP as the starting point rather than training from
scratch.

### Implementation Estimate

- **Training time**: ~2-4 hours on RTX 4070 SUPER (MoCo-style fine-tuning of DINOv2)
- **Batch size**: 256 with MoCo queue (fits in 12GB VRAM)
- **Expected improvement**: 10-20% if augmentations are well-calibrated
- **Risk**: High -- complex to implement, many hyperparameters, unclear benefit over
  simpler LoRA + contrastive loss

**Verdict**: Not recommended as a standalone approach. The contrastive objective is
useful but should be applied via simpler LoRA or adapter training (Approaches 3-4).

---

## 10. Approach 9: Meta-Learning (MAML, Reptile)

### Overview

Model-Agnostic Meta-Learning (MAML) learns an initialization of model weights that
can be quickly adapted to new tasks with very few examples. Reptile is a simpler
first-order approximation.

### How It Could Work

1. **Meta-training**: Simulate many N-way K-shot episodes using card images
2. **Meta-testing**: Given a new phone photo, adapt the model with 1-2 gradient steps
   using the reference image as the support set

### MAML for Card Matching

```python
# Outer loop: across episodes
for episode in episodes:
    # Inner loop: adapt to this episode's support set
    support_images, support_labels = episode.support  # K reference images
    query_images, query_labels = episode.query        # phone photos to classify

    # Clone model parameters
    adapted_params = clone(model.parameters())

    # Inner gradient steps (1-5 steps)
    for step in range(inner_steps):
        support_loss = loss_fn(model(support_images, adapted_params), support_labels)
        adapted_params = adapted_params - inner_lr * grad(support_loss, adapted_params)

    # Outer loss on query set
    query_loss = loss_fn(model(query_images, adapted_params), query_labels)
    outer_optimizer.step(query_loss)
```

### Practical Concerns

1. **Computational cost**: MAML requires second-order gradients (or first-order
   approximation), making it 2-3x slower than standard training
2. **Complexity**: Significant implementation effort
3. **Unclear benefit**: For our 1-shot retrieval problem, simpler metric learning
   approaches (LoRA + contrastive loss) are likely sufficient
4. **20k-way classification**: MAML episodes typically use 5-20 classes. Scaling
   to 20k classes per episode is impractical

### Reptile (Simpler Alternative)

```python
for episode in episodes:
    task_model = deepcopy(model)
    # Train on episode for a few steps
    for step in range(inner_steps):
        loss = loss_fn(task_model(support), labels)
        loss.backward()
        optimizer.step()
    # Move meta-model toward task-adapted model
    for p, q in zip(model.parameters(), task_model.parameters()):
        p.data += meta_lr * (q.data - p.data)
```

### Assessment

Meta-learning is theoretically elegant for few-shot learning but adds significant
complexity without clear advantages over simpler approaches for our specific problem.
Our problem is closer to "domain adaptation with abundant source data" than "few-shot
classification with novel classes."

**Verdict**: Not recommended. The complexity-to-benefit ratio is poor for our use case.

---

## 11. Industry Reference: PokeScope Architecture

PokeScope is a commercial Pokemon card scanner with 50k+ users that achieves 95%+ accuracy.

### Their Stack

1. **YOLOv8** for card detection in photos (similar to our OpenCV segmenter)
2. **CLIP fine-tuned on Pokemon cards** for visual matching
3. **OCR** for card number extraction (bottom corner of each card)
4. **Hybrid scoring**: CLIP similarity + OCR card number verification

### Key Technical Decisions

- CLIP provides semantic understanding beyond pixel matching
- CLIP handles bad lighting, angles, and holo patterns well
- OCR is the tiebreaker for visually identical cards with different numbers
- **10,000+ manual photographs** in every sleeve configuration for training data

### Lessons for Our Project

1. **CLIP fine-tuning works** -- PokeScope validates this approach at scale
2. **OCR is essential** for distinguishing similar cards (we have Tesseract in
   `MatchPipeline._ocr_extract_text()` already)
3. **Sleeve-specific training data** matters -- our 18 binder photos are a start
   but ideally we need more variety in sleeve types and lighting
4. **Hybrid approaches beat single-model** -- combining visual + text features
   outperforms either alone

---

## 12. Recommended Implementation Order

Based on cost-benefit analysis, implementation complexity, and hardware constraints:

### Phase A: Quick Wins (1-2 days, no training)

1. **Synthetic Prototype Enrichment** (Section 7)
   - Generate 10 augmented versions of each reference image
   - Average embeddings to create domain-robust prototypes
   - Rebuild FAISS index with averaged prototypes
   - Expected: +3-8% accuracy
   - Risk: Very low

2. **Tip-Adapter** (Section 2)
   - Build key-value cache from existing CLIP embeddings
   - Combine visual cache similarity with text matching
   - No training required
   - Expected: +5-10% accuracy
   - Risk: Very low

3. **Better Augmentations for Existing Pipeline**
   - Use the 18 real phone photos to calibrate augmentation parameters
   - Compare augmented images to real photos, tune until distributions match
   - Apply calibrated augmentations to prototype enrichment

### Phase B: Lightweight Training (3-5 days, GPU)

4. **CLIP-Adapter with Contrastive Loss** (Section 5)
   - Train a small adapter (~150K params) on CLIP's visual encoder
   - Use supervised contrastive loss with augmented pairs
   - Validate on the 18 real phone photos
   - Expected: +10-15% accuracy
   - Risk: Low-medium

5. **DINOv2 LoRA Fine-Tuning** (Section 4)
   - Apply LoRA (rank=8) to DINOv2 attention layers
   - Train with triplet loss + hard negative mining
   - Pre-mine hard negatives from existing FAISS index
   - Expected: +10-20% accuracy
   - Risk: Medium

### Phase C: Full Pipeline (1-2 weeks)

6. **OCR Integration** (inspired by PokeScope)
   - Extract card number from bottom corner of phone photos
   - Use OCR as hard constraint: if card number is readable, filter candidates
   - This alone could push accuracy above 95% for cards with readable numbers

7. **Collect More Real Phone Photos**
   - Scan more binder pages (target: 100+ real phone photos)
   - Use these for proper train/val/test splits
   - Re-evaluate all approaches with more data

### Phase D: Advanced (only if needed)

8. **Full CLIP Fine-Tuning** on larger phone photo dataset
9. **MoCo-style pretraining** on augmented card images
10. **Set-specific models** trained on cards within the same set (for hard negatives)

### Summary Table

| Approach | Trainable Params | Training Time | Expected Gain | Risk | Priority |
|---|---|---|---|---|---|
| Prototype Enrichment | 0 | 2-4h (encoding) | +3-8% | Very Low | 1 |
| Tip-Adapter | 0 | 0 | +5-10% | Very Low | 2 |
| CLIP-Adapter + SupCon | ~150K | 30-60 min GPU | +10-15% | Low | 3 |
| DINOv2 LoRA + Triplet | ~400K | 1-2h GPU | +10-20% | Medium | 4 |
| OCR Tiebreaker | 0 | N/A | +10-20% | Low | 5 |
| Full CLIP FT | ~300M | 4-8h GPU | +15-25% | High | 6 |
| SimCLR/MoCo FT | varies | 2-4h GPU | +10-20% | High | 7 |
| Meta-Learning (MAML) | varies | 4-8h GPU | unclear | Very High | Skip |

### Critical Path to >90%

The most likely path to >90% accuracy:

1. Prototype Enrichment + Tip-Adapter -> ~75-85% (up from ~60-80%)
2. Add CLIP-Adapter or DINOv2 LoRA -> ~85-92%
3. Add OCR tiebreaker -> ~92-97%

This matches the PokeScope architecture (CLIP + OCR) and should be achievable
within 1-2 weeks of development.

---

## 13. Sources

- [Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification (ECCV 2022)](https://arxiv.org/abs/2207.09519)
- [Tip-Adapter GitHub Implementation](https://github.com/gaopengcuhk/Tip-Adapter)
- [Proto-Adapter: Efficient Training-Free CLIP-Adapter](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175357/)
- [Fine-Tuning of CLIP in Few-Shot Scenarios via Supervised Contrastive Learning](https://link.springer.com/chapter/10.1007/978-981-97-8502-5_8)
- [Fully fine-tuned CLIP models are efficient few-shot learners](https://www.sciencedirect.com/science/article/abs/pii/S0950705125008652)
- [Adversarial domain adaptation with CLIP for few-shot classification](https://link.springer.com/article/10.1007/s10489-024-06088-4)
- [DINOv2 LoRA fine-tuning (dinov3-finetune)](https://github.com/RobvanGastel/dinov3-finetune)
- [DINOv2 for Image Classification: Fine-Tuning vs Transfer Learning](https://debuggercafe.com/dinov2-for-image-classification-fine-tuning-vs-transfer-learning/)
- [Parameter-Efficient Fine-Tuning of DINOv2 for Lung Nodule Classification](https://ieeexplore.ieee.org/document/10635887/)
- [Foundation vision models in agriculture: DINOv2, LoRA and knowledge distillation](https://www.sciencedirect.com/science/article/abs/pii/S0168169925010063)
- [PokeScope: How I Built a Pokemon Card Scanner App with AI](https://pokescope.app/blog/how-i-built-pokemon-card-scanner-ai-50000-users/)
- [Ximilar: Pokemon TCG Search Engine](https://www.ximilar.com/blog/pokemon-card-image-search-engine/)
- [PyTorch Metric Learning Library](https://kevinmusgrave.github.io/pytorch-metric-learning/)
- [PyTorch Metric Learning - Miners (Hard Negative Mining)](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/)
- [Prototypical Siamese Networks for Few-shot Learning](https://ieeexplore.ieee.org/document/9152261/)
- [Explaining Siamese networks in few-shot learning](https://link.springer.com/article/10.1007/s10994-024-06529-8)
- [SimCLR: A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709)
- [Self-supervised pre-training with contrastive methods for small datasets](https://www.nature.com/articles/s41598-023-46433-0)
- [CVPR 2025: Learning with Noisy Triplet Correspondence for Composed Image Retrieval](https://github.com/He-Changhao/2025-CVPR-TME)
- [Meta-Adapter (NeurIPS 2023)](https://github.com/ArsenalCheng/Meta-Adapter)
