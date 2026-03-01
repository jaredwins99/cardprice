# CLIP Fine-Tuning Research for Pokemon Card Matching

**Date**: 2026-02-28
**Context**: We currently use `openai/clip-vit-large-patch14` (base, no fine-tuning) for
card identification via image-to-text and image-to-image cosine similarity. This document
evaluates whether fine-tuning CLIP on Pokemon card images would improve phone-photo
matching accuracy.

---

## 1. Current Setup

- Model: `openai/clip-vit-large-patch14` (ViT-L/14, 428M params)
- Reference catalog: ~20k cards in `dim_cards`
- Two matching modes: image-to-text (card descriptions) and image-to-image (reference images)
- No domain adaptation -- using generic CLIP embeddings out of the box

## 2. Would Fine-Tuning Help?

**Yes, almost certainly.** Base CLIP is trained on generic internet image-text pairs. It has
no specific knowledge of:

- Pokemon card layouts (name bar, HP, artwork, attacks, set symbol, card number)
- Fine-grained differences between cards (same Pokemon, different set/rarity/variant)
- The domain shift between clean reference scans and noisy phone photos

Research on fine-grained visual recognition shows that contrastive learning models like CLIP
struggle with high intra-class similarity (same Pokemon across sets) and low inter-class
variance (different cards that look nearly identical). Domain-specific fine-tuning addresses
this by learning features relevant to the distinguishing details of Pokemon cards.

The key paper "Fully Fine-tuned CLIP Models are Efficient Few-Shot Learners" (2024) shows
that even modest fine-tuning significantly improves CLIP on domain-specific retrieval tasks.

## 3. Data Requirements

### How Much Data Do We Need?

| Approach | Data Required | Expected Improvement |
|---|---|---|
| Few-shot adapter (Tip-Adapter, Proto-Adapter) | 4-16 shots per class | Moderate; training-free |
| LoRA fine-tuning (vision encoder only) | 5k-50k image pairs | Strong |
| Full contrastive fine-tuning | 50k-500k pairs | Strongest, but risk of catastrophic forgetting |

**Our situation is favorable**: We have ~20k reference images (one per card). The bottleneck
is not reference images but *paired training data* -- we need pairs of (phone photo, reference
image) for the same card, or at minimum synthetic approximations of phone photos.

### Realistic Estimates

- **Minimum viable**: ~20k synthetic pairs (1 augmented version per reference image)
- **Recommended**: ~100k synthetic pairs (5 augmented versions per reference image)
- **Ideal**: ~100k synthetic + 1k-5k real phone photo pairs (for validation/calibration)

The few-shot literature shows that even 16 examples per class can improve CLIP significantly,
but for 20k classes (our card catalog), we need the model to learn *domain-level* features
(card layout awareness, robustness to phone photo artifacts), not per-class features. This
means we need enough augmented data to teach the model about the domain shift.

## 4. Synthetic Training Data Generation

**Yes, we can and should generate synthetic phone-photo-like training data.** This is a
well-established technique in computer vision. The augmentation pipeline would simulate
common phone photo conditions:

### Augmentation Pipeline (using torchvision / albumentations / imgaug)

```python
import albumentations as A

phone_photo_augmentation = A.Compose([
    # Geometric transforms (simulate hand-held angle)
    A.Perspective(scale=(0.02, 0.08), p=0.7),
    A.Rotate(limit=15, p=0.8),
    A.Affine(shear=(-5, 5), p=0.5),

    # Crop/pad (card not perfectly centered)
    A.RandomResizedCrop(224, 224, scale=(0.7, 1.0), ratio=(0.65, 0.80)),

    # Lighting conditions
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    A.RandomGamma(gamma_limit=(70, 130), p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05, p=0.6),

    # Phone camera artifacts
    A.GaussianBlur(blur_limit=(3, 7), p=0.4),
    A.MotionBlur(blur_limit=7, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.5),
    A.ISONoise(p=0.3),
    A.ImageCompression(quality_lower=60, quality_upper=95, p=0.5),

    # Glare simulation (bright spot overlay)
    A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1,
                     num_flare_circles_lower=1, num_flare_circles_upper=3,
                     src_radius=100, p=0.2),

    # Sleeve/surface reflections
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),
])
```

### Additional Synthetic Techniques

- **Background compositing**: Place card on random surfaces (tables, binders, playmats)
- **Sleeve overlay**: Semi-transparent layer with slight color tint
- **Partial occlusion**: Finger edges, other cards overlapping corners
- **Resolution downscaling**: Simulate distance/zoom levels

This approach is well-supported by research. The Ultralytics augmentation guide and
3D-aware blur synthesis papers confirm that synthetic augmentation significantly improves
model robustness to real-world conditions.

## 5. Training Cost and Time

### LoRA Fine-Tuning (Recommended Approach)

| Parameter | Value |
|---|---|
| Base model | `openai/clip-vit-large-patch14` (ViT-L/14) |
| LoRA rank | 16-64 (start with 16) |
| LoRA alpha | 2x rank (32-128) |
| Trainable params | ~2-8M (vs 428M total) |
| Training data | 50k-100k synthetic pairs |
| Batch size | 32-64 |
| Epochs | 5-20 |
| GPU requirement | Single GPU with 16+ GB VRAM |
| **Estimated time** | **2-6 hours on an A100, 4-12 hours on a consumer RTX 3090/4090** |
| **Cloud cost** | **$2-10 on Lambda/RunPod ($0.80-1.50/hr for A100)** |

### Full Fine-Tuning (Not Recommended Initially)

| Parameter | Value |
|---|---|
| Trainable params | 428M (all) |
| GPU requirement | 40+ GB VRAM (A100 40GB or multi-GPU) |
| Estimated time | 12-48 hours |
| Cloud cost | $15-75 |
| Risk | Catastrophic forgetting of general visual features |

### No-GPU Options

- **Tip-Adapter / Proto-Adapter**: Training-free, runs on CPU. Builds a key-value cache
  from few-shot examples. Could be a quick first experiment to validate the concept.
- **Linear probe**: Train only a linear layer on top of frozen CLIP features. Minutes on CPU.

## 6. LoRA vs Full Fine-Tuning

**LoRA is strongly recommended as the starting point.** Here is why:

### LoRA Advantages

1. **Parameter efficiency**: Only 0.5-2% of parameters are trainable, reducing memory
   from ~40GB to ~8-16GB (fits on consumer GPUs)
2. **No catastrophic forgetting**: Base CLIP weights are frozen, so general visual
   understanding is preserved. This is critical because we still want CLIP to understand
   "this is a card" vs "this is not a card"
3. **Composability**: LoRA weights are a small file (~10-30MB) that can be loaded on top
   of the base model. Easy to version, A/B test, and roll back
4. **Competitive performance**: Research consistently shows LoRA achieves 95-100% of
   full fine-tuning performance for domain adaptation tasks
5. **Fast iteration**: 2-6 hours per experiment vs 12-48 hours for full fine-tuning

### When Full Fine-Tuning Might Be Needed

- If LoRA plateaus and we need the last few percentage points of accuracy
- If the domain gap is extremely large (unlikely -- cards are still images of
  recognizable objects)
- If we want to significantly reduce the embedding dimensionality

### Recommended LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # Start with rank 16, increase to 64 if needed
    lora_alpha=32,           # 2x rank
    target_modules=[         # Apply to vision encoder attention layers
        "visual_projection",
        "vision_model.encoder.layers.*.self_attn.q_proj",
        "vision_model.encoder.layers.*.self_attn.v_proj",
    ],
    lora_dropout=0.1,
    bias="none",
)

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: ~2M trainable / 428M total (0.5%)
```

### Existing Tooling

- **[clipora](https://github.com/awilliamson10/clipora)**: Purpose-built toolkit for
  LoRA fine-tuning of OpenCLIP models
- **HuggingFace PEFT**: General-purpose LoRA library, works with CLIPModel
- **OpenCLIP**: Has built-in fine-tuning scripts with LoRA support

## 7. Existing Pokemon Card Models

No existing CLIP fine-tune for Pokemon card recognition was found. What exists:

| Model | Type | Task | Relevance |
|---|---|---|---|
| `hugginglearners/pokemon-card-checker` | ResNet34 (fastai) | Real vs fake card classification | Low -- binary classifier on card backs |
| `Matthieu68857/pokemon-cards-detection` | DETR (ResNet50) | Card detection/localization | Medium -- could help with card cropping |
| `ZeeshanGeoPk/pokemon-card-detection` | Object detection | Card detection | Medium -- same as above |
| `imjeffhi/pokemon_classifier` | ViT | Pokemon species classification | Low -- classifies species, not cards |
| `tooni/pokemoncards` | Dataset | Pokemon card images | High -- potential training data source |

**No one has published a CLIP fine-tune specifically for Pokemon card identification.**
This means we would be the first, which is both an opportunity (no competition) and a risk
(no prior art to validate the approach).

## 8. Recommended Implementation Plan

### Phase A: Baseline Measurement (1-2 days, no training)

1. Collect 50-100 real phone photos of known cards
2. Measure current top-1 and top-5 accuracy with base CLIP (image-to-image and image-to-text)
3. This gives us a concrete number to improve against

### Phase B: Synthetic Data Pipeline (2-3 days)

1. Build the augmentation pipeline (Section 4 above)
2. Generate 5 augmented versions per reference image = ~100k synthetic pairs
3. Format as `(augmented_image, reference_image, card_id)` triplets
4. Split 90/10 train/val

### Phase C: LoRA Training (1-2 days)

1. Fine-tune CLIP vision encoder with LoRA (rank 16)
2. Training objective: contrastive loss -- pull augmented photo embeddings toward their
   reference image embeddings, push away from other cards
3. Train for 10-20 epochs, monitor val loss
4. Export LoRA weights (~10-30MB file)

### Phase D: Evaluation (1 day)

1. Re-run the phone photo test set through the fine-tuned model
2. Compare top-1 and top-5 accuracy vs baseline
3. If top-1 improves by >10%, integrate into the pipeline
4. If marginal, try rank 64 or add more synthetic augmentations

### Phase E: Integration (1 day)

1. Modify `clip_matcher.py` to optionally load LoRA weights on top of base model
2. Rebuild image index with fine-tuned encoder
3. A/B test in the scan server

**Total estimated effort: 1-2 weeks**
**Total estimated cloud GPU cost: $5-20**

## 9. Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Synthetic data does not match real phone photos well enough | Collect 100+ real phone photos for validation; iterate on augmentation pipeline |
| LoRA overfits to augmentation artifacts | Use dropout (0.1), early stopping, diverse augmentations |
| Fine-tuning hurts performance on clean reference images | Keep base model as fallback; test both clean-to-clean and phone-to-clean retrieval |
| 20k classes is too many for contrastive learning | We are learning domain features, not per-class features; batch-level contrastive loss works fine |
| Training infrastructure (no local GPU) | Cloud GPU rental is cheap ($1-2/hr); entire training run costs under $20 |

## 10. Conclusion

Fine-tuning CLIP with LoRA for Pokemon card phone-photo matching is **strongly recommended**.
The approach is:

- **Feasible**: We have 20k reference images, synthetic augmentation is straightforward,
  and LoRA training fits on consumer/cloud GPUs for under $20
- **Low-risk**: LoRA preserves base model capabilities, and we can A/B test against the
  current pipeline
- **High expected impact**: Domain adaptation consistently improves retrieval accuracy
  by 10-30% in the literature, and our domain gap (phone photos vs clean scans) is
  exactly the kind of shift that fine-tuning addresses well

The recommended path is LoRA (rank 16) on the vision encoder only, trained on ~100k
synthetic phone-photo pairs generated from our existing reference images. No existing
Pokemon card CLIP model exists, so this would need to be trained from scratch on our data.

## Sources

- [A Guide to Fine-Tuning CLIP Models (Marqo)](https://www.marqo.ai/course/fine-tuning-clip-models)
- [LoRA Fine-tuning on CLIP from Scratch](https://medium.com/correll-lab/lora-fine-tuning-on-clip-from-scratch-83ff1a083bb5)
- [CLIP-LoRA: Efficient Adaptation](https://www.emergentmind.com/topics/clip-lora)
- [clipora - GitHub](https://github.com/awilliamson10/clipora)
- [Fully Fine-tuned CLIP Models are Efficient Few-Shot Learners](https://arxiv.org/html/2407.04003v1)
- [Fine-Tuning of CLIP in Few-Shot Scenarios via Supervised Contrastive Learning](https://link.springer.com/chapter/10.1007/978-981-97-8502-5_8)
- [Contrastive Learning for Fine-Grained Image Recognition](https://medium.com/@preeti.rana.ai/contrastive-learning-for-fine-grained-image-recognition-a-technical-deep-dive-46dea412926b)
- [PARTICLE: Part Discovery and Contrastive Learning for Fine-Grained Recognition (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023W/VIPriors/papers/Saha_PARTICLE_Part_Discovery_and_Contrastive_Learning_for_Fine-Grained_Recognition_ICCVW_2023_paper.pdf)
- [Data Augmentation: The Ultimate Guide (Ultralytics)](https://www.ultralytics.com/blog/the-ultimate-guide-to-data-augmentation-in-2025)
- [OpenCLIP Fine-Tuning Guide](https://github.com/mlfoundations/open_clip/discussions/812)
- [HuggingFace Pokemon Card Models](https://huggingface.co/hugginglearners/pokemon-card-checker)
- [tooni/pokemoncards Dataset](https://huggingface.co/datasets/tooni/pokemoncards)
