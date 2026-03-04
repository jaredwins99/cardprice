# CLIP Fine-Tuning Documentation Index

This directory contains comprehensive research and implementation guides for fine-tuning CLIP to improve Pokemon card identification accuracy on phone photos.

## Problem Statement

Current CLIP performance: **67% top-1 accuracy** on e-Card era binder page photos, much worse on mixed-era or holofoil pages. The main issue is the domain gap between clean reference images and phone photos taken through plastic sleeves with potential glare.

**Goal**: Achieve **>85% top-1 accuracy** through fine-tuning the CLIP vision encoder on synthetic phone-photo-like augmentations.

---

## Documentation Structure

### 1. **[clip-finetuning-pokemon.md](clip-finetuning-pokemon.md)** — COMPREHENSIVE GUIDE

**937 lines, 35KB** — Full technical reference for researchers and implementers.

**Covers**:
- Executive summary with feasibility assessment
- Current baseline analysis and available data inventory
- Why LoRA is recommended over full fine-tuning
- Detailed LoRA configuration for vision encoder only
- Loss function comparison (InfoNCE vs Triplet vs Classification)
- Training data strategy with synthetic augmentation pipeline
- Training loop implementation with hyperparameters tuned for RTX 4070 SUPER
- Expected timing: 4-8 hours GPU training
- Integration points with existing cascade pipeline
- A/B testing and evaluation methodology
- 6-phase implementation roadmap (A-F)
- Risk analysis and mitigation strategies
- Comparison of LoRA vs full fine-tuning vs Tip-Adapter
- Academic references and further reading

**Audience**: ML engineers, researchers, anyone building the solution

### 2. **[clip-finetuning-quickstart.md](clip-finetuning-quickstart.md)** — PRACTICAL PLAYBOOK

**~300 lines** — Quick reference guide with decision tree and code snippets.

**Contains**:
- 5-minute decision tree (do I need this? which option?)
- Key numbers at a glance (VRAM, time, cost, expected improvement)
- 3 implementation options with code:
  1. Augmented Index (free, 4-6 hours, +8-12% improvement)
  2. LoRA Training (4-8 GPU hours, $0-15, +15-25% improvement) ← RECOMMENDED
  3. Tip-Adapter (training-free, +5-10% improvement)
- Step-by-step quickstart for LoRA training
- Troubleshooting common issues
- Next steps checklist

**Audience**: Practitioners who want to get started quickly

### 3. **[clip-finetuning-research.md](clip-finetuning-research.md)** — EARLIER RESEARCH

Prior research document with foundational analysis.

---

## Key Findings Summary

### Will Fine-Tuning Help?

**Yes, significantly.** Base CLIP is trained on generic internet images and has no knowledge of:
- Pokemon card layouts (name bar, HP, artwork, attacks, set symbol)
- Fine-grained card differences (same Pokemon, different set/rarity/variant)
- Domain shift from clean scans to phone photos with glare/perspective/compression

Research shows contrastive learning improves domain-shift tasks by **10-30%**. Our estimated improvement: **+15-25% top-1 accuracy** (67% → 82-92%).

### Why LoRA (Not Full Fine-Tuning)?

| Factor | LoRA | Full FT |
|--------|------|---------|
| VRAM needed | 12GB ✓ | 40GB+ ✗ |
| Training time | 4-8 hours | 24+ hours |
| Trainable params | 0.5-2% | 100% |
| Catastrophic forgetting risk | Low | High |
| Composability | Yes (~20MB) | Large checkpoint |
| Expected final perf | 95-100% of full | 100% (overkill) |

### Data Requirements

- **Minimum viable**: 20k synthetic pairs (1 augmented version per reference card)
- **Recommended**: 100k synthetic pairs (5 augmented per reference)
- **Ideal**: 100k synthetic + 1k-5k real phone photo pairs for validation

We have 20,026 reference images and 26 real phone segments. Augmentation pipeline is already implemented in codebase.

### Loss Function: InfoNCE

Use **InfoNCE (Normalized Temperature-scaled Cross Entropy)** — the same loss CLIP uses:
- Better than triplet loss because it uses all in-batch negatives simultaneously
- Requires large batch sizes (32-64) for good performance
- Temperature = 0.07 (CLIP standard)

### Expected Training Time (RTX 4070 SUPER)

| Dataset | Batch Size | Epochs | Time |
|---------|-----------|--------|------|
| 100k pairs (5 aug) | 32 | 15 | **4-6 hours** |
| 200k pairs (10 aug) | 32 | 15 | **8-12 hours** |

Fits in 12GB VRAM with room to spare (~11GB usage).

### Integration Points

Existing codebase supports:
- `clipmatcher.py`: Already has `generate_augmented_views()` and `build_augmented_image_index()`
- `__init__.py`: Cascade already loads CLIP indexes in correct priority order
- Augmented index support: Already built into preference logic

Just need to:
1. Train LoRA weights (new)
2. Build index with fine-tuned encoder (new)
3. Update `__init__.py` to prefer LoRA index (1-line change)

---

## Quick Decision Tree

```
┌─ Do you have good phone photo test data?
│  └─ NO → Collect 50+ real photos (1-2 days)
│
└─ YES → What's current top-1 accuracy?
    ├─ Already 85%+? → Ship it; optional: try LoRA for marginal gain
    │
    └─ 67-80% → Try quickest solution first:
        │
        ├─ Option A: Augmented Index (free, 4-6 hrs, +8-12%)
        │   └─ python scripts/test_augmented_clip.py --build
        │
        └─ If A gives <80%, try Option B:
            └─ Option B: LoRA Training (4-8 GPU hrs, $0-15, +15-25%)
                └─ python scripts/train_clip_lora.py
```

---

## Implementation Roadmap (4-5 days active work)

| Phase | Task | Time | Deliverable |
|-------|------|------|-------------|
| **A** | Setup & measure baseline | 1 day | Baseline accuracy numbers |
| **B** | Generate augmented dataset | 1 day | 100k training pairs ready |
| **C** | Train LoRA weights | 1-2 days (4-8 GPU hrs) | LoRA checkpoint |
| **D** | Build fine-tuned index | 0.5 days | 20k card index with LoRA encoder |
| **E** | Evaluate and compare | 1 day | Accuracy report, go/no-go decision |
| **F** | Production integration | 0.5 days | Deployed in cascade pipeline |

---

## Resources

### In This Repository

- **Codebase**: `/home/godli/cardprice/cardprice/ml/clip_matcher.py` — Existing CLIP infrastructure
- **Augmentation**: Already implemented in `generate_augmented_views()`
- **Test harness**: `scripts/test_augmented_clip.py` — Compare indexes
- **Evaluation**: `scripts/eval_cascade.py` — Measure accuracy

### Academic Papers

- [Low-Rank Few-Shot Adaptation of Vision-Language Models (CLIP-LoRA)](https://openaccess.thecvf.com/content/CVPR2024W/PV/papers/Zanella_Low-Rank_Few-Shot_Adaptation_of_Vision-Language_Models_CVPRW_2024_paper.pdf) — CVPRW 2024
- [Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950487.pdf) — ECCV 2022
- [Domain Gap Embeddings for Generative Dataset Augmentation](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Domain_Gap_Embeddings_for_Generative_Dataset_Augmentation_CVPR_2024_paper.pdf) — CVPR 2024

### Implementation Libraries

- [HuggingFace PEFT](https://github.com/huggingface/peft) — LoRA implementation
- [CLIP-LoRA GitHub](https://github.com/MaxZanella/CLIP-LoRA) — Reference implementation
- [Tip-Adapter GitHub](https://github.com/gaopengcuhk/Tip-Adapter) — Training-free adapter

---

## FAQ

**Q: Do I need a GPU to train?**
A: Yes. LoRA training requires GPU; estimated 4-8 hours on RTX 4070 SUPER. Can rent cloud GPU for $5-15.

**Q: What if augmented index already gives 80%+ accuracy?**
A: Great! Ship it. LoRA training is optional in that case.

**Q: Can I use Tip-Adapter instead of LoRA?**
A: Yes, but limited to the 26 real examples you have. LoRA generalizes much better to unseen cards.

**Q: How much improvement should I expect?**
A: Literature shows +10-30% for domain-shift tasks. We estimate +15-25% absolute (67% → 82-92%).

**Q: Will fine-tuning hurt performance on clean reference images?**
A: LoRA risk is low; base weights frozen. Validate on clean index to be sure.

**Q: Can I use full fine-tuning instead of LoRA?**
A: Not recommended. LoRA is safer, faster, and achieves 95-100% of full fine-tuning performance.

**Q: How do I decide between options A (augmented) vs B (LoRA)?**
A: Try A first (free, 4-6 hours). If top-1 < 80%, do B. See decision tree above.

---

## Support & Next Steps

1. **Read the quickstart** (`clip-finetuning-quickstart.md`) — 10 minutes
2. **Follow the decision tree** — 5 minutes
3. **Measure baseline** — 1 hour (`scripts/eval_cascade.py`)
4. **Try augmented index** — 4-6 hours (free)
5. **If needed, train LoRA** — 4-8 GPU hours

Any questions? See the full guide: `clip-finetuning-pokemon.md`

---

**Status**: Ready to implement
**Last Updated**: 2026-02-28
**Author**: Claude Code Research
