# ML Card Recognition Pipeline

## Cascade Architecture

The cascade (`cardprice.ml.identify_card`) tries three identification tiers in order of cost and speed. It stops at the first tier that returns a confident match.

```
Image --> [Tier 1: Hash] --miss--> [Tier 2: DINOv2] --miss--> [Tier 3: Claude] --miss--> None
              |                         |                           |
           distance < 5            similarity > 0.95         confidence > 0.5
              |                         |                           |
           return                    return                      return
```

The cascade requires `ANTHROPIC_API_KEY` in the environment for Tier 3. Tiers 1 and 2 are skipped silently if their index files are missing.

### Return format

All tiers return the same dict:

```python
{"card_id": str | None, "confidence": float, "method": str, "raw_response": dict}
```

`method` is one of `"hash"`, `"dino"`, `"claude"`, or `None` if nothing matched.

---

## Tier 1: Perceptual Hash (`hash_matcher.py`)

**How it works:** Computes pHash (perceptual hash) of the query image and compares it against a prebuilt database using Hamming distance. Also stores dHash, aHash, and wHash for potential future multi-hash voting.

**Strengths:**
- Instant (sub-millisecond per query after DB load)
- Zero cost, no GPU needed
- Excellent for matching near-identical images (same scan, same source)

**Weaknesses:**
- Brittle to cropping, rotation, glare, or camera angles
- Useless for photos of physical cards (only works for digital scans/screenshots)
- No semantic understanding

**Thresholds (Hamming distance on pHash):**

| Distance | Label       | Cascade action |
|----------|-------------|----------------|
| < 5      | confident   | Accept         |
| 5-9      | likely      | Fall through   |
| 10-14    | possible    | Fall through   |
| >= 15    | no_match    | Fall through   |

**Confidence conversion in cascade:** `max(0, 1.0 - distance / 15.0)`

**Index file:** `data/hash_db.pkl` (pickled dict mapping card_id to hash objects)

---

## Tier 2: DINOv2 + FAISS (`dino_matcher.py`)

**How it works:** Extracts 768-dim CLS embeddings from DINOv2 ViT-B/14, L2-normalizes them, and stores them in a FAISS `IndexFlatIP` index. At query time, computes cosine similarity (inner product on normalized vectors) against the full index.

**Strengths:**
- Strong visual understanding -- handles different angles, lighting, minor wear
- No API cost (runs locally)
- FAISS makes search fast even at scale

**Weaknesses:**
- Requires GPU for reasonable index build time (~1s/query on CPU)
- First model load is slow (~5s)
- Cannot distinguish variants that look identical (e.g., holofoil vs reverse holofoil from a flat scan)

**Thresholds (cosine similarity):**

| Similarity | Action in cascade | Action in MatchPipeline |
|------------|-------------------|------------------------|
| >= 0.95    | Accept            | Accept                 |
| 0.85-0.95  | Fall through      | OCR verification       |
| < 0.85     | Fall through      | Manual review          |

The `MatchPipeline` class adds an OCR fallback layer (via pytesseract) for the 0.85-0.95 range, verifying the DINOv2 match by checking if OCR text from the card matches tokens in the card_id.

**Index files:**
- `data/dino_index.faiss` -- FAISS index (N x 768 float32 vectors)
- `data/dino_card_ids.pkl` -- pickled list of card_id strings, positionally aligned with the index

---

## Tier 3: Claude Vision (`claude_scanner.py`)

**How it works:** Sends the card image to Claude Haiku (claude-haiku-4-5) with a structured prompt. Claude extracts card name, set, number, rarity, condition, grading info, and language. The extracted fields are then matched against `dim_cards` in the database using progressively looser SQL queries.

**Strengths:**
- Highest accuracy -- understands text, symbols, set logos, and context
- Can identify condition (NM/LP/MP/HP/DMG) and grading (PSA/BGS/CGC)
- Works on any photo quality
- Can identify cards not in the image index

**Weaknesses:**
- Costs ~$0.0015 per card (Haiku pricing)
- Slowest tier (~1-2s per card, rate-limited to ~1.4 req/s in batch mode)
- Requires `ANTHROPIC_API_KEY`
- DB matching confidence depends on how well Claude reads the card:
  - 0.95 = name + set + number all match
  - 0.80 = name + number match (set might be wrong)
  - 0.70 = name + set match (number might be misread)
  - 0.50 = name only match

**Cascade acceptance threshold:** confidence > 0.5 (accepts even name-only matches)

**No index needed.** This tier queries the database directly.

---

## Card ID Derivation from Image Paths

Both hash_matcher and dino_matcher/clip_matcher derive `card_id` from filenames, but use slightly different conventions:

**hash_matcher:** Uses the filename stem directly.
```
base1-4_holofoil.png  -->  card_id = "base1-4_holofoil"
```

**dino_matcher and clip_matcher:** Uses relative path from image_dir, replaces the last underscore with `/` to reconstruct the variant separator.
```
data/card_images/sv8/sv8-162_normal.png
  relative to image_dir:  sv8/sv8-162_normal
  last '_' -> '/':        sv8/sv8-162/normal
```

This means image files must follow the naming convention `{set_prefix}/{pokemontcg_id}_{variant}.{ext}` for DINOv2 and CLIP indexes.

---

## Building Indexes (CLI Commands)

All commands assume card images are in `data/card_images/`.

### Perceptual hash database

```bash
python -m cardprice.cli build-hash-index
# Options:
#   --image-dir data/card_images   (default)
#   --output data/hash_db.pkl      (default)
```

### DINOv2 FAISS index

```bash
python -m cardprice.cli build-dino-index
# Options:
#   --image-dir data/card_images          (default)
#   --index-path data/dino_index.faiss    (default)
#   --mapping-path data/dino_card_ids.pkl (default)
```

### CLIP indexes

```bash
# Text index (from database card descriptions, no images needed):
python -m cardprice.cli build-clip-index --mode text

# Image index (from card image directory):
python -m cardprice.cli build-clip-index --mode image --image-dir data/card_images

# Both:
python -m cardprice.cli build-clip-index --mode both --image-dir data/card_images
```

**Build order recommendation:** Build hash first (fastest, seconds), then DINOv2 (minutes, needs GPU ideally), then CLIP text (minutes, needs DB), then CLIP image (minutes).

---

## When to Use Which Model Directly

| Scenario | Use | Why |
|----------|-----|-----|
| Batch scanning screenshots/digital images | `hash` | Instant, free, perfect for identical-source images |
| Scanning physical cards from phone photos | `dino` or `cascade` | DINOv2 handles camera variation well |
| Single high-value card identification | `claude-haiku-4-5` | Most accurate, worth the $0.0015 |
| Text-based search ("find Charizard from Base Set") | `clip` text index | CLIP bridges text and image modalities |
| Production scanning flow | `cascade` | Best cost/accuracy tradeoff -- free tiers handle easy cases, Claude handles the rest |
| Inventory scanning (need condition + grading) | `claude-haiku-4-5` directly | Only Claude extracts condition/grade info |

### CLI scan command

```bash
# Single image with specific model:
python -m cardprice.cli scan --model hash card_photo.jpg
python -m cardprice.cli scan --model dino card_photo.jpg
python -m cardprice.cli scan --model clip card_photo.jpg
python -m cardprice.cli scan --model claude-haiku-4-5 card_photo.jpg

# Full cascade:
python -m cardprice.cli scan --model cascade card_photo.jpg

# Batch scan a directory:
python -m cardprice.cli scan --dir photos/ --model cascade
```

---

## Confidence Scoring Summary

| Tier | Metric | Range | "Confident" threshold |
|------|--------|-------|-----------------------|
| Hash | 1 - (hamming_distance / 15) | 0.0 - 1.0 | > 0.67 (distance < 5) |
| DINOv2 | Cosine similarity | 0.0 - 1.0 | > 0.95 |
| CLIP | Cosine similarity | 0.0 - 1.0 | Not in cascade |
| Claude | DB match quality | 0.50 - 0.95 | > 0.50 (in cascade) |

Note that CLIP is not part of the cascade. It is available as a standalone model for direct use via the CLI but is not wired into `identify_card()`.
