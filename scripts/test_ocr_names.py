#!/usr/bin/env python3
"""Test EasyOCR name reading accuracy on eval card segments.

For each of the 27 cards in binder_eval.json:
1. Load the segment image
2. Crop to the top N% of the card (where the name is)
3. Run EasyOCR on that crop (raw color -- best for EasyOCR)
4. Compare the OCR result to ground truth name using fuzzy matching

Tests multiple crop heights (10%, 12%, 15%) to find the best one.
Saves results to data/eval/ocr_name_results.json.
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from rapidfuzz import fuzz

# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "binder_eval.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "eval" / "ocr_name_results.json"

CROP_PERCENTS = [0.10, 0.12, 0.15]
SIDE_MARGIN = 0.05  # trim 5% from each side to avoid binder sleeve edges


def load_eval_cards():
    """Load all cards from binder_eval.json with full image paths."""
    with open(EVAL_PATH) as f:
        data = json.load(f)

    cards = []
    for page in data["pages"]:
        seg_dir = PROJECT_ROOT / page["segments_dir"]
        for card in page["cards"]:
            # Skip empty slots
            if card.get("card_id") is None:
                continue
            cards.append({
                "image_path": str(seg_dir / card["segment"]),
                "card_id": card["card_id"],
                "name": card["name"],
                "segment": card["segment"],
                "page": page["image"],
            })
    return cards


def crop_name_region(image_path: str, top_pct: float) -> Image.Image:
    """Crop the top portion of a card image where the name is."""
    img = Image.open(image_path)
    w, h = img.size
    left = int(w * SIDE_MARGIN)
    right = int(w * (1.0 - SIDE_MARGIN))
    top = 0
    bottom = int(h * top_pct)
    return img.crop((left, top, right, bottom))


def preprocess_for_ocr(crop: Image.Image) -> Image.Image:
    """Preprocess crop for OCR: upscale, grayscale, contrast, sharpen, binarize."""
    w, h = crop.size
    if h < 100:
        scale = 300 / h
        crop = crop.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    elif h < 200:
        scale = 2.0
        crop = crop.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    gray = crop.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    gray = gray.filter(ImageFilter.SHARPEN)
    gray = gray.point(lambda p: 255 if p > 140 else 0)
    return gray


def clean_ocr_text(text: str) -> str:
    """Clean OCR text: remove common noise, keep alpha + spaces."""
    # Remove delta symbols and common OCR artifacts
    text = re.sub(r"[^A-Za-z\s\.\-']", "", text).strip()
    # Remove very short fragments that are likely noise
    return text


def run_easyocr_on_crop(reader, crop: Image.Image, use_raw: bool = True):
    """Run EasyOCR on a crop, return list of (text, confidence)."""
    if use_raw:
        img_array = np.array(crop)
    else:
        processed = preprocess_for_ocr(crop)
        img_array = np.array(processed)

    results = reader.readtext(img_array, detail=1)
    fragments = []
    for item in results:
        text = item[1].strip()
        conf = float(item[2])
        if text and len(text) >= 2:
            fragments.append((text, conf))
    return fragments


def best_name_from_fragments(fragments: list[tuple[str, float]]) -> str:
    """Pick the best card name from OCR fragments.

    Strategy: take the fragment with highest confidence that looks like a name
    (mostly alpha, length >= 3). If multiple, concatenate plausible name parts.
    """
    if not fragments:
        return ""

    # Filter to alpha-heavy fragments (card names are words)
    name_frags = []
    for text, conf in fragments:
        cleaned = clean_ocr_text(text)
        alpha_ratio = sum(1 for c in cleaned if c.isalpha()) / max(len(cleaned), 1)
        if alpha_ratio >= 0.6 and len(cleaned) >= 2:
            name_frags.append((cleaned, conf))

    if not name_frags:
        # Fallback: just use highest confidence fragment
        best = max(fragments, key=lambda x: x[1])
        return clean_ocr_text(best[0])

    # Sort by confidence descending
    name_frags.sort(key=lambda x: -x[1])

    # If top fragment is long enough, use it
    if len(name_frags[0][0]) >= 4:
        return name_frags[0][0]

    # Otherwise, combine top 2 fragments
    combined = " ".join(f[0] for f in name_frags[:2])
    return combined


def normalize_name(name: str) -> str:
    """Normalize a Pokemon name for comparison (lowercase, strip special chars)."""
    # Replace delta symbol variants
    name = name.replace("\u03b4", "delta").replace("δ", "delta")
    # Strip "ex" suffix for matching (it's often separate)
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


def compare_names(ocr_name: str, truth_name: str) -> dict:
    """Compare OCR name to ground truth using multiple fuzzy metrics."""
    ocr_norm = normalize_name(ocr_name)
    truth_norm = normalize_name(truth_name)

    # Also try without "ex" and "delta" suffixes
    truth_base = re.sub(r"\s*(ex|delta)\s*$", "", truth_norm).strip()
    ocr_base = re.sub(r"\s*(ex|delta)\s*$", "", ocr_norm).strip()

    ratio = fuzz.ratio(ocr_norm, truth_norm)
    partial = fuzz.partial_ratio(ocr_norm, truth_norm)
    token_set = fuzz.token_set_ratio(ocr_norm, truth_norm)

    # Also compare base names (without ex/delta)
    base_ratio = fuzz.ratio(ocr_base, truth_base)
    base_partial = fuzz.partial_ratio(ocr_base, truth_base)

    best_score = max(ratio, partial, token_set, base_ratio, base_partial)

    return {
        "ratio": ratio,
        "partial_ratio": partial,
        "token_set_ratio": token_set,
        "base_ratio": base_ratio,
        "base_partial": base_partial,
        "best_score": best_score,
        "match": best_score >= 70,
    }


def main():
    print("=" * 70)
    print("OCR Name Reading Test -- EasyOCR on Eval Card Segments")
    print("=" * 70)

    # Load eval cards
    cards = load_eval_cards()
    print(f"\nLoaded {len(cards)} eval cards (skipping empty slots)")

    # Initialize EasyOCR (slow first time -- downloads model)
    print("\nInitializing EasyOCR...")
    t0 = time.time()
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
    print(f"EasyOCR initialized in {time.time() - t0:.1f}s")

    # Results structure
    all_results = {
        "description": "OCR name reading accuracy test on binder eval cards",
        "crop_percents_tested": CROP_PERCENTS,
        "summary": {},
        "per_crop": {},
        "per_card": [],
    }

    # Test each crop percentage
    for crop_pct in CROP_PERCENTS:
        pct_key = f"top_{int(crop_pct * 100)}pct"
        print(f"\n{'='*60}")
        print(f"Testing crop: top {int(crop_pct * 100)}%")
        print(f"{'='*60}")

        crop_results = []
        matches = 0
        total = 0
        total_best_score = 0.0

        for card in cards:
            img_path = card["image_path"]
            truth_name = card["name"]

            # Crop
            crop = crop_name_region(img_path, crop_pct)

            # Run OCR on raw color (best for EasyOCR per project learnings)
            t1 = time.time()
            fragments_raw = run_easyocr_on_crop(reader, crop, use_raw=True)
            # Also try preprocessed
            fragments_proc = run_easyocr_on_crop(reader, crop, use_raw=False)
            ocr_time = time.time() - t1

            # Pick best name from each
            name_raw = best_name_from_fragments(fragments_raw)
            name_proc = best_name_from_fragments(fragments_proc)

            # Compare both to truth, pick better
            cmp_raw = compare_names(name_raw, truth_name)
            cmp_proc = compare_names(name_proc, truth_name)

            if cmp_raw["best_score"] >= cmp_proc["best_score"]:
                best_name = name_raw
                best_cmp = cmp_raw
                best_mode = "raw"
                all_frags = fragments_raw
            else:
                best_name = name_proc
                best_cmp = cmp_proc
                best_mode = "preprocessed"
                all_frags = fragments_proc

            is_match = best_cmp["match"]
            matches += int(is_match)
            total += 1
            total_best_score += best_cmp["best_score"]

            status = "OK" if is_match else "MISS"
            print(f"  [{status}] {truth_name:20s} -> OCR: {best_name:25s} "
                  f"(score={best_cmp['best_score']:.0f}, mode={best_mode}, "
                  f"time={ocr_time:.2f}s)")

            result_entry = {
                "segment": card["segment"],
                "page": card["page"],
                "card_id": card["card_id"],
                "truth_name": truth_name,
                "crop_pct": crop_pct,
                "ocr_name_raw": name_raw,
                "ocr_name_preprocessed": name_proc,
                "best_name": best_name,
                "best_mode": best_mode,
                "raw_fragments": [(t, round(c, 3)) for t, c in fragments_raw],
                "proc_fragments": [(t, round(c, 3)) for t, c in fragments_proc],
                "scores": best_cmp,
                "ocr_time_s": round(ocr_time, 3),
            }
            crop_results.append(result_entry)

            # Also store in per_card with crop_pct tagged
            if crop_pct == CROP_PERCENTS[0]:
                # Initialize per-card entry on first crop
                all_results["per_card"].append({
                    "segment": card["segment"],
                    "page": card["page"],
                    "card_id": card["card_id"],
                    "truth_name": truth_name,
                    "crops": {},
                })
            # Find the per_card entry
            for pc in all_results["per_card"]:
                if pc["segment"] == card["segment"] and pc["page"] == card["page"]:
                    pc["crops"][pct_key] = {
                        "ocr_name": best_name,
                        "mode": best_mode,
                        "best_score": best_cmp["best_score"],
                        "match": is_match,
                        "raw_fragments": [(t, round(c, 3)) for t, c in fragments_raw],
                    }
                    break

        accuracy = matches / total * 100 if total else 0
        avg_score = total_best_score / total if total else 0

        all_results["per_crop"][pct_key] = {
            "crop_percent": crop_pct,
            "accuracy": round(accuracy, 1),
            "matches": matches,
            "total": total,
            "avg_best_score": round(avg_score, 1),
            "details": crop_results,
        }

        print(f"\n  Accuracy: {matches}/{total} ({accuracy:.1f}%)")
        print(f"  Avg best score: {avg_score:.1f}")

    # Summary: find best crop
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    best_crop = None
    best_acc = -1
    for pct_key, data in all_results["per_crop"].items():
        acc = data["accuracy"]
        avg = data["avg_best_score"]
        print(f"  {pct_key}: {data['matches']}/{data['total']} "
              f"({acc:.1f}%) avg_score={avg:.1f}")
        if acc > best_acc or (acc == best_acc and avg > all_results["per_crop"].get(best_crop, {}).get("avg_best_score", 0)):
            best_acc = acc
            best_crop = pct_key

    all_results["summary"] = {
        "best_crop": best_crop,
        "best_accuracy": best_acc,
        "total_cards": len(cards),
    }
    print(f"\n  Best crop region: {best_crop} ({best_acc:.1f}% accuracy)")

    # Show misses for the best crop
    best_data = all_results["per_crop"][best_crop]
    misses = [d for d in best_data["details"] if not d["scores"]["match"]]
    if misses:
        print(f"\n  Misses ({len(misses)}):")
        for m in misses:
            print(f"    {m['truth_name']:20s} -> {m['best_name']:25s} "
                  f"(score={m['scores']['best_score']:.0f})")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
