#!/usr/bin/env python3
"""Comprehensive variant detection evaluation on all ground truth cards.

Runs both detect_variant() (from variant_detector.py) and detect_stamps()
(from stamp_detection.py) on every card with known variant labels, drawn from:
  1. data/ground_truth.json  (12 pages, ~96 unique cards)
  2. data/condition_training/stamps_real/binder_ground_truth.jsonl  (16 entries)

Reports per-variant-type accuracy:
  - True positives (correctly detected)
  - False positives (incorrectly detected on cards without that variant)
  - False negatives (missed on cards that have the variant)
  - False positive rate
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INBOX = PROJECT_ROOT / "data" / "inbox"

# ── Variant categories we track ──────────────────────────────────────────
VARIANT_TYPES = [
    "1st_edition",
    "holofoil",
    "reverse_holofoil",  # includes EX-era stamped (same pricing bucket)
    "ex_set_stamp",      # EX-era stamp specifically
    "prerelease",
    "promo",
    "full_art",
    "gold",
    "rainbow_rare",
    "shadowless",
    "illustration_rare",
]


# ── Build ground truth ───────────────────────────────────────────────────

def _build_ground_truth() -> list[dict]:
    """Return list of dicts with image_path, card_id, name, gt_variants (set)."""
    cards = []

    # --- Source 1: ground_truth.json ---
    gt_path = PROJECT_ROOT / "data" / "ground_truth.json"
    with open(gt_path) as f:
        gt = json.load(f)

    # Skip duplicate scans -- use first scan only
    dup_info = gt.get("summary", {}).get("duplicate_scans", {})
    skip_pages = set()
    for group_pages in dup_info.values():
        for p in group_pages[1:]:
            skip_pages.add(p)

    for page_dir, page_data in gt["pages"].items():
        if page_dir in skip_pages:
            continue

        for slot_key in sorted(k for k in page_data if k.startswith("card_")):
            card = page_data[slot_key]
            if card.get("empty_slot"):
                continue

            card_id = card.get("card_id")
            name = card.get("name", "")
            if not card_id and not name:
                continue

            img_path = INBOX / page_dir / f"{slot_key}.png"
            if not img_path.exists():
                continue

            gt_variants = set()
            notable = (card.get("notable") or "").lower()
            card_id_str = card_id or ""

            # Holofoil from "notable" field
            if "holo" in notable and "reverse" not in notable:
                gt_variants.add("holofoil")

            # 1st Edition
            if "1st edition" in notable:
                gt_variants.add("1st_edition")

            # Prerelease
            if "prerelease" in notable:
                gt_variants.add("prerelease")

            # Promo (black star promo sets)
            set_id = card_id_str.split("/")[0].rsplit("-", 1)[0] if card_id_str else ""
            if set_id in ("basep", "np"):
                gt_variants.add("promo")

            # Gold card
            if card.get("gold_card"):
                gt_variants.add("gold")

            # Illustration rare
            if card.get("illustration_rare"):
                gt_variants.add("illustration_rare")

            # Full art
            if card.get("full_art"):
                gt_variants.add("full_art")

            cards.append({
                "image_path": str(img_path),
                "card_id": card_id,
                "name": name,
                "page": page_dir,
                "slot": slot_key,
                "gt_variants": gt_variants,
                "source": "ground_truth.json",
            })

    # --- Source 2: binder_ground_truth.jsonl ---
    jsonl_path = (PROJECT_ROOT / "data" / "condition_training"
                  / "stamps_real" / "binder_ground_truth.jsonl")
    if jsonl_path.exists():
        existing_by_path = {c["image_path"]: c for c in cards}

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                img_rel = entry["image"]
                img_path = str(INBOX / img_rel)

                variant = entry.get("variant", "normal")
                set_id = entry.get("set_id", "")

                new_labels = set()
                if variant == "stamped" and set_id.startswith("ex"):
                    new_labels.add("ex_set_stamp")
                    new_labels.add("reverse_holofoil")
                if variant == "prerelease":
                    new_labels.add("prerelease")
                if variant == "holofoil":
                    new_labels.add("holofoil")
                if "promo" in variant:
                    new_labels.add("promo")

                if img_path in existing_by_path:
                    existing_by_path[img_path]["gt_variants"] |= new_labels
                else:
                    # Card not in ground_truth.json -- add it
                    card_name = entry.get("card_name", "")
                    # Try to infer card_id from set_id
                    inferred_id = None
                    for c in cards:
                        if c["image_path"] == img_path:
                            inferred_id = c.get("card_id")
                            break
                    cards.append({
                        "image_path": img_path,
                        "card_id": inferred_id,
                        "name": card_name,
                        "page": img_rel.split("/")[0],
                        "slot": Path(img_rel).stem,
                        "gt_variants": new_labels,
                        "source": "binder_ground_truth.jsonl",
                        "set_id_override": set_id,
                    })

    return cards


# ── Run detectors ────────────────────────────────────────────────────────

def _run_detectors(cards: list[dict]) -> list[dict]:
    """Run variant_detector and stamp_detection on each card."""
    from cardprice.ml.variant_detector import detect_variant_detailed
    from cardprice.ml.stamp_detection import detect_stamps
    from cardprice.ml.era_detector import get_card_era

    results = []
    total = len(cards)

    for i, card in enumerate(cards):
        img_path = card["image_path"]
        card_id = card.get("card_id") or ""
        name = card.get("name", "")

        era = 0
        if card_id:
            try:
                era = get_card_era(card_id)
            except Exception:
                pass

        print(f"  [{i+1:3d}/{total}] {name:30s} ({card_id or 'no-id':30s}) ...",
              end="", flush=True)

        # Run detailed variant analysis (includes variant result)
        try:
            detail = detect_variant_detailed(img_path, era=era,
                                             card_id=card_id or None)
            variant_result = detail.get("variant", "normal")
        except Exception as e:
            detail = {"variant": f"ERROR:{e}"}
            variant_result = f"ERROR:{e}"

        # Run stamp_detection
        stamp_result = {"stamps_detected": [], "stamp_details": {},
                        "stamps_checked": []}
        if card_id:
            try:
                stamp_result = detect_stamps(img_path, card_id)
            except Exception as e:
                stamp_result["error"] = str(e)

        # Aggregate all detected variant labels
        detected = set()

        vr = variant_result if isinstance(variant_result, str) else ""
        if vr == "1st_edition":
            detected.add("1st_edition")
        elif vr == "holofoil":
            detected.add("holofoil")
        elif vr == "reverse_holofoil":
            detected.add("reverse_holofoil")
        elif vr == "promo":
            detected.add("promo")
        elif vr == "full_art":
            detected.add("full_art")
        elif vr == "gold":
            detected.add("gold")
        elif vr == "rainbow_rare":
            detected.add("rainbow_rare")
        elif vr == "shadowless":
            detected.add("shadowless")

        # From stamp_detection
        for stamp in stamp_result.get("stamps_detected", []):
            if stamp == "1st_edition":
                detected.add("1st_edition")
            elif stamp == "ex_set_stamp":
                detected.add("ex_set_stamp")
                detected.add("reverse_holofoil")
            elif stamp in ("black_star_promo", "modern_promo", "promo_stamp"):
                detected.add("promo")

        # Detailed analysis extras
        if isinstance(detail, dict):
            if detail.get("has_1st_edition_stamp"):
                detected.add("1st_edition")
            if detail.get("is_full_art"):
                detected.add("full_art")
            gr = detail.get("gold_rare_result")
            if gr == "gold":
                detected.add("gold")
            elif gr == "rainbow_rare":
                detected.add("rainbow_rare")

        # variant_detector returns reverse_holofoil for EX stamped cards
        set_prefix = ""
        if card_id and "-" in card_id:
            set_prefix = card_id.split("/")[0].rsplit("-", 1)[0]
        if vr == "reverse_holofoil" and set_prefix.startswith("ex"):
            detected.add("ex_set_stamp")

        print(f" detected={detected or '{normal}'}")

        results.append({
            **card,
            "era": era,
            "variant_detector_result": variant_result,
            "variant_detail": detail,
            "stamp_result": stamp_result,
            "detected_variants": detected,
        })

    return results


# ── Compute metrics ──────────────────────────────────────────────────────

def _compute_metrics(results: list[dict]) -> dict:
    """Compute TP, FP, FN, TN per variant type."""
    metrics = {}
    for vtype in VARIANT_TYPES:
        tp, fp, fn, tn = 0, 0, 0, 0
        tp_cards, fp_cards, fn_cards = [], [], []

        for r in results:
            has_label = vtype in r["gt_variants"]
            is_detected = vtype in r["detected_variants"]

            if has_label and is_detected:
                tp += 1; tp_cards.append(r)
            elif not has_label and is_detected:
                fp += 1; fp_cards.append(r)
            elif has_label and not is_detected:
                fn += 1; fn_cards.append(r)
            else:
                tn += 1

        total_pos = tp + fn
        total_neg = fp + tn
        metrics[vtype] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "total_positive": total_pos, "total_negative": total_neg,
            "tpr": tp / total_pos if total_pos > 0 else 0.0,
            "fpr": fp / total_neg if total_neg > 0 else 0.0,
            "tp_cards": tp_cards, "fp_cards": fp_cards, "fn_cards": fn_cards,
        }
    return metrics


# ── Print report ─────────────────────────────────────────────────────────

def _print_report(results: list[dict], metrics: dict):
    total = len(results)

    print("\n" + "=" * 100)
    print("VARIANT DETECTION EVALUATION REPORT")
    print(f"Total cards evaluated: {total}")
    print("=" * 100)

    # Ground truth distribution
    print("\nGround truth label distribution:")
    for vtype in VARIANT_TYPES:
        count = sum(1 for r in results if vtype in r["gt_variants"])
        if count > 0:
            names = [r["name"] for r in results if vtype in r["gt_variants"]]
            print(f"  {vtype:22s}: {count:2d} cards  ({', '.join(names)})")
    normal_count = sum(1 for r in results if not r["gt_variants"])
    print(f"  {'normal (no labels)':22s}: {normal_count:2d} cards")

    # Per-variant accuracy table
    print("\n" + "-" * 100)
    print(f"{'Variant Type':22s} {'TP':>6s} {'FN':>6s} {'FP':>6s}"
          f" {'TPR':>12s} {'FPR':>14s}")
    print("-" * 100)

    for vtype in VARIANT_TYPES:
        m = metrics[vtype]
        tp, fn, fp = m["tp"], m["fn"], m["fp"]
        total_pos = m["total_positive"]
        total_neg = m["total_negative"]

        if total_pos == 0 and fp == 0:
            tpr_s = "n/a"
            fpr_s = f"0/{total_neg}"
        elif total_pos == 0:
            tpr_s = "n/a"
            fpr_s = f"{fp}/{total_neg} ({m['fpr']:.0%})"
        else:
            tpr_s = f"{tp}/{total_pos} ({m['tpr']:.0%})"
            fpr_s = f"{fp}/{total_neg} ({m['fpr']:.0%})"

        print(f"{vtype:22s} {tp:6d} {fn:6d} {fp:6d}"
              f" {tpr_s:>12s} {fpr_s:>14s}")

    # Detailed per-variant results
    print("\n" + "=" * 100)
    print("DETAILED RESULTS BY VARIANT TYPE")
    print("=" * 100)

    for vtype in VARIANT_TYPES:
        m = metrics[vtype]
        if m["total_positive"] == 0 and m["fp"] == 0:
            continue

        print(f"\n--- {vtype} ---")

        if m["tp_cards"]:
            print(f"  TRUE POSITIVES ({m['tp']}):")
            for r in m["tp_cards"]:
                sd = r.get("stamp_result", {}).get("stamp_details", {})
                extra = ""
                for sk, sv in sd.items():
                    extra += f" stamp:{sk}(conf={sv.get('confidence',0):.2f})"
                print(f"    [OK]   {r['name']:28s} ({(r.get('card_id') or '?'):24s})"
                      f" {r['page']}/{r['slot']}{extra}")

        if m["fn_cards"]:
            print(f"  FALSE NEGATIVES ({m['fn']}) -- MISSED:")
            for r in m["fn_cards"]:
                vr = r.get("variant_detector_result", "?")
                stamps = r.get("stamp_result", {}).get("stamps_detected", [])
                print(f"    [MISS] {r['name']:28s} ({(r.get('card_id') or '?'):24s})"
                      f" {r['page']}/{r['slot']}"
                      f" variant_det={vr} stamps={stamps}")

        if m["fp_cards"]:
            print(f"  FALSE POSITIVES ({m['fp']}):")
            for r in m["fp_cards"]:
                vr = r.get("variant_detector_result", "?")
                stamps = r.get("stamp_result", {}).get("stamps_detected", [])
                sd = r.get("stamp_result", {}).get("stamp_details", {})
                extra = ""
                for sk, sv in sd.items():
                    extra += f" stamp:{sk}(conf={sv.get('confidence',0):.2f})"
                print(f"    [FP]   {r['name']:28s} ({(r.get('card_id') or '?'):24s})"
                      f" gt={r['gt_variants'] or 'normal'}"
                      f" variant_det={vr} stamps={stamps}{extra}")

    # Overall single-label accuracy
    print("\n" + "=" * 100)
    print("VARIANT DETECTOR SINGLE-LABEL ACCURACY")
    print("(Does variant_detector's primary return value match ground truth?)")
    print("=" * 100)

    correct = 0
    wrong = []
    for r in results:
        vr = r.get("variant_detector_result", "normal")
        gt = r["gt_variants"]

        if not gt and vr == "normal":
            correct += 1
        elif vr in gt:
            correct += 1
        elif vr == "reverse_holofoil" and ("ex_set_stamp" in gt
                                           or "reverse_holofoil" in gt):
            correct += 1
        else:
            wrong.append(r)

    print(f"\nCorrect: {correct}/{total} ({100*correct/total:.1f}%)")
    if wrong:
        print(f"\nIncorrect ({len(wrong)}):")
        for r in wrong:
            vr = r.get("variant_detector_result", "normal")
            print(f"  {r['name']:28s} ({(r.get('card_id') or '?'):24s})"
                  f" gt={r['gt_variants'] or 'normal'} detected={vr}")


def main():
    print("Building ground truth...")
    cards = _build_ground_truth()
    print(f"Found {len(cards)} card entries\n")

    # Filter to only cards with existing images
    cards = [c for c in cards if Path(c["image_path"]).exists()]
    print(f"Cards with existing images: {len(cards)}\n")

    print("Running detectors on all cards...")
    results = _run_detectors(cards)

    print("\nComputing metrics...")
    metrics = _compute_metrics(results)

    _print_report(results, metrics)


if __name__ == "__main__":
    main()
