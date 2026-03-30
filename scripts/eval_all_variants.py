#!/usr/bin/env python3
"""Evaluate all stamp/variant detectors against ground truth cards.

Runs detect_stamps() on every ground truth card and reports per-detector
accuracy (TP/FP/FN/precision/recall) against known variant labels.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Ground truth variant labels
# ---------------------------------------------------------------------------
# Merge labels from binder_ground_truth.jsonl + ground_truth.json notable
# fields + user-provided known variants.
#
# Keys: "page/card" -> dict with expected stamps/variants.
# Stamp types map to stamp_detection.py detector names.

VARIANT_GROUND_TRUTH: dict[str, dict] = {
    # --- Delta Species page (ex15) ---
    # card_00 (Chikorita) + card_02 (Meganium) = stamped reverse holo
    "page_20260305_094228_cards/card_00": {
        "ex_set_stamp": True,
        "prerelease": False,
        "1st_edition": False,
    },
    "page_20260305_094228_cards/card_01": {
        "ex_set_stamp": False,
        "prerelease": False,
    },
    "page_20260305_094228_cards/card_02": {
        "ex_set_stamp": True,
        "prerelease": False,
    },
    "page_20260305_094228_cards/card_03": {
        "ex_set_stamp": False,
    },
    "page_20260305_094228_cards/card_04": {
        "ex_set_stamp": False,
    },
    # card_05 (Feraligatr) = holo (not stamped)
    "page_20260305_094228_cards/card_05": {
        "ex_set_stamp": False,
        "holo_finish": True,
    },
    "page_20260305_094228_cards/card_06": {
        "ex_set_stamp": False,
    },
    "page_20260305_094228_cards/card_07": {
        "ex_set_stamp": False,
    },
    # card_08 (Typhlosion) = holo (not stamped)
    "page_20260305_094228_cards/card_08": {
        "ex_set_stamp": False,
        "holo_finish": True,
    },

    # --- Misty page (gym1/gym2) ---
    "page_20260307_014406_cards/card_02": {
        "prerelease": True,
        "1st_edition": False,
    },
    # All others on Misty page: no 1st edition
    **{f"page_20260307_014406_cards/card_{i:02d}": {"1st_edition": False, "prerelease": False}
       for i in [0, 1, 3, 4, 5, 6, 7, 8]},

    # --- Mixed page (015320) ---
    # Dragonite (basep-5) = black star promo
    "page_20260307_015320_cards/card_05": {
        "prerelease": True,
        "black_star_promo": True,
    },
    # Dark Dragonite holo, not stamped
    "page_20260307_015320_cards/card_02": {
        "prerelease": False,
        "holo_finish": True,
    },

    # --- Fire page (020047) ---
    # Entei (basep-34) = black star promo
    "page_20260307_020047_cards/card_05": {
        "black_star_promo": True,
        "prerelease": False,
    },
    # Aerodactyl (base3-1) = prerelease stamp
    "page_20260307_020047_cards/card_08": {
        "prerelease": True,
        "1st_edition": False,
    },

    # --- Water page (120653) ---
    # Octillery (neo3-34) = 1st Edition
    "page_20260307_120653_cards/card_05": {
        "1st_edition": True,
        "prerelease": False,
    },
    # Seadra (neo1-48) = 1st Edition
    "page_20260307_120653_cards/card_07": {
        "1st_edition": True,
        "prerelease": False,
    },

    # --- Poochyena page (ex4) - none stamped ---
    **{f"page_20260320_223702_cards/card_{i:02d}": {"ex_set_stamp": False, "prerelease": False}
       for i in range(9)},

    # --- Absol page (mixed eras) ---
    # Absol ex13-18 = stamped (Holon Phantoms stamp)
    "page_20260320_191743_cards/card_00": {
        "ex_set_stamp": True,
    },

    # --- From binder_ground_truth.jsonl ---
    # Skitty (ex14) = stamped (Crystal Guardians)
    "page_20260228_174819_cards/card_01": {
        "ex_set_stamp": True,
    },
    # Vibrava (ex15) = stamped (Dragon Frontiers)
    "page_20260228_174819_cards/card_05": {
        "ex_set_stamp": True,
    },
}

# Overwrite Misty card_02 prerelease since we set it to False in the
# comprehension above — the explicit entry should win.
VARIANT_GROUND_TRUTH["page_20260307_014406_cards/card_02"] = {
    "prerelease": True,
    "1st_edition": False,
}


def load_ground_truth_cards() -> list[dict]:
    """Load all cards from ground_truth.json, returning flat list."""
    gt_path = Path(__file__).parent.parent / "data" / "ground_truth.json"
    with open(gt_path) as f:
        gt = json.load(f)

    inbox = Path(__file__).parent.parent / "data" / "inbox"
    cards = []
    for page_name, page_data in gt["pages"].items():
        for key in sorted(page_data.keys()):
            if not key.startswith("card_"):
                continue
            card = page_data[key]
            if card.get("empty_slot"):
                continue
            card_id = card.get("card_id")
            if not card_id:
                continue  # skip cards without ID (Japanese, etc.)

            img_path = inbox / page_name / f"{key}.png"
            if not img_path.exists():
                continue

            cards.append({
                "page": page_name,
                "slot": key,
                "card_id": card_id,
                "name": card.get("name", ""),
                "era": card.get("era", ""),
                "notable": card.get("notable", ""),
                "image_path": str(img_path),
                "gt_key": f"{page_name}/{key}",
            })
    return cards


def run_evaluation():
    """Run all stamp detectors on ground truth cards and report results."""
    from cardprice.ml.stamp_detection import detect_stamps

    cards = load_ground_truth_cards()
    print(f"Loaded {len(cards)} ground truth cards with card_id and images\n")

    # Track per-detector stats
    # detector_name -> {checked: int, tp: int, fp: int, fn: int, times: []}
    stats: dict[str, dict] = defaultdict(lambda: {
        "checked": 0, "tp": 0, "fp": 0, "fn": 0, "times": [],
        "fp_cards": [], "fn_cards": [], "tp_cards": [],
    })

    all_results = []

    for i, card in enumerate(cards):
        card_id = card["card_id"]
        img_path = card["image_path"]
        gt_key = card["gt_key"]

        t0 = time.perf_counter()
        result = detect_stamps(img_path, card_id, fast=False)
        elapsed = time.perf_counter() - t0

        stamps_detected = set(result.get("stamps_detected", []))
        stamps_checked = set(result.get("stamps_checked", []))

        all_results.append({
            "card": card,
            "result": result,
            "elapsed": elapsed,
        })

        # Update per-detector stats for every stamp type checked
        for stamp_type in stamps_checked:
            s = stats[stamp_type]
            s["checked"] += 1
            s["times"].append(elapsed / max(len(stamps_checked), 1))

            detected = stamp_type in stamps_detected
            gt_labels = VARIANT_GROUND_TRUTH.get(gt_key, {})

            if stamp_type in gt_labels:
                expected = gt_labels[stamp_type]
                if expected and detected:
                    s["tp"] += 1
                    s["tp_cards"].append(f"{card['name']} ({gt_key})")
                elif expected and not detected:
                    s["fn"] += 1
                    s["fn_cards"].append(f"{card['name']} ({gt_key})")
                elif not expected and detected:
                    s["fp"] += 1
                    conf = result.get("stamp_details", {}).get(stamp_type, {}).get("confidence", "?")
                    s["fp_cards"].append(f"{card['name']} ({gt_key}) conf={conf}")
                # else: TN (expected=False, detected=False) -- correct rejection

        # Progress
        if (i + 1) % 10 == 0 or i == len(cards) - 1:
            print(f"  [{i+1}/{len(cards)}] {card['name']:25s} "
                  f"stamps={list(stamps_detected) or 'none':40s} "
                  f"{elapsed:.2f}s")

    # --- Print summary report ---
    print("\n" + "=" * 80)
    print("=== Variant Detection Report ===")
    print("=" * 80)
    print(f"\nTotal cards evaluated: {len(cards)}")

    total_time = sum(r["elapsed"] for r in all_results)
    print(f"Total time: {total_time:.1f}s  ({total_time / len(cards):.2f}s/card avg)\n")

    # Sort detectors: those with ground truth first, then by check count
    def sort_key(item):
        name, s = item
        has_gt = s["tp"] + s["fp"] + s["fn"] > 0
        return (0 if has_gt else 1, -s["checked"], name)

    header = f"{'Detector':<25s} {'Checked':>7s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Prec':>8s} {'Recall':>8s} {'Avg ms':>8s}"
    print(header)
    print("-" * len(header))

    for name, s in sorted(stats.items(), key=sort_key):
        checked = s["checked"]
        tp = s["tp"]
        fp = s["fp"]
        fn = s["fn"]

        if tp + fp > 0:
            prec = f"{tp / (tp + fp):.3f}"
        else:
            prec = "-"

        if tp + fn > 0:
            recall = f"{tp / (tp + fn):.3f}"
        else:
            recall = "-"

        avg_ms = (sum(s["times"]) / len(s["times"]) * 1000) if s["times"] else 0

        print(f"{name:<25s} {checked:>7d} {tp:>4d} {fp:>4d} {fn:>4d} {prec:>8s} {recall:>8s} {avg_ms:>7.0f}ms")

    # --- Detail sections for FP/FN ---
    print("\n" + "=" * 80)
    print("=== False Positives (detected but should NOT be) ===")
    print("=" * 80)
    any_fp = False
    for name, s in sorted(stats.items()):
        if s["fp_cards"]:
            any_fp = True
            print(f"\n  {name}:")
            for c in s["fp_cards"]:
                print(f"    - {c}")
    if not any_fp:
        print("  (none)")

    print("\n" + "=" * 80)
    print("=== False Negatives (missed detections) ===")
    print("=" * 80)
    any_fn = False
    for name, s in sorted(stats.items()):
        if s["fn_cards"]:
            any_fn = True
            print(f"\n  {name}:")
            for c in s["fn_cards"]:
                print(f"    - {c}")
    if not any_fn:
        print("  (none)")

    print("\n" + "=" * 80)
    print("=== True Positives (correctly detected) ===")
    print("=" * 80)
    any_tp = False
    for name, s in sorted(stats.items()):
        if s["tp_cards"]:
            any_tp = True
            print(f"\n  {name}:")
            for c in s["tp_cards"]:
                print(f"    - {c}")
    if not any_tp:
        print("  (none)")

    # --- Cards with any stamps detected ---
    print("\n" + "=" * 80)
    print("=== All Detections ===")
    print("=" * 80)
    for r in all_results:
        stamps = r["result"].get("stamps_detected", [])
        if stamps:
            card = r["card"]
            details = r["result"].get("stamp_details", {})
            det_strs = []
            for st in stamps:
                conf = details.get(st, {}).get("confidence", "?")
                extra = ""
                if "holo_type" in details.get(st, {}):
                    extra = f" ({details[st]['holo_type']})"
                det_strs.append(f"{st}={conf}{extra}")
            print(f"  {card['name']:25s} {card['card_id']:25s} -> {', '.join(det_strs)}")


if __name__ == "__main__":
    run_evaluation()
