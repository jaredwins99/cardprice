#!/usr/bin/env python3
"""Compare both centering modules on binder page segments.

Runs BOTH centering approaches on 9 segments (one page) from binder_eval.json:
  1. centering.py         -- gradient-based inner frame detection
  2. centering_detector.py -- HSV border color detection

Reports side-by-side: L/R ratio, T/B ratio, PSA grade, confidence, and
notes on which module gives more reliable results for binder segments.

Usage:
    python scripts/test_centering_eval.py [--page N]  (default: page 0)
"""

import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cardprice.ml.centering import measure_centering as gradient_centering
from cardprice.ml.centering_detector import measure_centering as hsv_centering


def format_gradient_result(result):
    """Extract comparable fields from centering.py CenteringResult."""
    if not result.success:
        return {
            "lr": "FAIL",
            "tb": "FAIL",
            "psa": "FAIL",
            "borders": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "error": result.error,
        }
    lr_a, lr_b = result.lr_ratio
    tb_a, tb_b = result.tb_ratio
    # Format as bigger/smaller like centering_detector does
    lr_str = f"{max(lr_a, lr_b):.0f}/{min(lr_a, lr_b):.0f}"
    tb_str = f"{max(tb_a, tb_b):.0f}/{min(tb_a, tb_b):.0f}"
    psa = f"{result.psa_grade} ({result.psa_label})" if result.psa_grade else "below 7"
    return {
        "lr": lr_str,
        "tb": tb_str,
        "psa": psa,
        "lr_pct": result.lr_pct,
        "tb_pct": result.tb_pct,
        "borders": {
            "left": result.left.pixels,
            "right": result.right.pixels,
            "top": result.top.pixels,
            "bottom": result.bottom.pixels,
        },
        "error": "",
    }


def format_hsv_result(result):
    """Extract comparable fields from centering_detector.py dict result."""
    borders = result["borders"]
    return {
        "lr": result["front_lr"],
        "tb": result["front_tb"],
        "psa": f"{result['centering_score']:.1f}",
        "confidence": result["confidence"],
        "borders": borders,
        "error": "",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", type=int, default=0,
                        help="Page index from binder_eval.json (0-2)")
    args = parser.parse_args()

    eval_path = ROOT / "data" / "eval" / "binder_eval.json"
    with open(eval_path) as f:
        eval_data = json.load(f)

    if args.page >= len(eval_data["pages"]):
        print(f"Only {len(eval_data['pages'])} pages available (0-{len(eval_data['pages'])-1})")
        sys.exit(1)

    page = eval_data["pages"][args.page]
    seg_dir = ROOT / page["segments_dir"]

    print(f"Page {args.page}: {page['segments_dir']}")
    print(f"Segment resolution: 1008x1530 (binder page segments)")
    print()

    # Header
    hdr = (f"{'Pos':<6} {'Name':<20} "
           f"{'|':>1} {'Gradient L/R':<13} {'Gradient T/B':<13} {'Grad PSA':<14} {'Borders (L/R/T/B)':<28} "
           f"{'|':>1} {'HSV L/R':<10} {'HSV T/B':<10} {'HSV Score':<10} {'HSV Conf':<9} {'Borders (L/R/T/B)':<28} "
           f"{'|':>1} {'Notes'}")
    print(hdr)
    print("-" * len(hdr))

    gradient_results = []
    hsv_results = []
    issues = []

    for card in page["cards"]:
        seg_path = seg_dir / card["segment"]
        name = card["name"][:18]
        pos = f"[{card['position'][0]},{card['position'][1]}]"

        # --- Run gradient-based centering (centering.py) ---
        try:
            g_raw = gradient_centering(str(seg_path))
            g = format_gradient_result(g_raw)
        except Exception as e:
            g = {"lr": "ERR", "tb": "ERR", "psa": "ERR",
                 "borders": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                 "error": str(e)}

        # --- Run HSV-based centering (centering_detector.py) ---
        try:
            h_raw = hsv_centering(str(seg_path))
            h = format_hsv_result(h_raw)
        except Exception as e:
            h = {"lr": "ERR", "tb": "ERR", "psa": "ERR", "confidence": 0,
                 "borders": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                 "error": str(e)}

        # Format border strings
        gb = g["borders"]
        g_bstr = f"L={gb['left']:>5.1f} R={gb['right']:>5.1f} T={gb['top']:>5.1f} B={gb['bottom']:>5.1f}"
        hb = h["borders"]
        h_bstr = f"L={hb['left']:>3} R={hb['right']:>3} T={hb['top']:>3} B={hb['bottom']:>3}"

        # Notes on potential issues
        notes = []
        if g.get("error"):
            notes.append(f"GRAD_ERR:{g['error'][:30]}")
        if h.get("error"):
            notes.append(f"HSV_ERR:{h['error'][:30]}")

        # Check for disagreement
        if g["lr"] != "FAIL" and g["lr"] != "ERR" and h["lr"] != "ERR":
            # Compare: are they in the same ballpark?
            g_lr_parts = g["lr"].split("/")
            h_lr_parts = h["lr"].split("/")
            try:
                g_bigger_lr = int(g_lr_parts[0])
                h_bigger_lr = int(h_lr_parts[0])
                if abs(g_bigger_lr - h_bigger_lr) > 10:
                    notes.append(f"LR_DISAGREE(diff={abs(g_bigger_lr - h_bigger_lr)})")
                g_tb_parts = g["tb"].split("/")
                h_tb_parts = h["tb"].split("/")
                g_bigger_tb = int(g_tb_parts[0])
                h_bigger_tb = int(h_tb_parts[0])
                if abs(g_bigger_tb - h_bigger_tb) > 10:
                    notes.append(f"TB_DISAGREE(diff={abs(g_bigger_tb - h_bigger_tb)})")
            except (ValueError, IndexError):
                pass

        # Check for thin borders (segmenter artifact concern)
        min_g_border = min(gb["left"], gb["right"], gb["top"], gb["bottom"])
        min_h_border = min(hb["left"], hb["right"], hb["top"], hb["bottom"])
        if min_g_border < 10:
            notes.append("GRAD_THIN_BORDER")
        if min_h_border < 5:
            notes.append("HSV_THIN_BORDER")

        # HSV confidence flag
        conf = h.get("confidence", 0)
        if conf <= 0.5:
            notes.append(f"HSV_LOW_CONF({conf:.2f})")

        note_str = "; ".join(notes) if notes else "ok"

        print(f"{pos:<6} {name:<20} "
              f"| {g['lr']:<13} {g['tb']:<13} {g['psa']:<14} {g_bstr:<28} "
              f"| {h['lr']:<10} {h['tb']:<10} {h['psa']:<10} {conf:<9.2f} {h_bstr:<28} "
              f"| {note_str}")

        gradient_results.append({"name": card["name"], "pos": pos, **g})
        hsv_results.append({"name": card["name"], "pos": pos, **h})
        if notes and notes != ["ok"]:
            issues.append({"name": card["name"], "pos": pos, "notes": notes})

    # --- Summary ---
    print()
    print("=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)

    # Agreement analysis
    agree_lr = 0
    agree_tb = 0
    total = 0
    for g, h in zip(gradient_results, hsv_results):
        if g["lr"] in ("FAIL", "ERR") or h["lr"] == "ERR":
            continue
        total += 1
        try:
            g_lr = int(g["lr"].split("/")[0])
            h_lr = int(h["lr"].split("/")[0])
            g_tb = int(g["tb"].split("/")[0])
            h_tb = int(h["tb"].split("/")[0])
            if abs(g_lr - h_lr) <= 5:
                agree_lr += 1
            if abs(g_tb - h_tb) <= 5:
                agree_tb += 1
        except (ValueError, IndexError):
            pass

    print(f"\nCards compared: {total}")
    if total > 0:
        print(f"L/R agreement (within 5%): {agree_lr}/{total} ({100*agree_lr/total:.0f}%)")
        print(f"T/B agreement (within 5%): {agree_tb}/{total} ({100*agree_tb/total:.0f}%)")

    # Border size analysis
    print(f"\nGradient border sizes (px):")
    all_g_borders = [r["borders"] for r in gradient_results if r.get("lr") not in ("FAIL", "ERR")]
    if all_g_borders:
        all_left = [b["left"] for b in all_g_borders]
        all_right = [b["right"] for b in all_g_borders]
        all_top = [b["top"] for b in all_g_borders]
        all_bottom = [b["bottom"] for b in all_g_borders]
        print(f"  Left:   min={min(all_left):>6.1f}  max={max(all_left):>6.1f}  mean={sum(all_left)/len(all_left):>6.1f}")
        print(f"  Right:  min={min(all_right):>6.1f}  max={max(all_right):>6.1f}  mean={sum(all_right)/len(all_right):>6.1f}")
        print(f"  Top:    min={min(all_top):>6.1f}  max={max(all_top):>6.1f}  mean={sum(all_top)/len(all_top):>6.1f}")
        print(f"  Bottom: min={min(all_bottom):>6.1f}  max={max(all_bottom):>6.1f}  mean={sum(all_bottom)/len(all_bottom):>6.1f}")

    print(f"\nHSV border sizes (px):")
    all_h_borders = [r["borders"] for r in hsv_results if r.get("lr") != "ERR"]
    if all_h_borders:
        all_left = [b["left"] for b in all_h_borders]
        all_right = [b["right"] for b in all_h_borders]
        all_top = [b["top"] for b in all_h_borders]
        all_bottom = [b["bottom"] for b in all_h_borders]
        print(f"  Left:   min={min(all_left):>6}  max={max(all_left):>6}  mean={sum(all_left)/len(all_left):>6.1f}")
        print(f"  Right:  min={min(all_right):>6}  max={max(all_right):>6}  mean={sum(all_right)/len(all_right):>6.1f}")
        print(f"  Top:    min={min(all_top):>6}  max={max(all_top):>6}  mean={sum(all_top)/len(all_top):>6.1f}")
        print(f"  Bottom: min={min(all_bottom):>6}  max={max(all_bottom):>6}  mean={sum(all_bottom)/len(all_bottom):>6.1f}")

    # Issues
    if issues:
        print(f"\nIssues detected ({len(issues)}):")
        for iss in issues:
            print(f"  {iss['pos']} {iss['name']}: {'; '.join(iss['notes'])}")

    print()
    print("NOTES ON BINDER SEGMENTS (1008x1530):")
    print("  - Gradient (centering.py): Designed for high-res scans (~1290 DPI).")
    print("    On binder segments, the outer boundary detection may struggle because")
    print("    the segmenter already crops close to the card edge.")
    print("  - HSV (centering_detector.py): Designed for perspective-corrected segments.")
    print("    Uses color-based border detection which works at any resolution,")
    print("    but depends on border color being detectable (yellow, silver, etc.).")
    print("  - Binder segments have imprecise borders from the segmenter -- the")
    print("    perspective warp + padding can shift the apparent centering.")


if __name__ == "__main__":
    main()
