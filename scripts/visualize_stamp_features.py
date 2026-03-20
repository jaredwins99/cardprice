#!/usr/bin/env python3
"""Visualize stamp detection features for the Dragon Frontiers binder page.

Generates:
  1. Per-card detail images: original + stamp/control regions + edge maps + heatmaps
  2. 3x3 grid with green/red borders for correct/incorrect classification
  3. Comparison strips: stamped vs normal vs holo stamp regions

Saves all output to data/condition_training/stamps_analysis/visualizations/
"""

import json
import sys
import os
from pathlib import Path

import cv2
import numpy as np

BASE = Path("/home/godli/cardprice")
INBOX = BASE / "data" / "inbox"
CARDS_DIR = INBOX / "page_20260305_094228_cards"
GT_PATH = BASE / "data" / "condition_training" / "stamps_real" / "binder_ground_truth.jsonl"
OUT_DIR = BASE / "data" / "condition_training" / "stamps_analysis" / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Region cropping (same definitions as stamp_pixel_analysis.py)
# ---------------------------------------------------------------------------

def crop_stamp_region(img: np.ndarray) -> np.ndarray:
    """Bottom-right of artwork area where EX-era stamp sits."""
    h, w = img.shape[:2]
    x1, x2 = int(w * 0.55), int(w * 0.90)
    y1, y2 = int(h * 0.45), int(h * 0.70)
    return img[y1:y2, x1:x2]


def stamp_region_coords(img: np.ndarray) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) for the stamp region."""
    h, w = img.shape[:2]
    return int(w * 0.55), int(h * 0.45), int(w * 0.90), int(h * 0.70)


def crop_control_region(img: np.ndarray) -> np.ndarray:
    """Left-side control region at same vertical band."""
    h, w = img.shape[:2]
    x1, x2 = int(w * 0.10), int(w * 0.45)
    y1, y2 = int(h * 0.45), int(h * 0.70)
    return img[y1:y2, x1:x2]


def control_region_coords(img: np.ndarray) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) for the control region."""
    h, w = img.shape[:2]
    return int(w * 0.10), int(h * 0.45), int(w * 0.45), int(h * 0.70)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_edge_density(gray: np.ndarray) -> float:
    return float(np.mean(cv2.Canny(gray, 50, 150) > 0))


def compute_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude, returned as float32 image."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return mag.astype(np.float32)


def compute_laplacian_var(gray: np.ndarray) -> float:
    return float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))


def compute_gold_pixel_ratio(img_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 40, 120]), np.array([40, 255, 255]))
    return float(np.mean(mask > 0))


# ---------------------------------------------------------------------------
# Load ground truth for Dragon Frontiers page
# ---------------------------------------------------------------------------

def load_ground_truth() -> list[dict]:
    """Load binder ground truth entries for page_20260305_094228."""
    entries = []
    with open(GT_PATH) as f:
        for line in f:
            entry = json.loads(line.strip())
            if "page_20260305_094228" in entry["image"]:
                entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Try to run the actual stamp classifier for correct/incorrect labels
# ---------------------------------------------------------------------------

def classify_cards(entries: list[dict], images: list[np.ndarray]) -> list[dict]:
    """Run stamp classifier on each card. Returns list of prediction dicts."""
    predictions = []
    try:
        sys.path.insert(0, str(BASE))
        from cardprice.ml.stamp_classifier import classify_stamp
        for i, (entry, img) in enumerate(zip(entries, images)):
            try:
                # Classifier expects file path or PIL Image, not numpy array
                img_path = str(INBOX / entry["image"])
                result = classify_stamp(img_path)
                predictions.append(result)
            except Exception as e:
                print(f"  Classifier error on card_{i:02d}: {e}")
                predictions.append({
                    "stamped": False, "confidence": 0.0, "stamp_probability": 0.0
                })
    except Exception as e:
        print(f"  Could not load stamp classifier: {e}")
        print("  Using ground truth only (all shown as 'correct').")
        for entry in entries:
            predictions.append({
                "stamped": entry["stamped"],
                "confidence": 1.0,
                "stamp_probability": 1.0 if entry["stamped"] else 0.0,
            })
    return predictions


# ---------------------------------------------------------------------------
# 1. Per-card detail visualizations
# ---------------------------------------------------------------------------

def make_per_card_viz(img: np.ndarray, entry: dict, pred: dict) -> np.ndarray:
    """Create a detail visualization for one card.

    Layout (roughly 600 wide x 500 tall):
        Top row: Original card with stamp+control regions highlighted
        Mid row: Edge map (Canny) of stamp region | Gradient heatmap of stamp region
        Bottom: Feature values as text overlay
    """
    card_h, card_w = img.shape[:2]

    # -- Annotated original card (resize to fixed height) --
    disp_h = 300
    scale = disp_h / card_h
    disp_w = int(card_w * scale)
    card_disp = cv2.resize(img, (disp_w, disp_h))

    # Draw stamp region rectangle (blue)
    sx1, sy1, sx2, sy2 = stamp_region_coords(img)
    cv2.rectangle(card_disp,
                  (int(sx1 * scale), int(sy1 * scale)),
                  (int(sx2 * scale), int(sy2 * scale)),
                  (0, 140, 255), 2)  # orange
    cv2.putText(card_disp, "STAMP", (int(sx1 * scale) + 2, int(sy1 * scale) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255), 1)

    # Draw control region rectangle (cyan)
    cx1, cy1, cx2, cy2 = control_region_coords(img)
    cv2.rectangle(card_disp,
                  (int(cx1 * scale), int(cy1 * scale)),
                  (int(cx2 * scale), int(cy2 * scale)),
                  (255, 200, 0), 2)  # cyan
    cv2.putText(card_disp, "CTRL", (int(cx1 * scale) + 2, int(cy1 * scale) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)

    # -- Stamp region crops for analysis --
    stamp_crop = crop_stamp_region(img)
    control_crop = crop_control_region(img)
    stamp_gray = cv2.cvtColor(stamp_crop, cv2.COLOR_BGR2GRAY)

    # Edge map (Canny)
    edges = cv2.Canny(stamp_gray, 50, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Gradient magnitude heatmap
    grad_mag = compute_gradient_magnitude(stamp_gray)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(grad_norm, cv2.COLORMAP_JET)

    # Resize stamp visualizations to uniform size
    crop_disp_w = disp_w // 2
    crop_disp_h = 120
    stamp_resized = cv2.resize(stamp_crop, (crop_disp_w, crop_disp_h))
    edges_resized = cv2.resize(edges_bgr, (crop_disp_w, crop_disp_h))
    heatmap_resized = cv2.resize(heatmap, (crop_disp_w, crop_disp_h))

    # Compute features
    edge_density = compute_edge_density(stamp_gray)
    laplacian = compute_laplacian_var(stamp_gray)
    gold_ratio = compute_gold_pixel_ratio(stamp_crop)
    ctrl_gray = cv2.cvtColor(control_crop, cv2.COLOR_BGR2GRAY)
    ctrl_edge_density = compute_edge_density(ctrl_gray)
    edge_ratio = edge_density / (ctrl_edge_density + 1e-10)

    # -- Assemble the panel --
    panel_w = max(disp_w, crop_disp_w * 2 + 10)
    panel_h = disp_h + crop_disp_h + 80  # card + crops + text
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    # Place card (centered)
    x_off = (panel_w - disp_w) // 2
    panel[0:disp_h, x_off:x_off + disp_w] = card_disp

    # Place edge map and heatmap side by side below card
    y_mid = disp_h + 5
    panel[y_mid:y_mid + crop_disp_h, 0:crop_disp_w] = edges_resized
    panel[y_mid:y_mid + crop_disp_h,
          crop_disp_w + 5:crop_disp_w * 2 + 5] = heatmap_resized

    # Labels for the crops
    cv2.putText(panel, "Canny Edges", (2, y_mid + crop_disp_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(panel, "Gradient Heatmap", (crop_disp_w + 7, y_mid + crop_disp_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Overlay edge density on the edge map
    cv2.putText(panel, f"density={edge_density:.3f}",
                (2, y_mid + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Feature text at bottom
    y_text = disp_h + crop_disp_h + 28
    name = entry.get("card_name", "?")
    gt_label = "STAMPED" if entry["stamped"] else "CLEAN"
    pred_label = "STAMPED" if pred["stamped"] else "CLEAN"
    correct = pred["stamped"] == entry["stamped"]

    text_color = (0, 255, 0) if correct else (0, 0, 255)
    cv2.putText(panel, f"{name} | GT:{gt_label} Pred:{pred_label}",
                (4, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    cv2.putText(panel,
                f"EdgeDens={edge_density:.3f}  EdgeRatio={edge_ratio:.2f}  "
                f"Gold={gold_ratio:.3f}  Lap={laplacian:.0f}  "
                f"StampProb={pred['stamp_probability']:.2f}",
                (4, y_text + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1)

    return panel


# ---------------------------------------------------------------------------
# 2. 3x3 grid with green/red borders and stamp crop overlay
# ---------------------------------------------------------------------------

def make_grid_3x3(images: list[np.ndarray], entries: list[dict],
                  predictions: list[dict]) -> np.ndarray:
    """Create a 3x3 grid showing all 9 cards with classification borders."""
    cell_w, cell_h = 280, 400
    border = 6
    grid_w = 3 * cell_w + 4 * border
    grid_h = 3 * cell_h + 4 * border + 30  # +30 for title
    grid = np.full((grid_h, grid_w, 3), 40, dtype=np.uint8)

    cv2.putText(grid, "Dragon Frontiers Binder - Stamp Detection (Green=Correct, Red=Wrong)",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for i in range(min(9, len(images))):
        row, col = i // 3, i % 3
        x0 = border + col * (cell_w + border)
        y0 = 30 + border + row * (cell_h + border)

        img = images[i]
        entry = entries[i]
        pred = predictions[i]
        correct = pred["stamped"] == entry["stamped"]

        # Border color
        border_color = (0, 200, 0) if correct else (0, 0, 230)

        # Draw border
        cv2.rectangle(grid, (x0 - border // 2, y0 - border // 2),
                      (x0 + cell_w + border // 2, y0 + cell_h + border // 2),
                      border_color, border)

        # Resize card to fit cell (leaving room for text)
        card_area_h = cell_h - 60
        card_h_orig, card_w_orig = img.shape[:2]
        scale = min(cell_w / card_w_orig, card_area_h / card_h_orig)
        new_w = int(card_w_orig * scale)
        new_h = int(card_h_orig * scale)
        card_resized = cv2.resize(img, (new_w, new_h))

        # Center card in cell
        cx = x0 + (cell_w - new_w) // 2
        cy = y0 + (card_area_h - new_h) // 2
        grid[cy:cy + new_h, cx:cx + new_w] = card_resized

        # Overlay enlarged stamp crop in top-right corner of the cell
        stamp_crop = crop_stamp_region(img)
        crop_h_small, crop_w_small = 55, 80
        stamp_small = cv2.resize(stamp_crop, (crop_w_small, crop_h_small))
        # Add thin white border around stamp crop
        cv2.rectangle(stamp_small, (0, 0), (crop_w_small - 1, crop_h_small - 1),
                      (255, 255, 255), 1)
        ox = x0 + cell_w - crop_w_small - 4
        oy = y0 + 4
        grid[oy:oy + crop_h_small, ox:ox + crop_w_small] = stamp_small

        # Text labels below card
        ty = y0 + card_area_h + 8
        name = entry.get("card_name", f"card_{i:02d}")
        gt_label = "STAMP" if entry["stamped"] else entry.get("variant", "normal").upper()
        cv2.putText(grid, f"{name}", (x0 + 4, ty + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(grid, f"GT:{gt_label}  P:{pred['stamp_probability']:.2f}",
                    (x0 + 4, ty + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # Stamp region features
        stamp_gray = cv2.cvtColor(stamp_crop, cv2.COLOR_BGR2GRAY)
        ed = compute_edge_density(stamp_gray)
        gold = compute_gold_pixel_ratio(stamp_crop)
        cv2.putText(grid, f"E:{ed:.3f} G:{gold:.3f}",
                    (x0 + 4, ty + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    return grid


# ---------------------------------------------------------------------------
# 3. Comparison strips by variant type
# ---------------------------------------------------------------------------

def make_comparison_strips(images: list[np.ndarray],
                           entries: list[dict]) -> np.ndarray:
    """Create comparison strips: stamped / normal / holo stamp regions."""

    # Categorize cards
    stamped_cards = [(img, e) for img, e in zip(images, entries) if e["stamped"]]
    normal_cards = [(img, e) for img, e in zip(images, entries)
                    if not e["stamped"] and e.get("variant") == "normal"]
    holo_cards = [(img, e) for img, e in zip(images, entries)
                  if not e["stamped"] and e.get("variant") == "holofoil"]

    crop_w, crop_h = 180, 120
    max_cols = max(len(stamped_cards), len(normal_cards), len(holo_cards), 1)
    label_w = 120
    strip_w = label_w + max_cols * (crop_w + 5) + 10
    row_h = crop_h + 40  # crop + text
    strip_h = 3 * row_h + 50  # 3 rows + title

    canvas = np.zeros((strip_h, strip_w, 3), dtype=np.uint8)
    cv2.putText(canvas, "Stamp Region Comparison by Variant Type",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    categories = [
        ("STAMPED", stamped_cards, (0, 0, 230)),
        ("NORMAL", normal_cards, (0, 200, 0)),
        ("HOLO", holo_cards, (230, 180, 0)),
    ]

    for row_idx, (label, cards, color) in enumerate(categories):
        y_base = 35 + row_idx * row_h

        # Row label
        cv2.putText(canvas, label, (5, y_base + crop_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        for col_idx, (img, entry) in enumerate(cards):
            x = label_w + col_idx * (crop_w + 5)

            stamp_crop = crop_stamp_region(img)
            resized = cv2.resize(stamp_crop, (crop_w, crop_h))

            # Compute features and overlay
            stamp_gray = cv2.cvtColor(stamp_crop, cv2.COLOR_BGR2GRAY)
            ed = compute_edge_density(stamp_gray)
            gold = compute_gold_pixel_ratio(stamp_crop)
            lap = compute_laplacian_var(stamp_gray)

            # Place crop
            canvas[y_base:y_base + crop_h, x:x + crop_w] = resized

            # Colored border
            cv2.rectangle(canvas, (x, y_base), (x + crop_w - 1, y_base + crop_h - 1),
                          color, 2)

            # Feature text below crop
            name = entry.get("card_name", "?")
            cv2.putText(canvas, f"{name}", (x + 2, y_base + crop_h + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 220, 220), 1)
            cv2.putText(canvas, f"E:{ed:.3f} G:{gold:.3f} L:{lap:.0f}",
                        (x + 2, y_base + crop_h + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (170, 170, 170), 1)

    return canvas


# ---------------------------------------------------------------------------
# 4. Edge map comparison: stamp region of all 9 cards
# ---------------------------------------------------------------------------

def make_edge_comparison(images: list[np.ndarray],
                         entries: list[dict]) -> np.ndarray:
    """Side-by-side edge maps of stamp regions for all 9 cards."""
    crop_w, crop_h = 160, 100
    cols = min(9, len(images))
    canvas_w = cols * (crop_w + 5) + 5
    row_h = crop_h * 2 + 50  # original + edges + text
    canvas_h = row_h + 30

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    cv2.putText(canvas, "Stamp Region: Original (top) vs Canny Edges (bottom)",
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for i, (img, entry) in enumerate(zip(images, entries)):
        x = 5 + i * (crop_w + 5)
        y_top = 25

        stamp_crop = crop_stamp_region(img)
        stamp_gray = cv2.cvtColor(stamp_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(stamp_gray, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        crop_resized = cv2.resize(stamp_crop, (crop_w, crop_h))
        edges_resized = cv2.resize(edges_bgr, (crop_w, crop_h))

        canvas[y_top:y_top + crop_h, x:x + crop_w] = crop_resized
        canvas[y_top + crop_h + 2:y_top + crop_h * 2 + 2, x:x + crop_w] = edges_resized

        # Border color by ground truth
        if entry["stamped"]:
            color = (0, 0, 230)  # red for stamped
        elif entry.get("variant") == "holofoil":
            color = (230, 180, 0)  # yellow-ish for holo
        else:
            color = (0, 200, 0)  # green for normal

        cv2.rectangle(canvas, (x - 1, y_top - 1),
                      (x + crop_w, y_top + crop_h * 2 + 2), color, 2)

        name = entry.get("card_name", f"card_{i:02d}")
        ed = compute_edge_density(stamp_gray)
        cv2.putText(canvas, f"{name}", (x + 2, y_top + crop_h * 2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1)
        cv2.putText(canvas, f"ED={ed:.3f}", (x + 2, y_top + crop_h * 2 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading Dragon Frontiers binder cards...")
    entries = load_ground_truth()
    if not entries:
        print("ERROR: No ground truth found for page_20260305_094228")
        sys.exit(1)

    # Sort by card index
    entries.sort(key=lambda e: e["image"])
    print(f"  Found {len(entries)} cards")

    # Load images
    images = []
    for entry in entries:
        img_path = INBOX / entry["image"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ERROR: Could not load {img_path}")
            sys.exit(1)
        images.append(img)
        name = entry.get("card_name", "?")
        gt = "stamped" if entry["stamped"] else entry.get("variant", "normal")
        print(f"    {entry['image']}: {name} ({gt})")

    # Run classifier
    print("\nRunning stamp classifier...")
    predictions = classify_cards(entries, images)

    # Print classification summary
    correct_count = 0
    for i, (entry, pred) in enumerate(zip(entries, predictions)):
        ok = pred["stamped"] == entry["stamped"]
        if ok:
            correct_count += 1
        status = "OK" if ok else "WRONG"
        name = entry.get("card_name", f"card_{i:02d}")
        print(f"  [{status}] {name:12s}  gt={'stamped' if entry['stamped'] else 'clean':7s}  "
              f"pred={'stamped' if pred['stamped'] else 'clean':7s}  "
              f"prob={pred['stamp_probability']:.3f}")
    print(f"  Accuracy: {correct_count}/{len(entries)}")

    # 1. Per-card detail visualizations
    print("\nGenerating per-card detail visualizations...")
    for i, (img, entry, pred) in enumerate(zip(images, entries, predictions)):
        panel = make_per_card_viz(img, entry, pred)
        name = entry.get("card_name", f"card_{i:02d}")
        out_path = OUT_DIR / f"detail_card_{i:02d}_{name.lower()}.png"
        cv2.imwrite(str(out_path), panel)
        print(f"  Saved: {out_path.name}")

    # 2. 3x3 grid
    print("\nGenerating 3x3 classification grid...")
    grid = make_grid_3x3(images, entries, predictions)
    grid_path = OUT_DIR / "grid_3x3_classification.png"
    cv2.imwrite(str(grid_path), grid)
    print(f"  Saved: {grid_path.name}")

    # 3. Comparison strips
    print("\nGenerating comparison strips...")
    strips = make_comparison_strips(images, entries)
    strips_path = OUT_DIR / "comparison_strips.png"
    cv2.imwrite(str(strips_path), strips)
    print(f"  Saved: {strips_path.name}")

    # 4. Edge comparison
    print("\nGenerating edge map comparison...")
    edge_comp = make_edge_comparison(images, entries)
    edge_path = OUT_DIR / "edge_comparison.png"
    cv2.imwrite(str(edge_path), edge_comp)
    print(f"  Saved: {edge_path.name}")

    print(f"\nAll visualizations saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
