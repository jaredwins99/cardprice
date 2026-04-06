#!/usr/bin/env python3
"""Test auto-cropping individual cards from bad slide-scan captures.

Each capture is a tall narrow strip (540x1344) that shows portions of a binder
page -- typically 2-3 partial cards, blue binder borders, background carpet, etc.
Goal: find and extract the single most-complete card from each frame.
"""

import cv2
import numpy as np
import os

INPUT_DIR = "data/inbox/slide_20260322_160339_cards"
OUTPUT_DIR = "data/inbox/slide_20260322_160339_cards/cropped"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Standard Pokemon card aspect ratio: 63mm x 88mm = 0.716
CARD_RATIO = 63.0 / 88.0  # ~0.716


def find_card_contours(img):
    """Find rectangular contours that could be Pokemon cards."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = []

    # Try multiple edge detection approaches
    for label, preprocessed in _preprocess_variants(gray):
        edges = cv2.Canny(preprocessed, 30, 120)
        # Dilate to close gaps in card borders
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            area = cv2.contourArea(cnt)
            img_area = w * h
            # Card should be at least 8% of image (these are tight crops)
            if area < 0.08 * img_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if 4 <= len(approx) <= 6:
                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                if rw < rh:
                    rw, rh = rh, rw
                ratio = rh / rw if rw > 0 else 0
                # Pokemon cards: 0.716 ratio, allow generous range
                if 0.55 < ratio < 0.90:
                    results.append({
                        "contour": cnt,
                        "approx": approx,
                        "area": area,
                        "area_pct": area / img_area * 100,
                        "ratio": ratio,
                        "rect": rect,
                        "method": label,
                        "vertices": len(approx),
                    })

    # Deduplicate: keep the best contour per overlapping region
    results.sort(key=lambda r: r["area"], reverse=True)
    filtered = []
    for r in results:
        cx, cy = r["rect"][0]
        duplicate = False
        for existing in filtered:
            ex, ey = existing["rect"][0]
            if abs(cx - ex) < 50 and abs(cy - ey) < 50:
                duplicate = True
                break
        if not duplicate:
            filtered.append(r)

    return filtered


def _preprocess_variants(gray):
    """Generate multiple preprocessed versions for edge detection."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    yield "gaussian", blur

    # Bilateral filter preserves edges better
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    yield "bilateral", bilateral

    # Adaptive threshold to find card borders against binder
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    yield "adaptive", thresh

    # Try on the blue channel (binder is blue, cards are not)
    yield "raw", gray


def warp_card(img, contour):
    """Perspective-warp a 4-point contour to a standard card rectangle."""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = box.astype(np.float32)

    # Order points: top-left, top-right, bottom-right, bottom-left
    pts = order_points(box)

    rw, rh = rect[1]
    if rw < rh:
        rw, rh = rh, rw
    # Output at standard card proportions
    out_w = int(rh)
    out_h = int(rw)
    if out_w < 100 or out_h < 100:
        return None

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h))

    # Ensure portrait orientation (taller than wide)
    wh, ww = warped.shape[:2]
    if ww > wh:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped


def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def try_color_segmentation(img):
    """Try to find cards by detecting the blue binder as background."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    # Blue binder: hue ~100-130, high saturation
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([135, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Invert: non-blue regions are potential cards
    card_mask = cv2.bitwise_not(blue_mask)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(cnt)
        if area < 0.08 * w * h:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw < rh:
            rw, rh = rh, rw
        ratio = rh / rw if rw > 0 else 0
        results.append({
            "contour": cnt,
            "area": area,
            "area_pct": area / (w * h) * 100,
            "ratio": ratio,
            "rect": rect,
            "method": "color_seg",
        })

    return results


def analyze_card(i, img):
    """Analyze one capture and return findings."""
    h, w = img.shape[:2]
    print(f"\n{'='*60}")
    print(f"card_{i:02d}.jpg  ({w}x{h})")
    print(f"{'='*60}")

    # Method 1: Contour-based
    contour_results = find_card_contours(img)
    print(f"  Contour method: {len(contour_results)} card-shaped contours found")
    for j, r in enumerate(contour_results):
        print(f"    [{j}] ratio={r['ratio']:.3f}, area={r['area_pct']:.1f}%, "
              f"vertices={r.get('vertices', '?')}, method={r['method']}")

    # Method 2: Color segmentation (blue binder)
    color_results = try_color_segmentation(img)
    print(f"  Color segmentation: {len(color_results)} non-blue regions found")
    for j, r in enumerate(color_results):
        print(f"    [{j}] ratio={r['ratio']:.3f}, area={r['area_pct']:.1f}%")

    # Pick best candidate: prefer contour results with good ratio
    best = None
    best_score = 0

    for r in contour_results:
        # Score: how close to card ratio, weighted by area
        ratio_score = 1.0 - abs(r["ratio"] - CARD_RATIO) / 0.3
        area_score = min(r["area_pct"] / 50.0, 1.0)  # normalize
        score = ratio_score * 0.6 + area_score * 0.4
        if score > best_score:
            best_score = score
            best = r

    # Also consider color segmentation
    for r in color_results:
        ratio_score = 1.0 - abs(r["ratio"] - CARD_RATIO) / 0.3
        area_score = min(r["area_pct"] / 50.0, 1.0)
        score = ratio_score * 0.5 + area_score * 0.3  # slightly less weight
        if score > best_score:
            best_score = score
            best = r

    if best is not None:
        print(f"  BEST: ratio={best['ratio']:.3f}, area={best['area_pct']:.1f}%, "
              f"method={best['method']}, score={best_score:.3f}")

        # Try to extract
        warped = warp_card(img, best["contour"])
        if warped is not None:
            out_path = os.path.join(OUTPUT_DIR, f"card_{i:02d}_cropped.jpg")
            cv2.imwrite(out_path, warped)
            wh, ww = warped.shape[:2]
            print(f"  SAVED: {out_path} ({ww}x{wh})")
        else:
            print(f"  WARNING: Warp failed (too small)")

        # Draw contour on debug image
        debug = img.copy()
        cv2.drawContours(debug, [best["contour"]], -1, (0, 255, 0), 3)
        debug_path = os.path.join(OUTPUT_DIR, f"card_{i:02d}_debug.jpg")
        cv2.imwrite(debug_path, debug)
    else:
        print(f"  FAILED: No card-shaped contour found")

    # Assess challenges
    challenges = []
    if len(contour_results) == 0 and len(color_results) == 0:
        challenges.append("No detectable card boundaries")
    if len(contour_results) > 2:
        challenges.append(f"Multiple cards visible ({len(contour_results)} contours)")
    if best and best["area_pct"] < 25:
        challenges.append("Card is small fraction of frame")
    if best and abs(best["ratio"] - CARD_RATIO) > 0.1:
        challenges.append(f"Ratio {best['ratio']:.3f} far from expected {CARD_RATIO:.3f}")

    # Check for blur
    laplacian_var = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if laplacian_var < 50:
        challenges.append(f"Very blurry (Laplacian var={laplacian_var:.1f})")
    elif laplacian_var < 150:
        challenges.append(f"Somewhat blurry (Laplacian var={laplacian_var:.1f})")

    if not challenges:
        challenges.append("Clean extraction possible")

    print(f"  Challenges: {'; '.join(challenges)}")

    return {
        "contours": len(contour_results),
        "color_regions": len(color_results),
        "best": best,
        "challenges": challenges,
        "blur_var": laplacian_var,
    }


def main():
    print("Slide-scan card auto-crop test")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    summaries = []
    for i in range(9):
        path = os.path.join(INPUT_DIR, f"card_{i:02d}.jpg")
        img = cv2.imread(path)
        if img is None:
            print(f"\ncard_{i:02d}: FILE NOT FOUND")
            continue
        result = analyze_card(i, img)
        summaries.append((i, result))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    extractable = 0
    for i, s in summaries:
        status = "YES" if s["best"] and s["best"]["area_pct"] > 15 else "NO"
        if status == "YES":
            extractable += 1
        blur_status = "BLURRY" if s["blur_var"] < 50 else "OK" if s["blur_var"] > 150 else "SOFT"
        print(f"  card_{i:02d}: contours={s['contours']}, "
              f"extractable={status}, blur={blur_status} ({s['blur_var']:.0f}), "
              f"challenges={s['challenges'][0]}")

    print(f"\nExtractable: {extractable}/{len(summaries)}")


if __name__ == "__main__":
    main()
