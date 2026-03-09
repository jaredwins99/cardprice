"""Binder page card segmenter using OpenCV.

Detects individual trading cards in a binder page photo (typically 3x3 grid)
using edge detection, contour finding, and perspective correction.

Pipeline:
1. Preprocess: grayscale -> blur -> adaptive threshold / Canny edges
2. Find contours -> filter for card-sized rectangles (approxPolyDP)
3. Sort into grid order (top-left to bottom-right)
4. Perspective-correct each card -> crop and save

References:
- OpenCV contour detection: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
- Magic Card Detector: https://tmikonen.github.io/quantitatively/2020-01-01-magic-card-detector/
- OpenCV Playing Card Detector: https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector
"""

import logging
from itertools import combinations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard Pokemon card dimensions (mm): 63 x 88 -> aspect ratio ~0.716
CARD_ASPECT_RATIO = 63.0 / 88.0  # width / height = ~0.716
ASPECT_RATIO_TOLERANCE = 0.20     # allow 20% deviation

# Output card image size (pixels) -- 5x standard 63x88mm at ~16px/mm
# Higher resolution preserves text detail for OCR (especially holographic text)
# Phone cameras send 4032x3024; in a 3x3 grid each card is ~1344x1008
CARD_OUTPUT_W = 1008
CARD_OUTPUT_H = 1530  # ~8% vertical padding over exact card ratio to avoid clipping names

# --- Card back / empty slot detection thresholds ---
# Pokemon card backs have a very specific orange-red colour profile:
# nearly 100% of pixels in the center region fall in the orange hue range
# (H 0-25 in OpenCV's 0-180 scale) with high saturation and very low
# hue variance (the colour is almost monochromatic).
#
# Tested on 18 card segments from two binder pages: zero false positives.
_CARD_BACK_ORANGE_FRAC_MIN = 0.92   # fraction of center pixels that are orange
_CARD_BACK_HUE_STD_MAX = 5.0        # max hue standard deviation (card backs ~0.8)
_CARD_BACK_SAT_MEAN_MIN = 140       # min mean saturation (card backs ~225)
_CARD_BACK_CENTER_MARGIN = 0.20     # crop 20% from each edge to avoid sleeve edges

# Target mean brightness for per-cell normalization
_BRIGHTNESS_TARGET = 128.0


def is_card_back(image_or_path, *, debug: bool = False) -> bool:
    """Detect if an image is a Pokemon card back (or empty binder slot showing a card back).

    Uses HSV colour histogram analysis of the center region.  Pokemon card
    backs are dominated by a narrow orange-red hue with very high saturation
    and almost zero hue variance -- a profile that no card front shares.

    Args:
        image_or_path: Either a BGR numpy array (from cv2.imread) or a
            path to an image file (str or Path).
        debug: If True, return a dict with diagnostic values instead of bool.

    Returns:
        True if the image is a card back / empty slot, False if it is a
        real card front.  When debug=True, returns a dict with the computed
        metrics and the final verdict.
    """
    # Load image if given a path
    if isinstance(image_or_path, (str, Path)):
        image = cv2.imread(str(image_or_path))
        if image is None:
            logger.warning("is_card_back: could not read image %s", image_or_path)
            return False
    else:
        image = image_or_path

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_img, w_img = hsv.shape[:2]

    # Crop to center region to avoid binder sleeve edges and reflections
    margin_y = int(h_img * _CARD_BACK_CENTER_MARGIN)
    margin_x = int(w_img * _CARD_BACK_CENTER_MARGIN)
    center = hsv[margin_y:h_img - margin_y, margin_x:w_img - margin_x]

    hue = center[:, :, 0].astype(np.float32)
    sat = center[:, :, 1].astype(np.float32)

    # Fraction of pixels in the orange hue range (H 0-25) with meaningful saturation
    orange_mask = (hue <= 25) & (sat > 50)
    orange_frac = float(orange_mask.sum()) / max(orange_mask.size, 1)

    # Hue standard deviation (only within the orange region to avoid noise from
    # any stray non-orange pixels inflating the std)
    if orange_mask.sum() > 100:
        hue_std = float(hue[orange_mask].std())
    else:
        hue_std = float(hue.std())

    sat_mean = float(sat.mean())

    is_back = (
        orange_frac >= _CARD_BACK_ORANGE_FRAC_MIN
        and hue_std <= _CARD_BACK_HUE_STD_MAX
        and sat_mean >= _CARD_BACK_SAT_MEAN_MIN
    )

    if is_back:
        logger.info("Card back detected: orange=%.3f hue_std=%.1f sat_mean=%.0f",
                     orange_frac, hue_std, sat_mean)
    else:
        logger.debug("Not card back: orange=%.3f hue_std=%.1f sat_mean=%.0f",
                      orange_frac, hue_std, sat_mean)

    if debug:
        return {
            "is_card_back": is_back,
            "orange_frac": orange_frac,
            "hue_std": hue_std,
            "sat_mean": sat_mean,
            "thresholds": {
                "orange_frac_min": _CARD_BACK_ORANGE_FRAC_MIN,
                "hue_std_max": _CARD_BACK_HUE_STD_MAX,
                "sat_mean_min": _CARD_BACK_SAT_MEAN_MIN,
            },
        }

    return is_back


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left.

    Uses sum and difference of coordinates to determine corners.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]       # bottom-right has largest sum
    rect[1] = pts[np.argmin(diff)]    # top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]    # bottom-left has largest difference
    return rect


def _perspective_crop(image: np.ndarray, pts: np.ndarray,
                      output_w: int = CARD_OUTPUT_W,
                      output_h: int = CARD_OUTPUT_H,
                      force_landscape: bool = False) -> np.ndarray:
    """Apply a four-point perspective transform to extract a card.

    Args:
        image: Source image (BGR).
        pts: 4 corner points of the card quadrilateral.
        output_w: Width of the output image.
        output_h: Height of the output image.
        force_landscape: If True, treat the quad as landscape even if its
            measured width <= height.  This handles cases where a partial
            or skewed contour in a landscape page appears portrait-shaped
            but the card content is actually rotated.

    Returns:
        Warped rectangular card image in portrait orientation.
    """
    ordered = _order_points(pts.reshape(4, 2))

    # Measure the width and height of the detected quadrilateral
    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bot = np.linalg.norm(ordered[2] - ordered[3])
    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    avg_w = (width_top + width_bot) / 2
    avg_h = (height_left + height_right) / 2

    is_landscape = avg_w > avg_h or force_landscape

    # If the detected quad is landscape (wider than tall), swap output dims
    # so we warp into landscape first, then rotate to portrait
    if is_landscape:
        warp_w, warp_h = output_h, output_w
    else:
        warp_w, warp_h = output_w, output_h

    dst = np.array([
        [0, 0],
        [warp_w - 1, 0],
        [warp_w - 1, warp_h - 1],
        [0, warp_h - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (warp_w, warp_h))

    # Rotate landscape cards to portrait
    if is_landscape:
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return warped


def _is_card_shaped(contour: np.ndarray, image_area: float,
                    min_area_frac: float = 0.005,
                    max_area_frac: float = 0.25) -> bool:
    """Check if a contour is roughly card-shaped (rectangular, correct aspect ratio).

    Args:
        contour: The contour to test.
        image_area: Total area of the source image.
        min_area_frac: Minimum contour area as fraction of image area.
        max_area_frac: Maximum contour area as fraction of image area.

    Returns:
        True if the contour looks like a card.
    """
    area = cv2.contourArea(contour)
    if area < image_area * min_area_frac or area > image_area * max_area_frac:
        return False

    # Check that the contour approximates to a quadrilateral
    peri = cv2.arcLength(contour, True)
    # Try increasing epsilon until we get 4 vertices (handles noisy edges)
    approx = None
    for eps_mult in (0.02, 0.04, 0.06, 0.08, 0.10):
        candidate = cv2.approxPolyDP(contour, eps_mult * peri, True)
        if len(candidate) == 4:
            approx = candidate
            break
    if approx is None:
        return False

    # Check aspect ratio using minimum area bounding rectangle
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    if w == 0 or h == 0:
        return False
    aspect = min(w, h) / max(w, h)
    if abs(aspect - CARD_ASPECT_RATIO) > ASPECT_RATIO_TOLERANCE:
        return False

    # Check solidity (area / convex hull area) — relaxed from strict convexity
    # to handle sleeves, reflections, and slight card curvature
    hull = cv2.convexHull(approx)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0 and cv2.contourArea(approx) / hull_area < 0.85:
        return False

    return True


def _find_card_contours(image: np.ndarray,
                        expected_count: int = 9) -> list[np.ndarray]:
    """Detect card-shaped contours in an image using multiple strategies.

    Uses a diverse set of edge detection and thresholding approaches to
    handle varied lighting, glare, and contrast conditions typical of
    binder page photos taken with phone cameras.

    The strategies are grouped into several categories:
    - Multiple Canny edge detections with different blur kernels and
      thresholds (sensitive vs conservative)
    - CLAHE contrast enhancement + Canny (handles uneven lighting)
    - Adaptive thresholding with morphological closing
    - Bilateral filter + Canny (preserves card edges, smooths textures)
    - Morphological gradient (edge detection via dilation minus erosion)
    - Otsu thresholding as a simple global baseline

    After deduplication, an area consistency filter removes false
    positives (e.g. inner art frames, binder hardware) by discarding
    contours whose area deviates too far from the median card area.

    Args:
        image: Source image (BGR).
        expected_count: Expected number of cards (used for area filtering).

    Returns:
        List of 4-point contour arrays, one per detected card.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Pad image so cards touching edges get closed contours.
    # Use BORDER_REPLICATE to avoid creating artificial edges from black borders.
    pad = int(max(h, w) * 0.02)  # 2% of image dimension
    gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    image_area = h * w  # use original area for size thresholds

    candidates = []  # list of (cx, cy, area, approx)

    def _add_candidates(edges: np.ndarray):
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if not _is_card_shaped(cnt, image_area):
                continue
            peri = cv2.arcLength(cnt, True)
            approx = None
            for eps_mult in (0.02, 0.04, 0.06, 0.08, 0.10):
                candidate = cv2.approxPolyDP(cnt, eps_mult * peri, True)
                if len(candidate) == 4:
                    approx = candidate
                    break
            if approx is None:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area = cv2.contourArea(cnt)
            candidates.append((cx, cy, area, approx))

    # Reusable kernels
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # --- Strategy 1: Canny with wider blur (robust to noise) ---
    blurred_7 = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred_7, 20, 80)
    edges = cv2.dilate(edges, kernel_5, iterations=2)
    _add_candidates(edges)

    # --- Strategy 2: Canny with narrow blur, high thresholds (sharp edges) ---
    blurred_3 = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred_3, 50, 150)
    edges = cv2.dilate(edges, kernel_5, iterations=2)
    _add_candidates(edges)

    # --- Strategy 3: Canny with narrow blur, low thresholds (sensitive) ---
    edges = cv2.Canny(blurred_3, 15, 60)
    edges = cv2.dilate(edges, kernel_3, iterations=1)
    _add_candidates(edges)

    # --- Strategy 4: Original Canny (moderate params) ---
    blurred_5 = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred_5, 30, 100)
    edges = cv2.dilate(edges, kernel_3, iterations=1)
    _add_candidates(edges)

    # --- Strategy 5+: CLAHE contrast enhancement + Canny ---
    # CLAHE normalizes local contrast, helping with uneven binder page lighting.
    # Multiple clip limits and tile sizes provide robustness across different
    # lighting conditions and card contrasts.
    for clip_limit, tile_size in [(3.0, 8), (4.0, 8), (2.0, 8),
                                  (4.0, 16), (3.0, 16)]:
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(gray)
        enh_blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
        for canny_lo, canny_hi in [(20, 80), (30, 100), (15, 60)]:
            edges = cv2.Canny(enh_blurred, canny_lo, canny_hi)
            edges = cv2.dilate(edges, kernel_3, iterations=1)
            _add_candidates(edges)

    # --- Strategy 8-9: Adaptive thresholding (robust to uneven lighting) ---
    for block_size, C, close_k, close_i in [(51, 5, 5, 1), (51, 5, 7, 2)]:
        thresh = cv2.adaptiveThreshold(
            blurred_5, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, C
        )
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=close_i)
        _add_candidates(thresh)

    # --- Strategy 10-12: Adaptive thresholding variants ---
    # These wider-block and higher-C adaptive thresholds detect card
    # boundaries that other strategies miss in challenging lighting.
    for blur_k, block_size, C, close_k, close_i in [
        (7, 51, 8, 7, 2),   # wider blur, higher C
        (7, 51, 5, 5, 2),   # wider blur, moderate C, heavy closing
        (7, 61, 5, 5, 2),   # even wider block size
    ]:
        thresh = cv2.adaptiveThreshold(
            cv2.GaussianBlur(gray, (blur_k, blur_k), 0), 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, C
        )
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=close_i)
        _add_candidates(thresh)

    # --- Strategy 13: Otsu thresholding (simple global threshold) ---
    _, thresh_otsu = cv2.threshold(blurred_5, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel_5)
    _add_candidates(thresh_otsu)

    # --- Bilateral filter + Canny (edge-preserving smooth) ---
    bilateral = cv2.bilateralFilter(gray, 11, 75, 75)
    edges = cv2.Canny(bilateral, 20, 80)
    edges = cv2.dilate(edges, kernel_5, iterations=2)
    _add_candidates(edges)

    bilateral_strong = cv2.bilateralFilter(gray, 15, 100, 100)
    edges = cv2.Canny(bilateral_strong, 15, 60)
    edges = cv2.dilate(edges, kernel_5, iterations=2)
    _add_candidates(edges)

    # --- Morphological gradient (dilation - erosion) ---
    gradient = cv2.morphologyEx(blurred_5, cv2.MORPH_GRADIENT, kernel_5)
    _, edges = cv2.threshold(gradient, 20, 255, cv2.THRESH_BINARY)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_5, iterations=2)
    _add_candidates(edges)

    # --- Canny with extra dilation for large blur ---
    for blur_k in (9, 11):
        blurred_wide = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        edges = cv2.Canny(blurred_wide, 20, 80)
        edges = cv2.dilate(edges, kernel_5, iterations=2)
        _add_candidates(edges)

    # --- Canny with dilate-then-erode (closes gaps, restores edge width) ---
    for blur_k, lo, hi in [(5, 20, 80), (7, 15, 60)]:
        blurred_de = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        edges = cv2.Canny(blurred_de, lo, hi)
        edges = cv2.dilate(edges, kernel_5, iterations=2)
        edges = cv2.erode(edges, kernel_5, iterations=1)
        _add_candidates(edges)

    # --- Extra Canny variant with wider dilation ---
    edges = cv2.Canny(blurred_5, 10, 50)
    edges = cv2.dilate(edges, kernel_5, iterations=2)
    _add_candidates(edges)

    # Deduplicate overlapping detections: if two contour centers are within
    # dedup_radius, keep only the one with the larger area.  This prevents
    # inner card art frames from being detected as separate cards.
    dedup_radius = max(w, h) * 0.10  # ~10% of image dimension
    # Sort largest first so the biggest contour wins
    candidates.sort(key=lambda c: c[2], reverse=True)
    kept: list[np.ndarray] = []
    kept_centers: list[tuple[int, int]] = []
    kept_areas: list[float] = []
    for cx, cy, area, approx in candidates:
        too_close = False
        for kx, ky in kept_centers:
            if abs(cx - kx) < dedup_radius and abs(cy - ky) < dedup_radius:
                too_close = True
                break
        if not too_close:
            kept.append(approx)
            kept_centers.append((cx, cy))
            kept_areas.append(area)

    # Subtract padding offset so coordinates map back to original image
    for i, approx in enumerate(kept):
        kept[i] = approx - pad

    n_before_filter = len(kept)

    # --- Area consistency filter ---
    # Cards in a binder page should all be roughly the same size.
    # If we found more than expected, remove contours whose area deviates
    # most from the median.  False positives (inner art frames, binder
    # hardware) are typically much smaller than real cards.
    if len(kept) > expected_count and len(kept_areas) >= 3:
        sorted_areas = sorted(kept_areas)
        median_area = sorted_areas[len(sorted_areas) // 2]

        # Compute deviation from median for each contour
        deviations = []
        for i, area in enumerate(kept_areas):
            dev = abs(area - median_area) / median_area
            deviations.append((dev, i))

        # Remove worst outliers to get down to expected_count, but only
        # if their deviation exceeds 40% (i.e. clearly not a real card).
        # Real cards vary by ~30% max due to perspective; false positives
        # are typically 80%+ smaller than the median.
        deviations.sort(reverse=True)
        to_remove = set()
        for dev, idx in deviations:
            if len(kept) - len(to_remove) <= expected_count:
                break
            if dev > 0.40:
                to_remove.add(idx)
                logger.debug("Area filter: removing contour %d "
                             "(area=%.0f, median=%.0f, dev=%.2f)",
                             idx, kept_areas[idx], median_area, dev)

        if to_remove:
            kept = [c for i, c in enumerate(kept) if i not in to_remove]

    logger.info("Found %d card-shaped contours (before dedup: %d, "
                "before area filter: %d)",
                len(kept), len(candidates), n_before_filter)
    return kept


def _sort_grid(contours: list[np.ndarray]) -> list[np.ndarray]:
    """Sort contours into reading order (left-to-right, top-to-bottom).

    Groups contours into rows by their vertical center, then sorts
    each row left-to-right.
    """
    if not contours:
        return contours

    # Compute centers
    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            cx, cy = cnt.mean(axis=0).flatten()
        else:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        centers.append((cx, cy))

    # Sort by y first, then group into rows
    indexed = sorted(enumerate(centers), key=lambda x: x[1][1])

    # Determine row breaks: gap > 20% of average card height
    if len(indexed) > 1:
        avg_height = np.mean([cv2.boundingRect(c)[3] for c in contours])
        row_gap_threshold = avg_height * 0.3

        rows = [[indexed[0]]]
        for i in range(1, len(indexed)):
            if indexed[i][1][1] - indexed[i - 1][1][1] > row_gap_threshold:
                rows.append([])
            rows[-1].append(indexed[i])

        # Sort each row by x
        sorted_indices = []
        for row in rows:
            row.sort(key=lambda x: x[1][0])
            sorted_indices.extend([idx for idx, _ in row])
    else:
        sorted_indices = [indexed[0][0]]

    return [contours[i] for i in sorted_indices]


def _detect_cell_rotation(cell_region: np.ndarray,
                           ref_card: np.ndarray) -> int:
    """Determine which 90-degree rotation makes a landscape cell match a reference card.

    Compares both CW and CCW rotations of the cell against a known-correct
    reference card image using pixel correlation.

    Args:
        cell_region: Landscape cell image to test.
        ref_card: A correctly-oriented card image from contour detection.

    Returns:
        cv2.ROTATE_90_CLOCKWISE or cv2.ROTATE_90_COUNTERCLOCKWISE.
    """
    ref_gray = cv2.cvtColor(ref_card, cv2.COLOR_BGR2GRAY)
    rh, rw = ref_gray.shape[:2]

    best_rot = cv2.ROTATE_90_COUNTERCLOCKWISE
    best_corr = -2.0

    for rot_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        rotated = cv2.rotate(cell_region, rot_code)
        resized = cv2.resize(rotated, (rw, rh), interpolation=cv2.INTER_AREA)
        resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        corr = np.corrcoef(ref_gray.flatten().astype(float),
                           resized_gray.flatten().astype(float))[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_rot = rot_code

    rot_name = "CW" if best_rot == cv2.ROTATE_90_CLOCKWISE else "CCW"
    logger.debug("Cell rotation check: best=%s (corr=%.3f)", rot_name, best_corr)
    return best_rot


def _normalize_brightness(image: np.ndarray,
                           target: float = _BRIGHTNESS_TARGET) -> np.ndarray:
    """Normalize an image so its mean brightness is close to `target`.

    Uses simple linear scaling: pixel_values * (target / current_mean).
    Scales all channels uniformly to preserve colour ratios.

    Args:
        image: BGR image (uint8).
        target: Desired mean brightness (0-255). Default 128.

    Returns:
        Brightness-normalized BGR image (uint8).
    """
    current_mean = float(np.mean(image))

    if current_mean < 1.0:
        # Near-black image, avoid division by near-zero
        logger.debug("Brightness normalization skipped: mean=%.1f (near black)",
                      current_mean)
        return image

    scale = target / current_mean
    result = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    logger.debug("Brightness normalized: %.1f -> %.1f (scale=%.3f)",
                  current_mean, target, scale)
    return result



def _find_grid_lines(rectified_page, rows, cols):
    """Detect non-uniform grid cell boundaries using intensity projection.

    Projects the grayscale image intensity horizontally (mean of columns per
    row) and vertically (mean of rows per column) to produce 1-D profiles.
    The dark plastic gutters between cards show up as valleys (local minima)
    in these profiles.  Valley positions define cell boundaries that adapt
    to uneven outer margins vs inner gutters.

    Args:
        rectified_page: BGR image of the cropped binder page.
        rows: Expected number of card rows in this image orientation
            (already swapped for landscape pages if needed).
        cols: Expected number of card columns in this image orientation.

    Returns:
        ``(row_boundaries, col_boundaries)`` where each is a list of
        ``(start, end)`` pixel ranges for each cell, or ``None`` if
        valley detection fails (not enough valleys found).
    """
    gray = cv2.cvtColor(rectified_page, cv2.COLOR_BGR2GRAY)
    page_h, page_w = gray.shape[:2]

    # Horizontal projection: mean brightness of each pixel row -> 1D profile along Y
    h_proj = gray.mean(axis=1).astype(np.float64)
    # Vertical projection: mean brightness of each pixel column -> 1D profile along X
    v_proj = gray.mean(axis=0).astype(np.float64)

    def find_valleys(profile, n_cells, axis_len):
        if n_cells <= 1:
            return []
        # Smooth to suppress noise (kernel ~ 2% of axis length, must be odd)
        kernel_size = max(3, int(axis_len * 0.02) | 1)
        smoothed = cv2.GaussianBlur(profile.reshape(-1, 1),
                                     (1, kernel_size), 0).flatten()
        expected_cell = axis_len / n_cells
        # Exclude the outer margins: inter-card valleys can only occur
        # between cells, not at the page edges.  The first valley is at
        # roughly 1 * expected_cell, so we exclude the first and last
        # half-cell to prevent page-boundary dark bands (which are often
        # deeper than actual gutters) from being selected.
        margin = int(expected_cell * 0.50)
        minima_idx = []
        minima_val = []
        for i in range(margin, len(smoothed) - margin):
            if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
                minima_idx.append(i)
                minima_val.append(smoothed[i])
        if len(minima_idx) < n_cells - 1:
            logger.debug("Valley detection: only %d minima found, need %d",
                         len(minima_idx), n_cells - 1)
            return None
        # Score each minimum by depth relative to local neighbourhood
        neighbourhood = int(expected_cell * 0.3)
        scored = []
        for idx, val in zip(minima_idx, minima_val):
            lo = max(0, idx - neighbourhood)
            hi = min(len(smoothed), idx + neighbourhood)
            local_mean = smoothed[lo:hi].mean()
            depth = local_mean - val
            scored.append((idx, depth))

        # Pre-filter to top candidates by depth (keep top 3x needed)
        scored.sort(key=lambda x: x[1], reverse=True)
        n_needed = n_cells - 1
        top_k = min(len(scored), max(n_needed * 3, 8))
        candidates = scored[:top_k]

        if len(candidates) < n_needed:
            logger.debug("Valley detection: only %d candidates, need %d",
                         len(candidates), n_needed)
            return None

        # Evaluate all valid combinations of n_needed valleys.
        # Score = total_depth - penalty * cell_size_variance.
        # This prefers deep valleys that also produce uniform cells.
        min_spacing = expected_cell * 0.4
        # Normalize depth so penalty weight is comparable
        max_depth = candidates[0][1] if candidates[0][1] > 0 else 1.0

        best_combo = None
        best_score = -float("inf")

        for combo in combinations(range(len(candidates)), n_needed):
            idxs = sorted([candidates[c][0] for c in combo])
            depths = [candidates[c][1] for c in combo]

            # Check minimum spacing between selected valleys
            valid = True
            for i in range(len(idxs) - 1):
                if idxs[i + 1] - idxs[i] < min_spacing:
                    valid = False
                    break
            if not valid:
                continue

            # Check cell sizes are in range
            boundaries = [0] + idxs + [axis_len]
            cell_sizes = [boundaries[i + 1] - boundaries[i]
                          for i in range(len(boundaries) - 1)]
            out_of_range = False
            for cs in cell_sizes:
                if cs < expected_cell * 0.60 or cs > expected_cell * 1.50:
                    out_of_range = True
                    break
            if out_of_range:
                continue

            # Score: total depth (normalized) minus cell size variance penalty
            total_depth = sum(depths) / max_depth
            size_std = float(np.std(cell_sizes))
            size_penalty = size_std / expected_cell  # 0 = perfectly uniform
            score = total_depth - 1.5 * size_penalty

            if score > best_score:
                best_score = score
                best_combo = idxs

        if best_combo is None:
            logger.debug("Valley detection: no valid valley combination found")
            return None

        # Post-validation: reject if cells are too uneven (max/min ratio > 1.5)
        boundaries = [0] + list(best_combo) + [axis_len]
        cell_sizes = [boundaries[i + 1] - boundaries[i]
                      for i in range(len(boundaries) - 1)]
        if max(cell_sizes) / max(min(cell_sizes), 1) > 1.5:
            logger.debug("Valley detection: rejected combo %s — cell sizes %s "
                         "too uneven (ratio %.2f)",
                         best_combo, cell_sizes,
                         max(cell_sizes) / max(min(cell_sizes), 1))
            return None

        # Refine each valley to the deepest point on the smoothed profile
        # within a small neighbourhood.  The combinatorial scoring picks
        # from a discrete set of detected local minima; the actual
        # minimum in the smoothed curve may be a few pixels away.
        refine_radius = int(expected_cell * 0.04)
        refined = []
        for v in best_combo:
            lo = max(0, v - refine_radius)
            hi = min(axis_len, v + refine_radius + 1)
            local_min_offset = int(np.argmin(smoothed[lo:hi]))
            refined.append(lo + local_min_offset)
        selected = refined
        logger.debug("Valley refinement: %s -> %s (radius=%d)",
                     best_combo, refined, refine_radius)
        return selected

    row_valleys = find_valleys(h_proj, rows, page_h)
    col_valleys = find_valleys(v_proj, cols, page_w)

    if row_valleys is None or col_valleys is None:
        logger.info("Valley-based grid detection failed (rows_ok=%s, cols_ok=%s), "
                     "will use uniform grid",
                     row_valleys is not None, col_valleys is not None)
        return None

    def valleys_to_boundaries(valleys, axis_len):
        edges = [0] + valleys + [axis_len]
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    row_bounds = valleys_to_boundaries(row_valleys, page_h)
    col_bounds = valleys_to_boundaries(col_valleys, page_w)

    logger.info("Valley grid detection succeeded: row_bounds=%s, col_bounds=%s",
                [(s, e) for s, e in row_bounds],
                [(s, e) for s, e in col_bounds])

    return row_bounds, col_bounds


def _contour_guided_grid(image: np.ndarray, contours: list[np.ndarray],
                         rows: int = 3, cols: int = 3) -> Optional[list[np.ndarray]]:
    """Use detected contour positions to infer grid boundaries and extract all cells.

    When contour detection finds >=6 but <9 cards, we can use their positions
    to compute row/column boundaries more accurately than page-outline +
    uniform subdivision.  This avoids the systematic offset that uniform
    subdivision produces when the page has uneven margins.

    Returns list of card images in reading order, or None if contour positions
    don't form a clean grid.
    """
    h, w = image.shape[:2]

    # Get bounding boxes and centers for each contour
    boxes = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        cx, cy = x + cw / 2, y + ch / 2
        boxes.append((x, y, cw, ch, cx, cy))

    # Filter out undersized contours (< 50% of median area)
    areas = [cw * ch for _, _, cw, ch, _, _ in boxes]
    median_area = sorted(areas)[len(areas) // 2]
    good_boxes = [(x, y, cw, ch, cx, cy)
                  for x, y, cw, ch, cx, cy in boxes
                  if cw * ch >= median_area * 0.5]

    if len(good_boxes) < 6:
        logger.debug("Contour-guided grid: only %d good contours, need >=6",
                     len(good_boxes))
        return None

    # Cluster centers into rows and columns using k-means-style binning
    cy_vals = sorted(set(cy for _, _, _, _, _, cy in good_boxes))
    cx_vals = sorted(set(cx for _, _, _, _, cx, _ in good_boxes))

    def cluster_1d(vals, n_clusters):
        """Simple 1D clustering into n_clusters groups."""
        if len(vals) < n_clusters:
            return None
        vals = sorted(vals)
        # Use gaps to split
        if len(vals) == n_clusters:
            return [[v] for v in vals]
        gaps = [(vals[i + 1] - vals[i], i) for i in range(len(vals) - 1)]
        gaps.sort(reverse=True)
        split_points = sorted([g[1] for g in gaps[:n_clusters - 1]])
        clusters = []
        prev = 0
        for sp in split_points:
            clusters.append(vals[prev:sp + 1])
            prev = sp + 1
        clusters.append(vals[prev:])
        return clusters if len(clusters) == n_clusters else None

    row_clusters = cluster_1d(
        [cy for _, _, _, _, _, cy in good_boxes], rows)
    col_clusters = cluster_1d(
        [cx for _, _, _, _, cx, _ in good_boxes], cols)

    if row_clusters is None or col_clusters is None:
        logger.debug("Contour-guided grid: clustering failed")
        return None

    # Compute row/col boundaries from contour positions
    # Each row boundary = midpoint between bottom of row N and top of row N+1
    # Use the actual card tops/bottoms from contours assigned to each row
    row_info = []  # (min_y, max_y_plus_h) per row
    for rc in row_clusters:
        rc_center = np.mean(rc)
        row_boxes = [(x, y, cw, ch) for x, y, cw, ch, cx, cy in good_boxes
                     if abs(cy - rc_center) < h / (rows * 2)]
        if not row_boxes:
            return None
        min_y = min(y for _, y, _, _ in row_boxes)
        max_y = max(y + ch for _, y, _, ch in row_boxes)
        row_info.append((min_y, max_y))

    col_info = []
    for cc in col_clusters:
        cc_center = np.mean(cc)
        col_boxes = [(x, y, cw, ch) for x, y, cw, ch, cx, cy in good_boxes
                     if abs(cx - cc_center) < w / (cols * 2)]
        if not col_boxes:
            return None
        min_x = min(x for x, _, _, _ in col_boxes)
        max_x = max(x + cw for x, _, cw, _ in col_boxes)
        col_info.append((min_x, max_x))

    # Build boundaries: row_start = min(tops), row_end = max(bottoms)
    # Gutter midpoints between adjacent rows
    row_bounds = []
    for i in range(rows):
        if i == 0:
            top = max(0, row_info[i][0] - 5)
        else:
            top = (row_info[i - 1][1] + row_info[i][0]) // 2
        if i == rows - 1:
            bot = min(h, row_info[i][1] + 5)
        else:
            bot = (row_info[i][1] + row_info[i + 1][0]) // 2
        row_bounds.append((top, bot))

    col_bounds = []
    for i in range(cols):
        if i == 0:
            left = max(0, col_info[i][0] - 5)
        else:
            left = (col_info[i - 1][1] + col_info[i][0]) // 2
        if i == cols - 1:
            right = min(w, col_info[i][1] + 5)
        else:
            right = (col_info[i][1] + col_info[i + 1][0]) // 2
        col_bounds.append((left, right))

    logger.info("Contour-guided grid: row_bounds=%s, col_bounds=%s",
                row_bounds, col_bounds)

    # Extract cells in reading order
    cards = []
    for r in range(rows):
        for c in range(cols):
            y1, y2 = row_bounds[r]
            x1, x2 = col_bounds[c]
            cell = image[y1:y2, x1:x2]
            # Ensure portrait
            ch, cw = cell.shape[:2]
            if cw > ch:
                cell = cv2.rotate(cell, cv2.ROTATE_90_COUNTERCLOCKWISE)
            card = cv2.resize(cell, (CARD_OUTPUT_W, CARD_OUTPUT_H),
                              interpolation=cv2.INTER_AREA)
            cards.append(card)

    return cards


def _grid_fallback(image: np.ndarray, rows: int = 3, cols: int = 3,
                    pad_frac: float = 0.035,
                    ref_contours: Optional[list[np.ndarray]] = None,
                    ) -> list[np.ndarray]:
    """Extract cards by dividing the binder page area into a uniform grid.

    Used as a fallback when contour detection misses cards. Finds the
    binder page outline, perspective-warps to a rectangle (correcting
    camera angle distortion), then subdivides into rows x cols cells.
    Falls back to bounding-box crop if perspective warp fails.

    Each extracted cell is brightness-normalized to mean ~128 to correct
    for uneven lighting across the binder page.

    For landscape images (page rotated in photo), uses any available
    reference cards from contour detection to determine the correct
    rotation and reading order of grid cells.

    The rotation direction of landscape cells determines the grid
    traversal order:
    - CCW rotation (card tops pointed right in image) means the binder
      was rotated 90 CW, so reading order starts from the image's
      bottom-right corner (reverse traversal).
    - CW rotation (card tops pointed left) means the binder was rotated
      90 CCW, so reading order starts from image's top-left (forward).

    Args:
        image: Source image (BGR).
        rows: Number of rows in the binder grid.
        cols: Number of columns in the binder grid.
        pad_frac: Fraction of cell size to pad inward at page-edge
            boundaries only (avoids binder frame/sleeve edges).  Interior
            valley boundaries use zero padding since the valley refinement
            step places them at the true gutter centre.  Only the
            outermost edges of the grid (first/last row/col) apply this
            inward padding to skip the binder frame.
        ref_contours: Optional list of card contours detected by
            _find_card_contours, used to calibrate orientation for
            landscape pages.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Try to find the binder page outline (largest rectangular contour)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    page_corners = None

    for thresh_fn in [
        lambda g: cv2.Canny(g, 20, 60),
        lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 51, 5),
    ]:
        edges = thresh_fn(blurred)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        # Find largest contour that's at least 40% of image area
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(cnt)
            if area < h * w * 0.4:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.02, 0.04, 0.06, 0.08):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4:
                    page_corners = approx
                    break
            if page_corners is not None:
                break
        if page_corners is not None:
            break

    # Rectify the page: prefer perspective warp (corrects camera angle
    # distortion) over plain bounding-box crop.
    perspective_warped = False
    if page_corners is not None:
        try:
            ordered = _order_points(
                page_corners.reshape(4, 2).astype(np.float32)
            )

            # Expand the detected page corners outward by a small margin.
            # The contour detection may find an outline slightly inside the
            # actual page, clipping cards at the edge (especially card
            # names at the top of the page which, after landscape rotation,
            # become the left edge of the first column).  Expanding the
            # source quad by ~4% from the centroid ensures the rectified
            # image includes a safety margin beyond the detected page
            # outline.
            centroid = ordered.mean(axis=0)
            expand_frac = 0.02
            ordered_expanded = centroid + (1.0 + expand_frac) * (ordered - centroid)

            # Pad the source image so expanded corners never get clamped
            # at image edges. This prevents asymmetric warping when the
            # page is near a camera boundary.
            pad_needed = int(max(w, h) * expand_frac) + 10
            padded_image = cv2.copyMakeBorder(
                image, pad_needed, pad_needed, pad_needed, pad_needed,
                cv2.BORDER_REPLICATE,
            )
            # Shift corner coordinates to account for padding
            ordered_expanded += pad_needed

            # Compute destination size from average edge lengths of the
            # *original* (unexpanded) corners to preserve the page's
            # natural aspect ratio.
            width_top = np.linalg.norm(ordered[1] - ordered[0])
            width_bot = np.linalg.norm(ordered[2] - ordered[3])
            height_left = np.linalg.norm(ordered[3] - ordered[0])
            height_right = np.linalg.norm(ordered[2] - ordered[1])
            dst_w = int((width_top + width_bot) / 2)
            dst_h = int((height_left + height_right) / 2)

            if dst_w > 100 and dst_h > 100:
                dst = np.array([
                    [0, 0],
                    [dst_w - 1, 0],
                    [dst_w - 1, dst_h - 1],
                    [0, dst_h - 1],
                ], dtype=np.float32)
                M_warp = cv2.getPerspectiveTransform(ordered_expanded, dst)
                rectified = cv2.warpPerspective(padded_image, M_warp, (dst_w, dst_h))
                perspective_warped = True
                # After perspective warp, page fills the entire rectified
                # image so crop offset is zero
                bx, by = 0, 0
                logger.info(
                    "Grid fallback: perspective-warped page to %dx%d",
                    dst_w, dst_h,
                )
            else:
                raise ValueError("Destination rectangle too small")
        except Exception as e:
            logger.warning(
                "Grid fallback: perspective warp failed (%s), "
                "falling back to bbox crop", e,
            )
            perspective_warped = False

    if not perspective_warped and page_corners is not None:
        bx, by, bw, bh = cv2.boundingRect(page_corners)
        # Clamp to image bounds
        bx = max(0, bx)
        by = max(0, by)
        bx2 = min(w, bx + bw)
        by2 = min(h, by + bh)
        rectified = image[by:by2, bx:bx2]
        logger.info("Grid fallback: cropped to page bbox %dx%d at (%d,%d)",
                     bx2 - bx, by2 - by, bx, by)
    elif not perspective_warped:
        logger.info("Grid fallback: no page outline found, using full image")
        rectified = image
        bx, by = 0, 0

    page_h, page_w = rectified.shape[:2]
    page_is_landscape = page_w > page_h

    if page_is_landscape:
        # Binder page is portrait but image is landscape -- the page is
        # rotated ~90 degrees in the photo.  Grid rows in the image
        # correspond to binder columns and vice versa, so we swap the
        # subdivision dimensions.
        img_rows, img_cols = cols, rows
        logger.info("Grid fallback: landscape page detected, swapping grid dims to %dx%d",
                     img_rows, img_cols)
    else:
        img_rows, img_cols = rows, cols

    # --- Attempt valley-based non-uniform grid detection ---
    grid_lines = _find_grid_lines(rectified, img_rows, img_cols)
    use_valleys = grid_lines is not None

    if use_valleys:
        row_bounds, col_bounds = grid_lines
        logger.info("Grid fallback: using valley-based non-uniform grid")
    else:
        # Uniform subdivision fallback
        logger.info("Grid fallback: using uniform grid subdivision")
        cell_h = page_h / img_rows
        cell_w = page_w / img_cols
        row_bounds = [(int(r * cell_h), int((r + 1) * cell_h))
                      for r in range(img_rows)]
        col_bounds = [(int(c * cell_w), int((c + 1) * cell_w))
                      for c in range(img_cols)]

    # For landscape pages, determine:
    # 1. Which 90-degree rotation to apply to each landscape cell
    # 2. Grid traversal order (derived from the rotation direction)
    #
    # The rotation direction tells us how the binder was oriented:
    # - CCW rotation needed => card tops pointed right => binder rotated
    #   90 CW in photo => reading order starts at image bottom-right
    #   (reverse traversal).
    # - CW rotation needed => card tops pointed left => binder rotated
    #   90 CCW in photo => reading order starts at image top-left
    #   (forward traversal).
    cell_rot = cv2.ROTATE_90_COUNTERCLOCKWISE  # default
    reverse_grid = True  # default for CCW (most common phone orientation)

    if page_is_landscape and ref_contours:
        # Use a reference card from contour detection to determine the
        # correct rotation direction via pixel correlation.
        ref_card = _perspective_crop(image, ref_contours[0].astype(np.float32))
        M_ref = cv2.moments(ref_contours[0])
        if M_ref["m00"] != 0:
            ref_cx = M_ref["m10"] / M_ref["m00"]
            ref_cy = M_ref["m01"] / M_ref["m00"]

            # Find which image grid cell the reference card falls in
            def _find_cell_index(val, bounds):
                for idx, (s, e) in enumerate(bounds):
                    if s <= val < e:
                        return idx
                return len(bounds) - 1

            ref_ic = _find_cell_index(ref_cx - bx, col_bounds)
            ref_ir = _find_cell_index(ref_cy - by, row_bounds)

            # Extract that cell with padding
            ry1, ry2 = row_bounds[ref_ir]
            rx1, rx2 = col_bounds[ref_ic]
            _cell_h_px = ry2 - ry1
            _cell_w_px = rx2 - rx1
            _pad_y = int(_cell_h_px * pad_frac)
            _pad_x = int(_cell_w_px * pad_frac)
            ref_cell = rectified[ry1 + _pad_y:ry2 - _pad_y,
                                 rx1 + _pad_x:rx2 - _pad_x]

            # Determine best rotation by correlation with reference card
            cell_rot = _detect_cell_rotation(ref_cell, ref_card)

            # Derive traversal order from rotation direction
            if cell_rot == cv2.ROTATE_90_COUNTERCLOCKWISE:
                reverse_grid = True   # binder rotated 90 CW in photo
            else:
                reverse_grid = False  # binder rotated 90 CCW in photo

            rot_name = "CW" if cell_rot == cv2.ROTATE_90_CLOCKWISE else "CCW"
            logger.info("Grid fallback: landscape calibration: rotation=%s reverse=%s",
                         rot_name, reverse_grid)

    # Extract grid cells in binder reading order.
    #
    # Padding strategy: valley positions (after refinement) sit at gutter
    # centres.  At page-edge boundaries (first/last row/col) we pad inward
    # by pad_frac to skip the binder frame.  At interior valley boundaries
    # we use zero padding -- the valley *is* the gutter centre, so no
    # additional inset is needed and any inset risks clipping card content
    # (especially after 90-degree rotation on landscape pages).
    cards = []
    for br in range(rows):
        for bc in range(cols):
            if page_is_landscape:
                if reverse_grid:
                    ir = cols - 1 - bc
                    ic = rows - 1 - br
                else:
                    ir = bc
                    ic = br
            else:
                ir, ic = br, bc

            ry1, ry2 = row_bounds[ir]
            rx1, rx2 = col_bounds[ic]
            cell_h_px = ry2 - ry1
            cell_w_px = rx2 - rx1

            # Full padding at page edges, small interior padding to prevent
            # cell bleed (adjacent card text appearing in segments)
            interior_frac = pad_frac * 0.4  # ~1.4% of cell at interior boundaries
            pad_y_top = int(cell_h_px * pad_frac) if ir == 0 else int(cell_h_px * interior_frac)
            pad_y_bot = int(cell_h_px * pad_frac) if ir == len(row_bounds) - 1 else int(cell_h_px * interior_frac)
            pad_x_left = int(cell_w_px * pad_frac) if ic == 0 else int(cell_w_px * interior_frac)
            pad_x_right = int(cell_w_px * pad_frac) if ic == len(col_bounds) - 1 else int(cell_w_px * interior_frac)

            y1 = ry1 + pad_y_top
            y2 = ry2 - pad_y_bot
            x1 = rx1 + pad_x_left
            x2 = rx2 - pad_x_right
            cell = rectified[y1:y2, x1:x2]

            # Ensure portrait orientation: if cell is landscape (wider than
            # tall), rotate 90 degrees using the calibrated direction.
            ch, cw = cell.shape[:2]
            if cw > ch:
                cell = cv2.rotate(cell, cell_rot)

            card = cv2.resize(cell, (CARD_OUTPUT_W, CARD_OUTPUT_H),
                              interpolation=cv2.INTER_AREA)

            cards.append(card)

    logger.info("Grid fallback: extracted %d cards (%dx%d grid, perspective=%s)",
                len(cards), rows, cols, perspective_warped)
    return cards


def segment_cards(
    image_path: str | Path,
    output_dir: Optional[str | Path] = None,
    max_cards: int = 18,
    output_format: str = "png",
    expected_grid: Optional[tuple[int, int]] = None,
) -> list[Path]:
    """Detect and extract individual cards from a binder page photo.

    Takes a photo of a binder page (typically 3x3 or 3x3x2 grid of cards in
    plastic sleeves) and returns paths to cropped, perspective-corrected
    images of each detected card.

    Args:
        image_path: Path to the binder page photo.
        output_dir: Directory to save cropped card images. Defaults to
            a sibling directory named ``{stem}_cards/`` next to the input.
        max_cards: Maximum number of cards to extract (safety limit).
        output_format: Image format for output files (png, jpg).

    Returns:
        List of Path objects pointing to the saved card images, sorted in
        reading order (left-to-right, top-to-bottom).

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be read by OpenCV.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize very large images to speed up processing (keep aspect ratio)
    # 4500 preserves most of a 4032x3024 phone photo's resolution
    h, w = image.shape[:2]
    max_dim = 4500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        logger.info("Resized image from %dx%d to %dx%d", w, h,
                     image.shape[1], image.shape[0])

    # Set up output directory
    if output_dir is None:
        output_dir = image_path.parent / f"{image_path.stem}_cards"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine expected grid size
    if expected_grid is not None:
        expected_rows, expected_cols = expected_grid
        expected_count = expected_rows * expected_cols
    else:
        expected_rows, expected_cols = 3, 3  # standard binder page
        expected_count = 9

    # Detect card contours
    contours = _find_card_contours(image, expected_count=expected_count)

    use_grid_fallback = False
    if not contours:
        logger.warning("No cards detected via contours in %s, trying grid fallback",
                        image_path)
        use_grid_fallback = True
    elif len(contours) < expected_count:  # any missing cards triggers fallback
        logger.info("Contour detection found %d/%d cards, using grid fallback",
                     len(contours), expected_count)
        use_grid_fallback = True
    elif len(contours) >= 3:
        # Quality check: if any contour is much smaller than median, the
        # contour detection grabbed a bad region.  Use grid instead.
        areas = [cv2.contourArea(c) for c in contours]
        median_area = sorted(areas)[len(areas) // 2]
        min_area = min(areas)
        if min_area < median_area * 0.5:
            logger.info("Contour quality check: smallest area %.0f < 50%% of "
                         "median %.0f, using grid fallback",
                         min_area, median_area)
            use_grid_fallback = True

    if use_grid_fallback:
        # Try contour-guided grid first (uses detected card positions)
        card_images = None
        if contours and len(contours) >= 6:
            card_images = _contour_guided_grid(
                image, contours, expected_rows, expected_cols)
            if card_images:
                logger.info("Using contour-guided grid (%d contours)",
                            len(contours))

        if card_images is None:
            card_images = _grid_fallback(image, expected_rows, expected_cols,
                                          ref_contours=contours if contours else None)
        saved_paths = []
        for i, card_img in enumerate(card_images):
            out_path = output_dir / f"card_{i:02d}.{output_format}"
            cv2.imwrite(str(out_path), card_img)
            saved_paths.append(out_path)
        logger.info("Grid fallback extracted %d cards from %s -> %s",
                     len(saved_paths), image_path, output_dir)
        return saved_paths

    # Limit to max_cards (take largest by area)
    if len(contours) > max_cards:
        contours.sort(key=cv2.contourArea, reverse=True)
        contours = contours[:max_cards]
        logger.info("Limited to %d largest cards", max_cards)

    # Sort into grid reading order
    contours = _sort_grid(contours)

    # Detect page-level landscape orientation.
    # When the page image is landscape (wider than tall), all cards should be
    # landscape quads.  If a contour's measured dimensions happen to be
    # portrait-shaped (e.g. due to a partial or skewed detection), we still
    # need to apply the landscape->portrait rotation so the card content
    # comes out upright.
    page_h, page_w = image.shape[:2]
    page_is_landscape = page_w > page_h
    if page_is_landscape and len(contours) >= 3:
        # Count how many contours are landscape-shaped
        n_landscape = 0
        for cnt in contours:
            pts = _order_points(cnt.reshape(4, 2).astype(np.float32))
            qw = (np.linalg.norm(pts[1] - pts[0]) +
                  np.linalg.norm(pts[2] - pts[3])) / 2
            qh = (np.linalg.norm(pts[3] - pts[0]) +
                  np.linalg.norm(pts[2] - pts[1])) / 2
            if qw > qh:
                n_landscape += 1
        # If the majority of contours are landscape, force landscape
        # treatment for all cards on this page.
        force_landscape = n_landscape > len(contours) // 2
        if force_landscape:
            logger.info("Page is landscape with %d/%d landscape contours; "
                        "forcing landscape rotation for all cards",
                        n_landscape, len(contours))
    else:
        force_landscape = False

    # Extract and save each card
    saved_paths = []
    for i, cnt in enumerate(contours):
        card_img = _perspective_crop(image, cnt.astype(np.float32),
                                     force_landscape=force_landscape)
        out_path = output_dir / f"card_{i:02d}.{output_format}"
        cv2.imwrite(str(out_path), card_img)
        saved_paths.append(out_path)
        logger.debug("Saved card %d to %s", i, out_path)

    logger.info("Extracted %d cards from %s -> %s",
                len(saved_paths), image_path, output_dir)
    return saved_paths


def segment_cards_debug(
    image_path: str | Path,
    output_dir: Optional[str | Path] = None,
) -> Path:
    """Run segmentation with visual debug output.

    Saves an annotated copy of the input image showing detected contours,
    grid order numbers, and bounding boxes. Useful for tuning parameters.

    Args:
        image_path: Path to the binder page photo.
        output_dir: Directory for debug output. Defaults to same as segment_cards.

    Returns:
        Path to the annotated debug image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]
    max_dim = 4500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)

    if output_dir is None:
        output_dir = image_path.parent / f"{image_path.stem}_cards"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contours = _find_card_contours(image)
    contours = _sort_grid(contours)

    debug_img = image.copy()
    for i, cnt in enumerate(contours):
        # Draw contour in green
        cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 3)
        # Label with index
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = int(cnt[:, 0, 0].mean()), int(cnt[:, 0, 1].mean())
        cv2.putText(debug_img, str(i), (cx - 15, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

    debug_path = output_dir / "debug_contours.jpg"
    cv2.imwrite(str(debug_path), debug_img)
    logger.info("Debug image saved to %s", debug_path)
    return debug_path
