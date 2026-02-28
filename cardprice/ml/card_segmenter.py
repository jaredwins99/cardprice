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
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard Pokemon card dimensions (mm): 63 x 88 -> aspect ratio ~0.716
CARD_ASPECT_RATIO = 63.0 / 88.0  # width / height = ~0.716
ASPECT_RATIO_TOLERANCE = 0.25     # allow 25% deviation

# Output card image size (pixels) -- 2.5x standard 63x88mm at ~10px/mm
CARD_OUTPUT_W = 630
CARD_OUTPUT_H = 880


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
                      output_h: int = CARD_OUTPUT_H) -> np.ndarray:
    """Apply a four-point perspective transform to extract a card.

    Args:
        image: Source image (BGR).
        pts: 4 corner points of the card quadrilateral.
        output_w: Width of the output image.
        output_h: Height of the output image.

    Returns:
        Warped rectangular card image.
    """
    ordered = _order_points(pts.reshape(4, 2))
    dst = np.array([
        [0, 0],
        [output_w - 1, 0],
        [output_w - 1, output_h - 1],
        [0, output_h - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, M, (output_w, output_h))


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
    for eps_mult in (0.02, 0.04, 0.06, 0.08):
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


def _find_card_contours(image: np.ndarray) -> list[np.ndarray]:
    """Detect card-shaped contours in an image using multiple strategies.

    Tries several preprocessing approaches and merges results to handle
    varied lighting conditions typical of binder page photos.

    Returns:
        List of 4-point contour arrays, one per detected card.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    image_area = h * w

    candidates = {}  # keyed by center to deduplicate

    def _add_candidates(edges: np.ndarray):
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if not _is_card_shaped(cnt, image_area):
                continue
            peri = cv2.arcLength(cnt, True)
            approx = None
            for eps_mult in (0.02, 0.04, 0.06, 0.08):
                candidate = cv2.approxPolyDP(cnt, eps_mult * peri, True)
                if len(candidate) == 4:
                    approx = candidate
                    break
            if approx is None:
                continue
            # Deduplicate by center position (within 30px)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            grid_key = (cx // 30, cy // 30)
            area = cv2.contourArea(cnt)
            if grid_key not in candidates or area > cv2.contourArea(candidates[grid_key]):
                candidates[grid_key] = approx

    # Strategy 1: Canny edge detection (good for clear edges)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges_canny = cv2.Canny(blurred, 30, 100)
    # Dilate to close small gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_canny = cv2.dilate(edges_canny, kernel, iterations=1)
    _add_candidates(edges_canny)

    # Strategy 2: Adaptive thresholding (robust to uneven lighting)
    thresh_adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 5
    )
    # Morphological close to fill gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh_adapt = cv2.morphologyEx(thresh_adapt, cv2.MORPH_CLOSE, kernel_close)
    _add_candidates(thresh_adapt)

    # Strategy 3: Otsu thresholding (simple global threshold)
    _, thresh_otsu = cv2.threshold(blurred, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel_close)
    _add_candidates(thresh_otsu)

    # Strategy 4: Bilateral filter + Canny (preserves edges, smooths textures)
    bilateral = cv2.bilateralFilter(gray, 11, 75, 75)
    edges_bilateral = cv2.Canny(bilateral, 20, 80)
    edges_bilateral = cv2.dilate(edges_bilateral, kernel, iterations=1)
    _add_candidates(edges_bilateral)

    result = list(candidates.values())
    logger.info("Found %d card-shaped contours", len(result))
    return result


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


def segment_cards(
    image_path: str | Path,
    output_dir: Optional[str | Path] = None,
    max_cards: int = 18,
    output_format: str = "png",
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
    h, w = image.shape[:2]
    max_dim = 3000
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

    # Detect card contours
    contours = _find_card_contours(image)

    if not contours:
        logger.warning("No cards detected in %s", image_path)
        return []

    # Limit to max_cards (take largest by area)
    if len(contours) > max_cards:
        contours.sort(key=cv2.contourArea, reverse=True)
        contours = contours[:max_cards]
        logger.info("Limited to %d largest cards", max_cards)

    # Sort into grid reading order
    contours = _sort_grid(contours)

    # Extract and save each card
    saved_paths = []
    for i, cnt in enumerate(contours):
        card_img = _perspective_crop(image, cnt.astype(np.float32))
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
    max_dim = 3000
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
