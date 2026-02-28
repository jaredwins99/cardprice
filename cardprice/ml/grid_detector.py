"""Grid detector for binder page images.

Detects a regular grid of card slots in a binder page photo and returns
crop coordinates for each cell.  Binder pages have a predictable layout
(typically 3x3 for standard 9-pocket pages) and cards have a known
aspect ratio (2.5" x 3.5", i.e. 5:7).

Three detection strategies, tried in order:
1. Hough line detection -- find horizontal/vertical lines, cluster them,
   derive grid cells from intersections.
2. Contour-based -- find the binder page rectangle, perspective-correct,
   subdivide evenly.
3. Uniform subdivision -- assume the page fills (most of) the image and
   divide into equal cells.

Usage:
    from cardprice.ml.grid_detector import detect_grid
    cells = detect_grid("binder_page.jpg", rows=3, cols=3)
    # cells is a list of (x, y, w, h) tuples in row-major order
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Pokemon card aspect ratio: 2.5" x 3.5" = 5:7
CARD_ASPECT = 5.0 / 7.0  # width / height ~ 0.714


@dataclass
class GridResult:
    """Result of grid detection."""

    cells: List[Tuple[int, int, int, int]]  # (x, y, w, h) per cell, row-major
    rows: int
    cols: int
    method: str  # "hough", "contour", "uniform"
    page_corners: Optional[np.ndarray] = None  # 4x2 corners if detected
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_grid(
    image_path: str,
    rows: int = 3,
    cols: int = 3,
    *,
    padding_pct: float = 0.02,
    min_page_area_pct: float = 0.15,
) -> GridResult:
    """Detect a grid of card slots in a binder page image.

    Parameters
    ----------
    image_path : str
        Path to the binder page image.
    rows, cols : int
        Expected grid dimensions (default 3x3 for standard 9-pocket).
    padding_pct : float
        Fraction of cell size to trim from each edge (removes binder seams).
    min_page_area_pct : float
        Minimum area (as fraction of image) for a contour to be considered
        the binder page.

    Returns
    -------
    GridResult with a list of (x, y, w, h) crop rectangles in row-major order.
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    logger.info("Grid detection on %s (%dx%d), expecting %dx%d grid",
                image_path, w, h, rows, cols)

    # Strategy 1: Hough line detection
    result = _detect_via_hough(img, rows, cols, padding_pct)
    if result is not None:
        logger.info("Hough detection succeeded (confidence=%.2f)", result.confidence)
        return result

    # Strategy 2: Contour-based page detection + subdivision
    result = _detect_via_contour(img, rows, cols, padding_pct, min_page_area_pct)
    if result is not None:
        logger.info("Contour detection succeeded (confidence=%.2f)", result.confidence)
        return result

    # Strategy 3: Uniform subdivision (always succeeds)
    logger.info("Falling back to uniform subdivision")
    return _uniform_subdivision(img, rows, cols, padding_pct)


def crop_cells(image_path: str, grid: GridResult) -> List[np.ndarray]:
    """Crop individual cells from the image using a GridResult.

    Returns a list of BGR numpy arrays, one per cell in row-major order.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    crops = []
    for x, y, cw, ch in grid.cells:
        crop = img[y : y + ch, x : x + cw]
        crops.append(crop)
    return crops


# ---------------------------------------------------------------------------
# Strategy 1: Hough line detection
# ---------------------------------------------------------------------------

def _detect_via_hough(
    img: np.ndarray,
    rows: int,
    cols: int,
    padding_pct: float,
) -> Optional[GridResult]:
    """Detect grid lines using Hough transform, cluster them, derive cells."""
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to handle uneven lighting (common in phone photos)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # Dilate edges slightly to close small gaps in binder seam lines
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Detect lines -- use standard Hough for full-length lines
    # rho=1 pixel, theta=1 degree, threshold scales with image size
    threshold = max(w, h) // 4
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=threshold)

    if lines is None or len(lines) < (rows + cols - 2):
        logger.debug("Hough: not enough lines detected (%s)",
                      len(lines) if lines is not None else 0)
        return None

    # Separate into horizontal and vertical lines
    h_lines: List[float] = []  # y-intercepts of horizontal lines
    v_lines: List[float] = []  # x-intercepts of vertical lines

    for line in lines:
        rho, theta = line[0]
        # Horizontal: theta near pi/2 (90 degrees)
        if abs(theta - np.pi / 2) < np.pi / 12:  # within 15 degrees
            y_intercept = abs(rho / np.sin(theta)) if np.sin(theta) != 0 else abs(rho)
            h_lines.append(y_intercept)
        # Vertical: theta near 0 or pi
        elif theta < np.pi / 12 or theta > np.pi - np.pi / 12:
            x_intercept = abs(rho / np.cos(theta)) if np.cos(theta) != 0 else abs(rho)
            v_lines.append(x_intercept)

    logger.debug("Hough: %d horizontal, %d vertical candidates", len(h_lines), len(v_lines))

    # We need rows+1 horizontal lines and cols+1 vertical lines to define the grid
    h_clusters = _cluster_lines(sorted(h_lines), expected=rows + 1, span=h)
    v_clusters = _cluster_lines(sorted(v_lines), expected=cols + 1, span=w)

    if h_clusters is None or v_clusters is None:
        logger.debug("Hough: clustering failed (h=%s, v=%s)",
                      h_clusters is not None, v_clusters is not None)
        return None

    # Validate: lines should be roughly evenly spaced
    if not _check_even_spacing(h_clusters, tolerance=0.35):
        logger.debug("Hough: horizontal lines not evenly spaced: %s", h_clusters)
        return None
    if not _check_even_spacing(v_clusters, tolerance=0.35):
        logger.debug("Hough: vertical lines not evenly spaced: %s", v_clusters)
        return None

    # Build cells from grid intersections
    cells = _grid_from_lines(h_clusters, v_clusters, w, h, padding_pct)

    return GridResult(
        cells=cells,
        rows=rows,
        cols=cols,
        method="hough",
        confidence=0.85,
    )


def _cluster_lines(
    positions: List[float],
    expected: int,
    span: int,
    min_gap_pct: float = 0.03,
) -> Optional[List[float]]:
    """Cluster nearby line positions into *expected* groups.

    Uses a simple merge: lines within min_gap_pct * span of each other
    are merged into one cluster (averaged position).
    """
    if len(positions) < expected:
        return None

    min_gap = span * min_gap_pct
    clusters: List[List[float]] = []
    current: List[float] = [positions[0]]

    for pos in positions[1:]:
        if pos - current[-1] < min_gap:
            current.append(pos)
        else:
            clusters.append(current)
            current = [pos]
    clusters.append(current)

    # Average each cluster
    centers = [float(np.mean(c)) for c in clusters]

    if len(centers) < expected:
        return None

    # If we have more clusters than expected, pick the *expected* most prominent
    # (the ones with the most lines, i.e. most votes)
    if len(centers) > expected:
        scored = sorted(zip(clusters, centers), key=lambda x: -len(x[0]))
        centers = sorted([c for _, c in scored[:expected]])

    return centers


def _check_even_spacing(positions: List[float], tolerance: float = 0.35) -> bool:
    """Check that positions are roughly evenly spaced."""
    if len(positions) < 3:
        return True
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    mean_gap = float(np.mean(gaps))
    if mean_gap == 0:
        return False
    return all(abs(g - mean_gap) / mean_gap < tolerance for g in gaps)


def _grid_from_lines(
    h_lines: List[float],
    v_lines: List[float],
    img_w: int,
    img_h: int,
    padding_pct: float,
) -> List[Tuple[int, int, int, int]]:
    """Build (x, y, w, h) cell rectangles from sorted grid line positions."""
    cells = []
    for r in range(len(h_lines) - 1):
        for c in range(len(v_lines) - 1):
            x1 = v_lines[c]
            x2 = v_lines[c + 1]
            y1 = h_lines[r]
            y2 = h_lines[r + 1]
            cw = x2 - x1
            ch = y2 - y1
            # Apply padding to trim binder seam artifacts
            px = cw * padding_pct
            py = ch * padding_pct
            x = int(max(0, x1 + px))
            y = int(max(0, y1 + py))
            w = int(min(img_w - x, cw - 2 * px))
            h = int(min(img_h - y, ch - 2 * py))
            cells.append((x, y, max(1, w), max(1, h)))
    return cells


# ---------------------------------------------------------------------------
# Strategy 2: Contour-based page detection
# ---------------------------------------------------------------------------

def _detect_via_contour(
    img: np.ndarray,
    rows: int,
    cols: int,
    padding_pct: float,
    min_page_area_pct: float,
) -> Optional[GridResult]:
    """Find the binder page rectangle, perspective-correct, subdivide."""
    h, w = img.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # Use adaptive thresholding for better performance under varied lighting
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )
    # Close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.debug("Contour: no contours found")
        return None

    # Find the largest contour that could be the binder page
    best_quad = None
    best_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * min_page_area_pct:
            continue
        # Approximate to polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and area > best_area:
            best_quad = approx.reshape(4, 2).astype(np.float32)
            best_area = area

    if best_quad is None:
        # Try the largest contour's bounding rect as fallback
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < img_area * min_page_area_pct:
            logger.debug("Contour: no large enough contour found")
            return None
        rect = cv2.minAreaRect(largest)
        best_quad = cv2.boxPoints(rect).astype(np.float32)
        best_area = area

    # Order corners: top-left, top-right, bottom-right, bottom-left
    corners = _order_corners(best_quad)

    # Compute target dimensions preserving card aspect ratio
    # Page with cols cards wide and rows cards tall
    target_w, target_h = _compute_page_dimensions(corners, rows, cols)

    dst = np.array(
        [[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]],
        dtype=np.float32,
    )

    # Perspective transform to get a rectified page
    M = cv2.getPerspectiveTransform(corners, dst)

    # Map the cells back to original image coordinates (inverse transform)
    # Build cell rects on the rectified page, then inverse-transform corners
    M_inv = cv2.getPerspectiveTransform(dst, corners)

    cell_w = target_w / cols
    cell_h = target_h / rows
    px = cell_w * padding_pct
    py = cell_h * padding_pct

    cells = []
    for r in range(rows):
        for c in range(cols):
            # Cell corners in rectified space (with padding)
            x1 = c * cell_w + px
            y1 = r * cell_h + py
            x2 = (c + 1) * cell_w - px
            y2 = (r + 1) * cell_h - py

            # Transform the four cell corners back to original image space
            rect_pts = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
            ).reshape(-1, 1, 2)
            orig_pts = cv2.perspectiveTransform(rect_pts, M_inv).reshape(4, 2)

            # Axis-aligned bounding box in original image
            bx = int(max(0, orig_pts[:, 0].min()))
            by = int(max(0, orig_pts[:, 1].min()))
            bx2 = int(min(w, orig_pts[:, 0].max()))
            by2 = int(min(h, orig_pts[:, 1].max()))
            cells.append((bx, by, max(1, bx2 - bx), max(1, by2 - by)))

    return GridResult(
        cells=cells,
        rows=rows,
        cols=cols,
        method="contour",
        page_corners=corners,
        confidence=0.70,
    )


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    # Sum of coords: smallest = top-left, largest = bottom-right
    # Diff of coords: smallest = top-right, largest = bottom-left
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _compute_page_dimensions(
    corners: np.ndarray,
    rows: int,
    cols: int,
    target_short: int = 900,
) -> Tuple[int, int]:
    """Compute rectified page dimensions that preserve card aspect ratio.

    A page of cols x rows cards should have aspect ratio:
        page_w / page_h = (cols * card_w) / (rows * card_h)
                        = (cols * 5) / (rows * 7)
    """
    page_aspect = (cols * 5.0) / (rows * 7.0)
    if page_aspect >= 1.0:
        target_w = int(target_short * page_aspect)
        target_h = target_short
    else:
        target_w = target_short
        target_h = int(target_short / page_aspect)
    return target_w, target_h


# ---------------------------------------------------------------------------
# Strategy 3: Uniform subdivision (fallback, always succeeds)
# ---------------------------------------------------------------------------

def _uniform_subdivision(
    img: np.ndarray,
    rows: int,
    cols: int,
    padding_pct: float,
) -> GridResult:
    """Simply divide the image into equal cells."""
    h, w = img.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    px = cell_w * padding_pct
    py = cell_h * padding_pct

    cells = []
    for r in range(rows):
        for c in range(cols):
            x = int(c * cell_w + px)
            y = int(r * cell_h + py)
            cw = int(cell_w - 2 * px)
            ch = int(cell_h - 2 * py)
            # Clamp to image bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            cw = max(1, min(cw, w - x))
            ch = max(1, min(ch, h - y))
            cells.append((x, y, cw, ch))

    return GridResult(
        cells=cells,
        rows=rows,
        cols=cols,
        method="uniform",
        confidence=0.40,
    )
