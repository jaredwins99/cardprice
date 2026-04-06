"""Card centering measurement for condition grading.

Detects the printed border widths on a Pokemon card image and calculates
centering ratios used by PSA, BGS, and CGC grading scales.

How it works:
  1. Convert to grayscale, apply adaptive thresholding + Canny edges
  2. Detect card outer boundary via contour detection (largest quadrilateral)
  3. Detect the inner printed frame via a second contour pass on the
     interior region (the artwork/text boundary)
  4. Measure the gap between outer edge and inner frame on all four sides
  5. Compute centering ratios (smaller / larger * 100 for each axis)

Centering grades (PSA scale):
  - PSA 10 (Gem Mint):  55/45 or better on front, 75/25 or better on back
  - PSA 9 (Mint):       60/40 or better on front, 75/25 or better on back
  - PSA 8 (NM-MT):      65/35 or better
  - PSA 7 (NM):         70/30 or better

The PSA centering formula:
  ratio = smaller_border / larger_border * 100
  displayed as "smaller/larger" e.g. "55/45"

Physical card dimensions (standard Pokemon):
  - 63mm x 88mm (2.5" x 3.5")
  - At 1290 DPI: ~3225 x 4515 pixels

Pure OpenCV -- no ML models required.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard Pokemon card dimensions in mm
CARD_WIDTH_MM = 63.0
CARD_HEIGHT_MM = 88.0

# Default scan DPI (high-quality flatbed)
DEFAULT_DPI = 1290

# PSA centering thresholds: (max ratio of smaller/larger as percentage)
# e.g. PSA 10 requires the smaller border to be at least 45% of the
# larger border, i.e. no worse than 55/45.
PSA_THRESHOLDS = {
    10: {"front_pct": 45.0, "back_pct": 25.0, "label": "Gem Mint"},
    9:  {"front_pct": 40.0, "back_pct": 25.0, "label": "Mint"},
    8:  {"front_pct": 35.0, "back_pct": None, "label": "NM-MT"},
    7:  {"front_pct": 30.0, "back_pct": None, "label": "NM"},
}


@dataclass
class BorderMeasurement:
    """Border width measurements for one side of a card."""
    pixels: float = 0.0
    mm: float = 0.0


@dataclass
class CenteringResult:
    """Full centering analysis result."""

    # Border widths
    top: BorderMeasurement = field(default_factory=BorderMeasurement)
    bottom: BorderMeasurement = field(default_factory=BorderMeasurement)
    left: BorderMeasurement = field(default_factory=BorderMeasurement)
    right: BorderMeasurement = field(default_factory=BorderMeasurement)

    # Centering ratios: displayed as "smaller/larger"
    # e.g. lr_ratio=(45, 55) means 45% left, 55% right
    lr_ratio: Tuple[float, float] = (50.0, 50.0)
    tb_ratio: Tuple[float, float] = (50.0, 50.0)

    # PSA centering percentage (smaller/larger * 100)
    # Perfect = 100 (50/50), worst = 0 (100/0)
    lr_pct: float = 100.0
    tb_pct: float = 100.0

    # Overall centering grade (PSA scale, front face)
    psa_grade: Optional[int] = None
    psa_label: str = ""

    # Whether detection succeeded
    success: bool = False
    error: str = ""

    # Debug info
    outer_contour: Optional[np.ndarray] = None
    inner_contour: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Human-readable summary string."""
        if not self.success:
            return f"Centering detection failed: {self.error}"

        lines = [
            f"Borders (px): L={self.left.pixels:.1f}  R={self.right.pixels:.1f}"
            f"  T={self.top.pixels:.1f}  B={self.bottom.pixels:.1f}",
            f"Borders (mm): L={self.left.mm:.2f}  R={self.right.mm:.2f}"
            f"  T={self.top.mm:.2f}  B={self.bottom.mm:.2f}",
            f"L/R centering: {self.lr_ratio[0]:.1f}/{self.lr_ratio[1]:.1f}",
            f"T/B centering: {self.tb_ratio[0]:.1f}/{self.tb_ratio[1]:.1f}",
        ]
        if self.psa_grade is not None:
            lines.append(
                f"PSA centering grade: {self.psa_grade} ({self.psa_label})"
            )
        else:
            lines.append("PSA centering grade: below 7")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------

def _pixels_to_mm(pixels: float, dpi: float) -> float:
    """Convert pixel measurement to millimeters given DPI."""
    return pixels / dpi * 25.4


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as [top-left, top-right, bottom-right, bottom-left].

    Uses the sum and difference of coordinates to determine corner positions.
    This is robust to slight rotation.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    ordered[0] = pts[np.argmin(s)]   # top-left: smallest x+y
    ordered[2] = pts[np.argmax(s)]   # bottom-right: largest x+y
    ordered[1] = pts[np.argmin(d)]   # top-right: smallest x-y
    ordered[3] = pts[np.argmax(d)]   # bottom-left: largest x-y

    return ordered


def _deskew_image(img: np.ndarray, quad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perspective-warp a quadrilateral region to a rectangle.

    Parameters
    ----------
    img : np.ndarray
        Source image.
    quad : np.ndarray
        Four corner points of the region [TL, TR, BR, BL].

    Returns
    -------
    (warped_image, transform_matrix)
    """
    ordered = _order_quad_points(quad)

    # Compute output dimensions from the quad
    w1 = np.linalg.norm(ordered[1] - ordered[0])
    w2 = np.linalg.norm(ordered[2] - ordered[3])
    h1 = np.linalg.norm(ordered[3] - ordered[0])
    h2 = np.linalg.norm(ordered[2] - ordered[1])

    out_w = int(max(w1, w2))
    out_h = int(max(h1, h2))

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h),
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped, M


def _find_largest_quad(contours, img_area: float,
                       min_area_frac: float = 0.3) -> Optional[np.ndarray]:
    """Find the largest approximately-quadrilateral contour.

    Parameters
    ----------
    contours : list
        Contours from cv2.findContours.
    img_area : float
        Total image area (for filtering tiny contours).
    min_area_frac : float
        Minimum contour area as fraction of image area.

    Returns
    -------
    np.ndarray or None
        Four corner points, or None if no suitable quad found.
    """
    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * min_area_frac:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and area > best_area:
            best = approx
            best_area = area
        elif len(approx) > 4 and area > best_area:
            # Try with looser approximation for rounded corners
            approx2 = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            if len(approx2) == 4:
                best = approx2
                best_area = area

    if best is not None:
        return best.reshape(4, 2).astype(np.float32)
    return None


def _find_card_boundary(gray: np.ndarray) -> Optional[np.ndarray]:
    """Detect the outer card boundary using edge detection and contours.

    Tries multiple strategies to handle different background colors and
    scan qualities.

    Returns four corner points [TL, TR, BR, BL] or None.
    """
    h, w = gray.shape[:2]
    img_area = h * w

    strategies = []

    # Strategy 1: Canny on blurred image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    strategies.append(("canny", edges))

    # Strategy 2: Adaptive threshold (handles varied backgrounds)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )
    strategies.append(("adaptive", adaptive))

    # Strategy 3: Otsu threshold (works well on high-contrast scans)
    _, otsu = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    strategies.append(("otsu", otsu))

    for name, binary in strategies:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        quad = _find_largest_quad(contours, img_area, min_area_frac=0.5)
        if quad is not None:
            logger.debug("Card boundary found via %s strategy", name)
            return quad

    # Fallback: assume the entire image IS the card (common for cropped scans)
    logger.debug("Card boundary not found; assuming full image is card")
    margin = 2
    return np.array([
        [margin, margin],
        [w - margin, margin],
        [w - margin, h - margin],
        [margin, h - margin],
    ], dtype=np.float32)


def _find_inner_frame(card_img: np.ndarray,
                      border_search_frac: float = 0.15) -> dict:
    """Detect the inner printed frame boundary on a deskewed card image.

    The inner frame is the edge of the artwork/text box area. The gap
    between the card edge and this inner frame is the "border" that
    centering measures.

    Uses edge detection on each border strip independently for robustness
    against cards with different border colors on different sides.

    Parameters
    ----------
    card_img : np.ndarray
        Grayscale, deskewed, portrait-oriented card image.
    border_search_frac : float
        How far into the card to search for the inner edge (fraction).
        Default 0.15 = search the outer 15% of each side.

    Returns
    -------
    dict with keys 'top', 'bottom', 'left', 'right' (pixel offsets from
    the respective card edge to the detected inner frame edge).
    """
    h, w = card_img.shape[:2]
    search_px_h = int(h * border_search_frac)
    search_px_w = int(w * border_search_frac)

    # Ensure minimum search area
    search_px_h = max(search_px_h, 20)
    search_px_w = max(search_px_w, 20)

    borders = {}

    # For each side, extract the border strip and find the strongest
    # horizontal or vertical gradient (the inner frame edge).

    # --- Top border ---
    top_strip = card_img[0:search_px_h, int(w * 0.15):int(w * 0.85)]
    borders["top"] = _find_edge_in_strip(top_strip, axis="horizontal")

    # --- Bottom border ---
    bot_strip = card_img[h - search_px_h:h, int(w * 0.15):int(w * 0.85)]
    bot_edge = _find_edge_in_strip(bot_strip, axis="horizontal",
                                    from_end=True)
    borders["bottom"] = bot_edge

    # --- Left border ---
    left_strip = card_img[int(h * 0.15):int(h * 0.85), 0:search_px_w]
    borders["left"] = _find_edge_in_strip(left_strip, axis="vertical")

    # --- Right border ---
    right_strip = card_img[int(h * 0.15):int(h * 0.85),
                           w - search_px_w:w]
    right_edge = _find_edge_in_strip(right_strip, axis="vertical",
                                      from_end=True)
    borders["right"] = right_edge

    logger.debug(
        "Inner frame borders (px): T=%.1f B=%.1f L=%.1f R=%.1f",
        borders["top"], borders["bottom"],
        borders["left"], borders["right"],
    )
    return borders


def _find_edge_in_strip(strip: np.ndarray, axis: str = "horizontal",
                         from_end: bool = False) -> float:
    """Find the strongest edge transition in a border strip.

    Collapses the strip along one axis to produce a 1-D intensity profile,
    then finds the location of the maximum gradient (the inner frame edge).

    Parameters
    ----------
    strip : np.ndarray
        Grayscale image strip from the border region.
    axis : str
        "horizontal" for top/bottom borders (collapse columns, scan rows),
        "vertical" for left/right borders (collapse rows, scan columns).
    from_end : bool
        If True, search from the far edge inward (for bottom/right borders).

    Returns
    -------
    float
        Distance in pixels from the card edge to the inner frame edge.
    """
    if strip.size == 0:
        return 0.0

    # Apply gentle blur to reduce noise
    strip = cv2.GaussianBlur(strip, (3, 3), 0)

    if axis == "horizontal":
        # Collapse along columns -> 1-D profile along rows (top to bottom)
        profile = strip.mean(axis=1).astype(np.float64)
    else:
        # Collapse along rows -> 1-D profile along columns (left to right)
        profile = strip.mean(axis=0).astype(np.float64)

    if len(profile) < 5:
        return 0.0

    if from_end:
        profile = profile[::-1]

    # Compute gradient magnitude (absolute first derivative)
    gradient = np.abs(np.diff(profile))

    # Smooth the gradient to avoid noise spikes
    kernel_size = max(3, len(gradient) // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1
    gradient_smooth = cv2.GaussianBlur(
        gradient.reshape(1, -1).astype(np.float32),
        (kernel_size, 1), 0
    ).ravel()

    # Find the peak gradient location (this is the inner frame edge)
    # Skip the first few pixels (card edge artifacts)
    skip = max(3, len(gradient_smooth) // 15)
    search_region = gradient_smooth[skip:]

    if len(search_region) == 0:
        return 0.0

    edge_pos = skip + np.argmax(search_region)

    # Subpixel refinement: fit a parabola around the peak
    if 1 <= edge_pos < len(gradient_smooth) - 1:
        y0 = float(gradient_smooth[edge_pos - 1])
        y1 = float(gradient_smooth[edge_pos])
        y2 = float(gradient_smooth[edge_pos + 1])
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        if abs(denom) > 1e-6:
            offset = (y0 - y2) / denom
            edge_pos = edge_pos + offset

    return float(edge_pos)


# ---------------------------------------------------------------------------
# Centering ratio computation
# ---------------------------------------------------------------------------

def _compute_ratio(border_a: float, border_b: float) -> Tuple[float, float, float]:
    """Compute centering ratio for two opposing borders.

    Parameters
    ----------
    border_a : float
        Width of one border (e.g. left or top) in pixels.
    border_b : float
        Width of the opposing border (e.g. right or bottom) in pixels.

    Returns
    -------
    (ratio_a, ratio_b, psa_pct)
        ratio_a, ratio_b: percentage split (sum to 100).
        psa_pct: PSA centering percentage = smaller/larger * 100.
            Perfect centering = 100, worst = 0.
    """
    total = border_a + border_b
    if total < 1e-6:
        return 50.0, 50.0, 100.0

    pct_a = border_a / total * 100.0
    pct_b = border_b / total * 100.0

    smaller = min(border_a, border_b)
    larger = max(border_a, border_b)
    psa_pct = smaller / larger * 100.0 if larger > 1e-6 else 100.0

    return pct_a, pct_b, psa_pct


def _assign_psa_grade(lr_pct: float, tb_pct: float,
                      is_back: bool = False) -> Tuple[Optional[int], str]:
    """Determine PSA centering grade from centering percentages.

    Parameters
    ----------
    lr_pct : float
        Left/right PSA percentage (smaller/larger * 100).
    tb_pct : float
        Top/bottom PSA percentage (smaller/larger * 100).
    is_back : bool
        If True, use back-face thresholds (more lenient).

    Returns
    -------
    (grade, label) or (None, "") if below PSA 7.
    """
    # The overall centering is limited by the worse axis
    worse_pct = min(lr_pct, tb_pct)

    for grade in (10, 9, 8, 7):
        threshold_key = "back_pct" if is_back else "front_pct"
        threshold = PSA_THRESHOLDS[grade][threshold_key]
        if threshold is None:
            # No specific threshold for this grade on this face;
            # use the front threshold as fallback
            threshold = PSA_THRESHOLDS[grade]["front_pct"]

        # PSA centering formula: the ratio (displayed as smaller/larger)
        # must be at least threshold/(100-threshold).
        # Since worse_pct = smaller/larger * 100, we need:
        #   worse_pct >= threshold / (100 - threshold) * 100
        # Simplify: for 55/45 threshold, the PSA % must be >= 45/55*100 = 81.8%
        min_psa_pct = threshold / (100.0 - threshold) * 100.0
        if worse_pct >= min_psa_pct:
            return grade, PSA_THRESHOLDS[grade]["label"]

    return None, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure_centering(
    image_path: str,
    dpi: float = DEFAULT_DPI,
    is_back: bool = False,
) -> CenteringResult:
    """Measure card centering from a high-resolution scan.

    Takes a single card face image (front or back) and measures the border
    widths on all four sides.  Returns centering ratios and a PSA-scale
    centering grade.

    The input should be a clean, high-resolution scan (~1290 DPI) of a
    single card.  The card can be slightly rotated -- the algorithm will
    detect and correct for rotation via perspective transform.

    Parameters
    ----------
    image_path : str or Path
        Path to the card scan image.
    dpi : float
        Scanner DPI.  Used to convert pixel measurements to millimeters.
        Default is 1290 (a common high-quality scan setting).
    is_back : bool
        If True, use PSA back-face centering thresholds (more lenient:
        75/25 for grades 9 and 10).

    Returns
    -------
    CenteringResult
        Dataclass with border measurements, ratios, and PSA grade.
    """
    image_path = str(image_path)
    result = CenteringResult()

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        result.error = f"Could not read image: {image_path}"
        logger.warning(result.error)
        return result

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    logger.debug("Centering analysis: %s (%dx%d, %.0f DPI)",
                 Path(image_path).name, w, h, dpi)

    # Step 1: Find the outer card boundary
    outer_quad = _find_card_boundary(gray)
    if outer_quad is None:
        result.error = "Could not detect card boundary"
        logger.warning(result.error)
        return result

    result.outer_contour = outer_quad

    # Step 2: Deskew the card to a rectangle
    card_img, _ = _deskew_image(gray, outer_quad)
    card_h, card_w = card_img.shape[:2]

    # Ensure portrait orientation
    if card_w > card_h:
        card_img = cv2.rotate(card_img, cv2.ROTATE_90_CLOCKWISE)
        card_h, card_w = card_img.shape[:2]

    logger.debug("Deskewed card: %dx%d", card_w, card_h)

    # Step 3: Find the inner frame edges
    borders = _find_inner_frame(card_img)

    top_px = borders["top"]
    bottom_px = borders["bottom"]
    left_px = borders["left"]
    right_px = borders["right"]

    # Sanity check: borders should be reasonable (1-20% of card dimension)
    min_border_h = card_h * 0.005
    max_border_h = card_h * 0.20
    min_border_w = card_w * 0.005
    max_border_w = card_w * 0.20

    for name, val, lo, hi in [
        ("top", top_px, min_border_h, max_border_h),
        ("bottom", bottom_px, min_border_h, max_border_h),
        ("left", left_px, min_border_w, max_border_w),
        ("right", right_px, min_border_w, max_border_w),
    ]:
        if val < lo or val > hi:
            logger.warning(
                "Border '%s' = %.1f px looks suspect (expected %.1f-%.1f)",
                name, val, lo, hi,
            )

    # Step 4: Convert to mm and compute ratios
    result.top = BorderMeasurement(pixels=top_px,
                                   mm=_pixels_to_mm(top_px, dpi))
    result.bottom = BorderMeasurement(pixels=bottom_px,
                                      mm=_pixels_to_mm(bottom_px, dpi))
    result.left = BorderMeasurement(pixels=left_px,
                                    mm=_pixels_to_mm(left_px, dpi))
    result.right = BorderMeasurement(pixels=right_px,
                                     mm=_pixels_to_mm(right_px, dpi))

    lr_a, lr_b, lr_pct = _compute_ratio(left_px, right_px)
    tb_a, tb_b, tb_pct = _compute_ratio(top_px, bottom_px)

    result.lr_ratio = (lr_a, lr_b)
    result.tb_ratio = (tb_a, tb_b)
    result.lr_pct = lr_pct
    result.tb_pct = tb_pct

    # Step 5: Assign PSA grade
    grade, label = _assign_psa_grade(lr_pct, tb_pct, is_back=is_back)
    result.psa_grade = grade
    result.psa_label = label
    result.success = True

    logger.info(
        "Centering for %s: L/R=%.1f/%.1f  T/B=%.1f/%.1f  PSA=%s",
        Path(image_path).name,
        lr_a, lr_b, tb_a, tb_b,
        f"{grade} ({label})" if grade else "below 7",
    )

    return result


def measure_centering_from_array(
    img: np.ndarray,
    dpi: float = DEFAULT_DPI,
    is_back: bool = False,
    label: str = "<array>",
) -> CenteringResult:
    """Measure centering from an already-loaded BGR image array.

    Same as measure_centering() but accepts a numpy array instead of a
    file path.  Useful when the image is already in memory (e.g. from
    the scan server pipeline).

    Parameters
    ----------
    img : np.ndarray
        BGR image (as from cv2.imread).
    dpi : float
        Scanner DPI for pixel-to-mm conversion.
    is_back : bool
        If True, use back-face PSA thresholds.
    label : str
        Label for logging.

    Returns
    -------
    CenteringResult
    """
    result = CenteringResult()

    if img is None or img.size == 0:
        result.error = "Empty or None image"
        return result

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    logger.debug("Centering analysis: %s (%dx%d, %.0f DPI)", label, w, h, dpi)

    outer_quad = _find_card_boundary(gray)
    if outer_quad is None:
        result.error = "Could not detect card boundary"
        return result

    result.outer_contour = outer_quad

    card_img, _ = _deskew_image(gray, outer_quad)
    card_h, card_w = card_img.shape[:2]

    if card_w > card_h:
        card_img = cv2.rotate(card_img, cv2.ROTATE_90_CLOCKWISE)
        card_h, card_w = card_img.shape[:2]

    borders = _find_inner_frame(card_img)

    top_px = borders["top"]
    bottom_px = borders["bottom"]
    left_px = borders["left"]
    right_px = borders["right"]

    result.top = BorderMeasurement(pixels=top_px,
                                   mm=_pixels_to_mm(top_px, dpi))
    result.bottom = BorderMeasurement(pixels=bottom_px,
                                      mm=_pixels_to_mm(bottom_px, dpi))
    result.left = BorderMeasurement(pixels=left_px,
                                    mm=_pixels_to_mm(left_px, dpi))
    result.right = BorderMeasurement(pixels=right_px,
                                     mm=_pixels_to_mm(right_px, dpi))

    lr_a, lr_b, lr_pct = _compute_ratio(left_px, right_px)
    tb_a, tb_b, tb_pct = _compute_ratio(top_px, bottom_px)

    result.lr_ratio = (lr_a, lr_b)
    result.tb_ratio = (tb_a, tb_b)
    result.lr_pct = lr_pct
    result.tb_pct = tb_pct

    grade, label_str = _assign_psa_grade(lr_pct, tb_pct, is_back=is_back)
    result.psa_grade = grade
    result.psa_label = label_str
    result.success = True

    logger.info(
        "Centering for %s: L/R=%.1f/%.1f  T/B=%.1f/%.1f  PSA=%s",
        label, lr_a, lr_b, tb_a, tb_b,
        f"{grade} ({label_str})" if grade else "below 7",
    )

    return result


def draw_debug_overlay(
    image_path: str,
    result: CenteringResult,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Draw centering measurement overlay on the card image.

    Draws:
    - Green lines showing detected outer card boundary
    - Red lines showing detected inner frame boundary
    - Border width annotations on each side
    - Centering ratio text

    Parameters
    ----------
    image_path : str
        Path to the original card image.
    result : CenteringResult
        Result from measure_centering().
    output_path : str, optional
        If provided, save the annotated image to this path.

    Returns
    -------
    np.ndarray
        Annotated BGR image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    overlay = img.copy()
    h, w = overlay.shape[:2]

    # Draw outer contour (green)
    if result.outer_contour is not None:
        pts = result.outer_contour.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)

    # Draw text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(h, w) / 2000.0)
    thickness = max(1, int(font_scale * 2))

    texts = [
        f"L/R: {result.lr_ratio[0]:.1f}/{result.lr_ratio[1]:.1f}",
        f"T/B: {result.tb_ratio[0]:.1f}/{result.tb_ratio[1]:.1f}",
    ]
    if result.psa_grade is not None:
        texts.append(f"PSA: {result.psa_grade} ({result.psa_label})")
    else:
        texts.append("PSA: below 7")

    y_offset = int(h * 0.03)
    for i, text in enumerate(texts):
        y = y_offset + int(i * font_scale * 35)
        cv2.putText(overlay, text, (10, y), font, font_scale,
                    (0, 0, 255), thickness, cv2.LINE_AA)

    # Border width labels
    border_texts = [
        (f"L: {result.left.mm:.2f}mm", (5, h // 2)),
        (f"R: {result.right.mm:.2f}mm", (w - int(w * 0.18), h // 2)),
        (f"T: {result.top.mm:.2f}mm", (w // 2 - int(w * 0.06), int(h * 0.06))),
        (f"B: {result.bottom.mm:.2f}mm",
         (w // 2 - int(w * 0.06), h - int(h * 0.02))),
    ]
    for text, pos in border_texts:
        cv2.putText(overlay, text, pos, font, font_scale * 0.7,
                    (255, 255, 0), thickness, cv2.LINE_AA)

    if output_path:
        cv2.imwrite(output_path, overlay)
        logger.info("Debug overlay saved to %s", output_path)

    return overlay


# ---------------------------------------------------------------------------
# CLI test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG,
                        format="%(name)s %(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m cardprice.ml.centering <image_path> [--dpi N] [--back] [--debug output.png]")
        print()
        print("Measure card centering from a high-resolution scan.")
        print()
        print("Options:")
        print("  --dpi N         Scanner DPI (default: 1290)")
        print("  --back          Use back-face PSA thresholds")
        print("  --debug FILE    Save debug overlay image")
        sys.exit(1)

    args = sys.argv[1:]
    image_paths = []
    dpi = DEFAULT_DPI
    is_back = False
    debug_path = None

    i = 0
    while i < len(args):
        if args[i] == "--dpi" and i + 1 < len(args):
            dpi = float(args[i + 1])
            i += 2
        elif args[i] == "--back":
            is_back = True
            i += 1
        elif args[i] == "--debug" and i + 1 < len(args):
            debug_path = args[i + 1]
            i += 2
        else:
            image_paths.append(args[i])
            i += 1

    for path in image_paths:
        print()
        print("=" * 60)
        print(f"File: {path}")
        print("=" * 60)

        result = measure_centering(path, dpi=dpi, is_back=is_back)
        print(result.summary())

        if debug_path and result.success:
            draw_debug_overlay(path, result, output_path=debug_path)
            print(f"\nDebug overlay: {debug_path}")
