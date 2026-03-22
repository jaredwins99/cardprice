"""Perspective correction and normalization for segmented binder page cards.

After segment_cards() extracts individual cards, each image may have residual
perspective distortion, uneven lighting from binder sleeves, and color cast
from colored binder pages. This module applies:

1. Perspective correction -- warp detected card quadrilateral to a standard rectangle
2. CLAHE contrast normalization -- compensate for uneven lighting across the card
3. White balance correction -- remove color cast from binder sleeve tint

Standard Pokemon card: 63mm x 88mm -> 420 x 586 pixels at ~6.67 px/mm.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard card output dimensions (63mm x 88mm at ~6.67 px/mm)
CORRECTED_W = 420
CORRECTED_H = 586


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _detect_card_corners(card_img: np.ndarray) -> Optional[np.ndarray]:
    """Detect card edges in a segmented card image and return 4 corner points.

    Uses Canny edge detection + contour finding to locate the card rectangle
    within the already-cropped card image.

    Returns:
        4x2 float32 array of ordered corners (TL, TR, BR, BL), or None.
    """
    h, w = card_img.shape[:2]
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    median_val = np.median(blurred)
    low = int(max(0, 0.5 * median_val))
    high = int(min(255, 1.5 * median_val))
    edges = cv2.Canny(blurred, low, high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = h * w
    best = None
    best_area = 0

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(cnt)
        if area < image_area * 0.4:  # must cover 40%+ of image
            continue
        peri = cv2.arcLength(cnt, True)
        for eps_mult in (0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(cnt, eps_mult * peri, True)
            if len(approx) == 4 and area > best_area:
                # Validate: corners should be near image edges (this is a
                # cropped card, so the card border should span most of the
                # image). Reject detections that are interior features.
                pts = _order_points(approx.reshape(4, 2).astype(np.float32))
                bx, by, bw, bh = cv2.boundingRect(pts.astype(np.int32))
                # Bounding box should cover at least 70% of each dimension
                if bw < w * 0.7 or bh < h * 0.7:
                    continue
                # Check aspect ratio is card-like (0.55-0.85, card is ~0.716)
                aspect = bw / bh if bh > 0 else 0
                if aspect < 0.55 or aspect > 0.85:
                    continue
                best = approx
                best_area = area
                break

    if best is None:
        return None

    return _order_points(best.reshape(4, 2).astype(np.float32))


def _perspective_warp(card_img: np.ndarray, corners: np.ndarray,
                      out_w: int = CORRECTED_W,
                      out_h: int = CORRECTED_H) -> np.ndarray:
    """Warp card image using corners to a standard rectangle.

    Skips warp if the quad is already nearly rectangular (ratio > 0.97)
    to avoid introducing resampling artifacts.
    """
    ordered = _order_points(corners.reshape(-1, 2).astype(np.float32))

    # Check if distortion is significant enough to warrant correction
    tl, tr, br, bl = ordered
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    w_ratio = min(width_top, width_bot) / max(width_top, width_bot) if max(width_top, width_bot) > 0 else 1.0
    h_ratio = min(height_left, height_right) / max(height_left, height_right) if max(height_left, height_right) > 0 else 1.0

    if w_ratio > 0.97 and h_ratio > 0.97:
        logger.debug("Quad near-rectangular (w=%.3f, h=%.3f), skipping warp",
                     w_ratio, h_ratio)
        return cv2.resize(card_img, (out_w, out_h),
                          interpolation=cv2.INTER_LANCZOS4)

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(card_img, M, (out_w, out_h),
                                  flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REPLICATE)
    return warped


def _apply_clahe(image: np.ndarray, clip_limit: float = 1.5,
                 grid_size: tuple[int, int] = (4, 4)) -> np.ndarray:
    """Apply CLAHE contrast normalization to handle uneven lighting.

    Converts to LAB color space and applies CLAHE to the L channel only,
    preserving color while normalizing brightness. Uses conservative
    clip_limit=1.5 and coarser grid (4x4) to avoid over-enhancing noise
    and artifacts in binder sleeve photos.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_corrected = clahe.apply(l_channel)

    lab_corrected = cv2.merge([l_corrected, a_channel, b_channel])
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


def _white_balance(image: np.ndarray) -> np.ndarray:
    """Gray-world white balance to remove binder sleeve color cast.

    Uses a conservative 40% blend to avoid over-correcting cards that are
    legitimately dominated by one color (fire-type, water-type, etc.).
    """
    result = image.astype(np.float32)
    avg_b = result[:, :, 0].mean()
    avg_g = result[:, :, 1].mean()
    avg_r = result[:, :, 2].mean()
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    if avg_b < 1 or avg_g < 1 or avg_r < 1:
        return image

    blend = 0.4
    scale_b = 1.0 + blend * (avg_all / avg_b - 1.0)
    scale_g = 1.0 + blend * (avg_all / avg_g - 1.0)
    scale_r = 1.0 + blend * (avg_all / avg_r - 1.0)

    result[:, :, 0] *= scale_b
    result[:, :, 1] *= scale_g
    result[:, :, 2] *= scale_r

    return np.clip(result, 0, 255).astype(np.uint8)


def correct_card_image(card_img: np.ndarray,
                       corners: Optional[np.ndarray] = None,
                       apply_perspective: bool = False,
                       apply_contrast: bool = False,
                       apply_wb: bool = True) -> np.ndarray:
    """Apply perspective correction and normalization to a segmented card.

    By default only applies white balance correction, which gives the most
    consistent DINOv2 score improvement (+0.0020 mean across 94 test cards).
    Perspective correction is off by default because the segmenter already
    applies it; doing it twice introduces resampling artifacts. CLAHE is off
    by default because it helps some cards but hurts others equally.

    Args:
        card_img: BGR image of a single card (from segmenter).
        corners: Optional 4 corner points (4x2 array). If None and
            apply_perspective is True, corners are auto-detected.
        apply_perspective: Whether to apply perspective correction.
            Default False (segmenter already corrects perspective).
        apply_contrast: Whether to apply CLAHE contrast normalization.
            Default False (inconsistent improvement).
        apply_wb: Whether to apply white balance correction.
            Default True (consistent +0.0020 mean improvement).

    Returns:
        Corrected BGR image at standard card dimensions (420x586).
    """
    result = card_img.copy()

    if apply_perspective:
        if corners is not None:
            result = _perspective_warp(result, corners)
        else:
            detected = _detect_card_corners(result)
            if detected is not None:
                result = _perspective_warp(result, detected)
            else:
                result = cv2.resize(result, (CORRECTED_W, CORRECTED_H),
                                    interpolation=cv2.INTER_LANCZOS4)
                logger.debug("No card corners detected, using simple resize")
    else:
        result = cv2.resize(result, (CORRECTED_W, CORRECTED_H),
                            interpolation=cv2.INTER_LANCZOS4)

    if apply_contrast:
        result = _apply_clahe(result)

    if apply_wb:
        result = _white_balance(result)

    return result


def correct_page_cards(card_images: list[np.ndarray],
                       card_corners_list: Optional[list[Optional[np.ndarray]]] = None,
                       apply_perspective: bool = False,
                       apply_contrast: bool = False,
                       apply_wb: bool = True) -> list[np.ndarray]:
    """Correct all cards from a binder page.

    Args:
        card_images: List of BGR card images from segment_cards().
        card_corners_list: Optional list of corner arrays, one per card.
        apply_perspective: Whether to apply perspective correction.
        apply_contrast: Whether to apply CLAHE contrast normalization.
        apply_wb: Whether to apply white balance correction.

    Returns:
        List of corrected BGR images, same length as input.
    """
    if card_corners_list is None:
        card_corners_list = [None] * len(card_images)

    corrected = []
    for i, (img, corners) in enumerate(zip(card_images, card_corners_list)):
        try:
            corrected_img = correct_card_image(
                img, corners=corners,
                apply_perspective=apply_perspective,
                apply_contrast=apply_contrast,
                apply_wb=apply_wb,
            )
            corrected.append(corrected_img)
        except Exception:
            logger.warning("Failed to correct card %d, returning original resized",
                           i, exc_info=True)
            corrected.append(cv2.resize(img, (CORRECTED_W, CORRECTED_H),
                                        interpolation=cv2.INTER_LANCZOS4))

    return corrected
