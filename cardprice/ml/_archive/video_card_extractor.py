"""Extract individual card images from a slide-across video of a binder row.

The user records a ~3-5 second video sliding their phone across a row of cards
in a binder page. This module extracts individual card images by analyzing
frame-to-frame brightness, detail, and color signals to find the moment each
card is centered in the camera's field of view.

Two detection strategies:
1. **Brightness + detail peaks**: Build a combined brightness/Laplacian signal
   from the center strip of each frame. Cards are brighter and more detailed
   than binder gutters. Smooth and find peaks.
2. **HSV gutter detection**: Detect frames where the center strip is dominated
   by binder-colored gutters (high saturation, uniform hue, low detail).
   Card frames are the midpoints between consecutive gutter regions.

Strategy 2 is preferred when gutters are clearly colored (blue/red binders).
Strategy 1 is the fallback for neutral-colored binders.

Usage:
    from cardprice.ml.video_card_extractor import extract_cards_from_video
    results = extract_cards_from_video("slide.webm", num_cards=3, output_dir="out/")
    for r in results:
        print(r["path"], r["frame_number"], r["confidence"])
"""

import logging
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Pokemon card aspect ratio (63mm x 88mm)
CARD_RATIO = 63.0 / 88.0  # ~0.716
OUT_W, OUT_H = 420, 586


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_cards_from_video(
    video_path: str,
    num_cards: int = 3,
    output_dir: str | None = None,
    strategy: str = "auto",
    debug: bool = False,
) -> list[dict]:
    """Extract individual card images from a slide-across video.

    Args:
        video_path: path to MP4/WEBM video of sliding across a binder row
        num_cards: expected number of cards in the row (default 3)
        output_dir: directory to write card images; if None, uses
                    <video_dir>/<video_stem>_cards/
        strategy: "auto" (gutter first, then brightness), "gutter", or
                  "brightness"
        debug: if True, save debug signal plots and raw frames

    Returns:
        list of dicts, each with:
            - path: str, path to extracted card image
            - frame_number: int, source frame index
            - confidence: float, detection confidence (0-1)
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_dir is None:
        out = video_path.parent / f"{video_path.stem}_cards"
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Read all frames
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        "Video: %s, %.1f fps, %d frames (%.1fs)",
        video_path.name, fps, total_frames, total_frames / fps,
    )

    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < num_cards:
        raise RuntimeError(
            f"Video too short: {len(frames)} frames for {num_cards} cards"
        )

    logger.info("Read %d frames from video", len(frames))

    # ------------------------------------------------------------------
    # Step 2: Detect card-centered frames
    # ------------------------------------------------------------------
    if strategy == "auto":
        result = _detect_via_gutters(frames, fps, num_cards)
        if result is None:
            logger.info("Gutter detection failed, falling back to brightness peaks")
            result = _detect_via_brightness(frames, fps, num_cards)
        else:
            logger.info("Gutter detection succeeded")
    elif strategy == "gutter":
        result = _detect_via_gutters(frames, fps, num_cards)
        if result is None:
            raise ValueError("Gutter detection failed - no clear gutters found")
    elif strategy == "brightness":
        result = _detect_via_brightness(frames, fps, num_cards)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    peak_indices, confidences = result

    # ------------------------------------------------------------------
    # Step 3: Extract and auto-crop card at each peak frame
    # ------------------------------------------------------------------
    extracted = []
    for i, (frame_idx, conf) in enumerate(zip(peak_indices, confidences)):
        frame = frames[frame_idx]
        card_img = _autocrop_card_frame(frame)
        card_path = out / f"card_{i:02d}.jpg"
        cv2.imwrite(str(card_path), card_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        logger.info(
            "Card %d: frame %d/%d (conf=%.3f) -> %s",
            i, frame_idx, len(frames), conf, card_path.name,
        )
        extracted.append({
            "path": str(card_path),
            "frame_number": int(frame_idx),
            "confidence": float(conf),
        })

    # ------------------------------------------------------------------
    # Step 4: Save debug visualization
    # ------------------------------------------------------------------
    if debug:
        _save_debug_all(frames, fps, num_cards, peak_indices, out)
    else:
        # Always save the lightweight brightness plot
        brightness = _compute_brightness_signal(frames)
        _save_debug_plot(brightness, peak_indices, out / "debug_brightness.png")

    return extracted


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def _compute_brightness_signal(frames: list[np.ndarray]) -> np.ndarray:
    """Compute center-strip brightness for each frame.

    Uses the center 30% width, full height. This strip covers the card
    when it's centered in the frame, and catches the darker gutter when
    the camera is between cards.
    """
    signal = np.zeros(len(frames), dtype=np.float64)
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        x0 = int(w * 0.35)
        x1 = int(w * 0.65)
        gray = cv2.cvtColor(frame[:, x0:x1], cv2.COLOR_BGR2GRAY)
        signal[i] = gray.mean()
    return signal


def _compute_detail_signal(frames: list[np.ndarray]) -> np.ndarray:
    """Compute center-strip detail level (Laplacian variance) per frame.

    Cards have more high-frequency detail (text, art) than gutters, making
    this more robust than brightness when cards and gutters have similar
    luminance.
    """
    signal = np.zeros(len(frames), dtype=np.float64)
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        x0 = int(w * 0.35)
        x1 = int(w * 0.65)
        gray = cv2.cvtColor(frame[:, x0:x1], cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        signal[i] = lap.var()
    return signal


def _compute_gutter_signal(frames: list[np.ndarray]) -> np.ndarray:
    """Compute a gutter-likelihood signal for each frame.

    Gutters are the colored binder material between cards. They typically
    have high saturation (colored plastic) and uniform hue. Card frames
    have low saturation or varied hue (artwork).
    """
    signal = np.zeros(len(frames), dtype=np.float64)
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        x0 = int(w * 0.30)
        x1 = int(w * 0.70)
        strip = frame[:, x0:x1]
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

        sat = hsv[:, :, 1].astype(np.float64)
        hue = hsv[:, :, 0].astype(np.float64)

        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Gutter indicators: high saturation, low hue variance, low detail
        sat_score = sat.mean() / 255.0
        hue_uniformity = max(0.0, 1.0 - hue.std() / 40.0)
        detail_inv = max(0.0, 1.0 - min(lap_var / 500.0, 1.0))

        signal[i] = sat_score * 0.4 + hue_uniformity * 0.3 + detail_inv * 0.3

    return signal


def _smooth(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply a rolling average to smooth a 1D signal."""
    if len(signal) < window or window < 2:
        return signal.copy()
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(signal, size=window)


# ---------------------------------------------------------------------------
# Strategy: brightness + detail peaks
# ---------------------------------------------------------------------------

def _detect_via_brightness(
    frames: list[np.ndarray],
    fps: float,
    num_cards: int,
) -> tuple[list[int], list[float]]:
    """Detect card frames using combined brightness + detail peak detection."""
    from scipy.signal import find_peaks

    brightness = _compute_brightness_signal(frames)
    detail = _compute_detail_signal(frames)

    smooth_window = max(3, int(fps * 0.15))
    b_smooth = _smooth(brightness, smooth_window)
    d_smooth = _smooth(detail, smooth_window)

    # Normalize to [0, 1] and combine
    def _norm(sig):
        r = sig.max() - sig.min()
        return (sig - sig.min()) / r if r > 1e-6 else np.zeros_like(sig)

    combined = 0.4 * _norm(b_smooth) + 0.6 * _norm(d_smooth)

    sig_range = combined.max() - combined.min()
    if sig_range < 0.01:
        logger.warning("Flat combined signal (range=%.4f), using uniform sampling", sig_range)
        return _uniform_sample(len(frames), num_cards)

    min_distance = max(3, int(fps * 0.4))

    # Try decreasing prominence thresholds
    for prom_frac in [0.25, 0.15, 0.10, 0.05, 0.02]:
        peaks, props = find_peaks(
            combined,
            distance=min_distance,
            prominence=sig_range * prom_frac,
        )
        if len(peaks) >= num_cards:
            break

    if len(peaks) < num_cards:
        logger.warning(
            "Only found %d peaks (need %d), using uniform sampling",
            len(peaks), num_cards,
        )
        return _uniform_sample(len(frames), num_cards)

    if len(peaks) > num_cards:
        # Keep the num_cards most prominent, in temporal order
        prominences = props["prominences"]
        top = np.argsort(prominences)[-num_cards:]
        top = np.sort(top)
        peaks = peaks[top]
        prominences = prominences[top]
    else:
        prominences = props["prominences"]

    logger.info("Peak frames: %s", list(peaks))

    # Confidence from prominence
    max_prom = prominences.max() if len(prominences) > 0 else 1.0
    confidences = [
        float(min(1.0, 0.5 + 0.5 * (p / max_prom))) for p in prominences
    ]

    return list(peaks), confidences


# ---------------------------------------------------------------------------
# Strategy: HSV gutter detection
# ---------------------------------------------------------------------------

def _detect_via_gutters(
    frames: list[np.ndarray],
    fps: float,
    num_cards: int,
) -> tuple[list[int], list[float]] | None:
    """Detect card frames by finding gutters between cards.

    For N cards, we expect N-1 internal gutters (and possibly 2 edge gutters).
    Finds gutter regions, then picks the best frame in each card segment.

    Returns (frame_indices, confidences) or None if gutter detection fails.
    """
    gutter_signal = _compute_gutter_signal(frames)
    smoothed = _smooth(gutter_signal, window=max(3, int(fps * 0.2)))

    sig_range = smoothed.max() - smoothed.min()
    if sig_range < 0.10:
        return None  # not enough contrast

    # Threshold: median + 0.2 * range
    threshold = np.median(smoothed) + 0.2 * sig_range
    is_gutter = smoothed > threshold

    # Find contiguous gutter regions
    gutter_regions = []
    in_region = False
    start = 0
    for i in range(len(is_gutter)):
        if is_gutter[i] and not in_region:
            start = i
            in_region = True
        elif not is_gutter[i] and in_region:
            gutter_regions.append((start, i - 1))
            in_region = False
    if in_region:
        gutter_regions.append((start, len(is_gutter) - 1))

    # Filter short gutter regions (noise)
    min_gutter_frames = max(2, int(fps * 0.1))
    gutter_regions = [
        (s, e) for s, e in gutter_regions if (e - s + 1) >= min_gutter_frames
    ]

    logger.info("Found %d gutter regions (threshold=%.3f)", len(gutter_regions), threshold)

    if len(gutter_regions) < num_cards - 1:
        return None

    # Build card segments between gutters
    card_segments = []

    # Before first gutter (if enough frames for a card)
    if gutter_regions[0][0] > max(2, int(fps * 0.15)):
        card_segments.append((0, gutter_regions[0][0] - 1))

    # Between consecutive gutters
    for i in range(len(gutter_regions) - 1):
        seg_start = gutter_regions[i][1] + 1
        seg_end = gutter_regions[i + 1][0] - 1
        if seg_end > seg_start:
            card_segments.append((seg_start, seg_end))

    # After last gutter (if enough frames for a card)
    if gutter_regions[-1][1] < len(frames) - 1 - max(2, int(fps * 0.15)):
        card_segments.append((gutter_regions[-1][1] + 1, len(frames) - 1))

    logger.info("Found %d card segments between gutters", len(card_segments))

    if len(card_segments) < num_cards:
        return None

    # Score segments by card-ness (inverse gutter signal)
    scored = []
    for seg_start, seg_end in card_segments:
        seg_len = seg_end - seg_start + 1
        if seg_len < max(2, int(fps * 0.15)):
            continue
        card_score = 1.0 - smoothed[seg_start:seg_end + 1].mean()
        scored.append((card_score, seg_start, seg_end))

    scored.sort(key=lambda x: -x[0])
    selected = scored[:num_cards]
    selected.sort(key=lambda x: x[1])  # temporal order

    frame_indices = []
    confidences = []
    for card_score, seg_start, seg_end in selected:
        # Within the segment, pick the frame with lowest gutter signal
        seg_signal = smoothed[seg_start:seg_end + 1]
        best_offset = int(np.argmin(seg_signal))
        best_frame = seg_start + best_offset
        frame_indices.append(best_frame)
        confidences.append(float(min(1.0, card_score + 0.3)))

    return frame_indices, confidences


# ---------------------------------------------------------------------------
# Uniform fallback
# ---------------------------------------------------------------------------

def _uniform_sample(n: int, num_cards: int) -> tuple[list[int], list[float]]:
    """Uniformly sample num_cards frame indices, avoiding first/last 10%."""
    start = int(n * 0.10)
    end = int(n * 0.90)
    indices = np.linspace(start, end, num_cards, dtype=int).tolist()
    confidences = [0.3] * num_cards
    logger.info("Uniform sample frames: %s", indices)
    return indices, confidences


# ---------------------------------------------------------------------------
# Card extraction from individual frames
# ---------------------------------------------------------------------------

def _autocrop_card_frame(frame: np.ndarray) -> np.ndarray:
    """Auto-crop the most prominent card from a single video frame.

    Uses contour detection to find the card, then perspective-warps to a
    clean rectangle. Falls back to a center crop if no card is found.
    """
    h, w = frame.shape[:2]
    min_area = 0.05 * w * h
    max_area = 0.85 * w * h
    center = np.array([w / 2.0, h / 2.0])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates = []

    # Strategy 1: Canny edges at multiple thresholds
    for lo, hi in [(20, 80), (30, 100), (50, 150)]:
        edges = cv2.Canny(blur, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.02, 0.03, 0.04, 0.05, 0.06):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4:
                    candidates.append(approx.reshape(4, 2).astype(np.float32))
                    break

    # Strategy 2: minAreaRect from large contours
    for lo, hi in [(20, 80), (30, 100)]:
        edges = cv2.Canny(blur, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=3)
        edges = cv2.erode(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            rect = cv2.minAreaRect(cnt)
            rw, rh = sorted(rect[1])
            if rh == 0:
                continue
            ratio = rw / rh
            if 0.55 < ratio < 0.90:
                box = cv2.boxPoints(rect).astype(np.float32)
                candidates.append(box)

    # Strategy 3: HSV-based binder exclusion
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, (90, 40, 40), (130, 255, 255))
    mask_red1 = cv2.inRange(hsv, (0, 40, 40), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (160, 40, 40), (180, 255, 255))
    mask_dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
    mask_binder = mask_blue | mask_red1 | mask_red2 | mask_dark
    mask_card = cv2.bitwise_not(mask_binder)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask_card = cv2.morphologyEx(mask_card, cv2.MORPH_OPEN, kernel)
    mask_card = cv2.morphologyEx(mask_card, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_card, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = sorted(rect[1])
        if rh > 0 and 0.55 < rw / rh < 0.90:
            box = cv2.boxPoints(rect).astype(np.float32)
            candidates.append(box)

    # Score candidates: prefer central, card-shaped, large
    best = None
    best_score = float("inf")

    for pts in candidates:
        area = cv2.contourArea(pts)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(pts)
        rw, rh = sorted(rect[1])
        if rh == 0:
            continue
        ratio = rw / rh
        ratio_err = abs(ratio - CARD_RATIO)
        if ratio_err > 0.20:
            continue

        centroid = pts.mean(axis=0)
        dist = np.linalg.norm(centroid - center) / max(w, h)

        score = ratio_err * 3.0 + dist * 2.0 - (area / (w * h))
        if score < best_score:
            best_score = score
            best = pts

    if best is not None:
        return _perspective_crop(frame, best)
    else:
        return _center_crop(frame)


def _perspective_crop(frame: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Perspective-warp a 4-point quadrilateral to a clean rectangle."""
    rect = _order_points(pts)
    dst = np.array(
        [[0, 0], [OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]],
        dtype=np.float32,
    )

    w_top = np.linalg.norm(rect[1] - rect[0])
    h_left = np.linalg.norm(rect[3] - rect[0])
    if w_top > h_left * 1.1:
        # Landscape -- rotate destination points
        dst = np.array(
            [[OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1], [0, 0]],
            dtype=np.float32,
        )

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, M, (OUT_W, OUT_H))


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _center_crop(frame: np.ndarray) -> np.ndarray:
    """Center-crop a frame to card aspect ratio and resize."""
    h, w = frame.shape[:2]
    target_ratio = CARD_RATIO  # w/h
    frame_ratio = w / h

    if frame_ratio > target_ratio:
        crop_w = int(h * target_ratio)
        x0 = (w - crop_w) // 2
        cropped = frame[:, x0:x0 + crop_w]
    else:
        crop_h = int(w / target_ratio)
        y0 = (h - crop_h) // 2
        cropped = frame[y0:y0 + crop_h, :]

    return cv2.resize(cropped, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Synthetic test video creation
# ---------------------------------------------------------------------------

def create_synthetic_test_video(
    card_image_paths: list[str],
    output_path: str,
    fps: float = 30.0,
    slide_duration: float = 3.0,
    gutter_color: tuple[int, int, int] = (180, 60, 60),
    frame_size: tuple[int, int] = (640, 480),
) -> str:
    """Create a synthetic slide video from card images for testing.

    Simulates a camera sliding across a row of cards with colored gutters
    between them. Useful for testing the extraction pipeline without a
    real binder.

    Args:
        card_image_paths: list of paths to card images
        output_path: where to save the video (must end with .mp4)
        fps: frames per second
        slide_duration: total video duration in seconds
        gutter_color: BGR color of the gutter between cards
        frame_size: (width, height) of each video frame

    Returns:
        Path to the created video.
    """
    num_cards = len(card_image_paths)
    fw, fh = frame_size
    total_frames = int(fps * slide_duration)

    # Load and resize card images to fit the frame height
    card_h = fh
    card_w = int(card_h * CARD_RATIO)
    gutter_w = int(card_w * 0.20)  # gutter is 20% of card width

    cards = []
    for path in card_image_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot read card image: {path}")
        resized = cv2.resize(img, (card_w, card_h))
        cards.append(resized)

    # Build panorama: gutter | card | gutter | card | ... | gutter
    total_w = gutter_w + num_cards * card_w + (num_cards - 1) * gutter_w + gutter_w
    panorama = np.full((fh, total_w, 3), gutter_color, dtype=np.uint8)

    x = gutter_w
    for card in cards:
        panorama[:, x:x + card_w] = card
        x += card_w + gutter_w

    # Simulate sliding camera viewport across the panorama
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (fw, fh))

    max_offset = max(1, total_w - fw)

    for i in range(total_frames):
        t = i / max(1, total_frames - 1)
        # Smooth ease-in-out motion
        t_smooth = 0.5 * (1 - np.cos(np.pi * t))
        x_offset = int(t_smooth * max_offset)
        x_offset = min(x_offset, total_w - fw)
        x_offset = max(0, x_offset)

        frame = panorama[:, x_offset:x_offset + fw].copy()
        out.write(frame)

    out.release()
    logger.info(
        "Created synthetic video: %s (%d frames, %d cards)",
        output_path, total_frames, num_cards,
    )
    return str(output_path)


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------

def _save_debug_plot(
    brightness: np.ndarray,
    peak_indices: list[int],
    path: Path,
) -> None:
    """Save a brightness signal plot with peak markers for debugging."""
    try:
        plot_w, plot_h = 800, 200
        plot = np.ones((plot_h, plot_w, 3), dtype=np.uint8) * 30

        n = len(brightness)
        if n < 2:
            return

        b_min, b_max = brightness.min(), brightness.max()
        b_range = b_max - b_min if b_max > b_min else 1.0

        for i in range(1, n):
            x0 = int((i - 1) / (n - 1) * (plot_w - 1))
            x1 = int(i / (n - 1) * (plot_w - 1))
            y0 = plot_h - 1 - int((brightness[i - 1] - b_min) / b_range * (plot_h - 20))
            y1 = plot_h - 1 - int((brightness[i] - b_min) / b_range * (plot_h - 20))
            cv2.line(plot, (x0, y0), (x1, y1), (100, 200, 100), 1)

        for idx in peak_indices:
            x = int(idx / (n - 1) * (plot_w - 1))
            y = plot_h - 1 - int((brightness[idx] - b_min) / b_range * (plot_h - 20))
            cv2.circle(plot, (x, y), 5, (0, 0, 255), -1)
            cv2.line(plot, (x, 0), (x, plot_h), (0, 0, 200), 1)

        cv2.imwrite(str(path), plot)
        logger.info("Debug plot saved: %s", path)
    except Exception as e:
        logger.warning("Could not save debug plot: %s", e)


def _save_debug_all(
    frames: list[np.ndarray],
    fps: float,
    num_cards: int,
    peak_indices: list[int],
    output_dir: Path,
) -> None:
    """Save full debug visualizations: all signals + raw peak frames."""
    try:
        brightness = _compute_brightness_signal(frames)
        detail = _compute_detail_signal(frames)
        gutter = _compute_gutter_signal(frames)

        sw = max(3, int(fps * 0.15))
        b_smooth = _smooth(brightness, sw)
        d_smooth = _smooth(detail, sw)
        g_smooth = _smooth(gutter, max(3, int(fps * 0.2)))

        n = len(frames)
        plot_w, plot_h = max(n, 600), 150
        canvas = np.ones((plot_h * 3 + 20, plot_w, 3), dtype=np.uint8) * 30

        def _draw(signal, y_off, color, label):
            s = signal.copy()
            r = s.max() - s.min()
            s = (s - s.min()) / r if r > 1e-6 else np.zeros_like(s)
            for j in range(1, len(s)):
                y0 = y_off + plot_h - 10 - int(s[j - 1] * (plot_h - 20))
                y1 = y_off + plot_h - 10 - int(s[j] * (plot_h - 20))
                cv2.line(canvas, (j - 1, y0), (j, y1), color, 1)
            cv2.putText(canvas, label, (5, y_off + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        _draw(b_smooth, 0, (100, 200, 100), "Brightness")
        _draw(d_smooth, plot_h, (200, 150, 50), "Detail")
        _draw(g_smooth, plot_h * 2, (50, 100, 255), "Gutter")

        for idx in peak_indices:
            for y_off in [0, plot_h, plot_h * 2]:
                cv2.line(canvas, (idx, y_off), (idx, y_off + plot_h), (0, 200, 0), 2)

        cv2.imwrite(str(output_dir / "debug_signals.png"), canvas)

        # Save raw peak frames
        for i, idx in enumerate(peak_indices):
            cv2.imwrite(str(output_dir / f"debug_frame_{i:02d}.jpg"), frames[idx])

        logger.info("Debug visualizations saved to %s", output_dir)
    except Exception as e:
        logger.warning("Failed to save debug visualizations: %s", e)
