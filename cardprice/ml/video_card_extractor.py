"""Extract individual card images from a slide-across video of a binder row.

The user records a ~3-5 second video sliding their phone across a row of cards
in a binder page. This module extracts individual card images by analyzing
frame-to-frame motion and brightness signals to find the moment each card is
centered in the camera's field of view.

Algorithm:
    1. Read all frames, compute per-frame brightness (center strip).
    2. Smooth the brightness signal to remove noise.
    3. Find peaks in brightness (cards are brighter than binder gutters).
    4. If peak count != num_cards, fall back to uniform time-based sampling.
    5. At each peak frame, auto-crop the card from the frame.

Usage:
    from cardprice.ml.video_card_extractor import extract_cards_from_video
    card_paths = extract_cards_from_video("slide.webm", num_cards=3, output_dir="out/")
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Pokemon card aspect ratio (63mm x 88mm)
CARD_RATIO = 63.0 / 88.0  # ~0.716
OUT_W, OUT_H = 420, 586


def extract_cards_from_video(
    video_path: str,
    num_cards: int = 3,
    output_dir: str | None = None,
) -> list[str]:
    """Extract individual card images from a slide-across video.

    Args:
        video_path: path to MP4/WEBM video of sliding across a binder row
        num_cards: expected number of cards in the row (default 3)
        output_dir: directory to write card images; if None, uses
                    <video_dir>/<video_stem>_cards/

    Returns:
        list of file paths to extracted card images, ordered left-to-right
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
    # Step 1: Read all frames and compute brightness signal
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
    brightness_signal: list[float] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        # Sample brightness from center vertical strip (middle 30% of width)
        h, w = frame.shape[:2]
        x0 = int(w * 0.35)
        x1 = int(w * 0.65)
        center_strip = frame[:, x0:x1]
        # Convert to grayscale luminance
        gray_strip = cv2.cvtColor(center_strip, cv2.COLOR_BGR2GRAY)
        brightness_signal.append(float(np.mean(gray_strip)))

    cap.release()

    if len(frames) < num_cards:
        raise RuntimeError(
            f"Video too short: {len(frames)} frames for {num_cards} cards"
        )

    brightness = np.array(brightness_signal, dtype=np.float64)
    logger.info(
        "Brightness range: %.1f - %.1f (mean %.1f)",
        brightness.min(), brightness.max(), brightness.mean(),
    )

    # ------------------------------------------------------------------
    # Step 2: Smooth the signal and find peaks
    # ------------------------------------------------------------------
    peak_indices = _find_card_peaks(brightness, fps, num_cards)

    # ------------------------------------------------------------------
    # Step 3: Extract and auto-crop card at each peak frame
    # ------------------------------------------------------------------
    card_paths = []
    for i, frame_idx in enumerate(peak_indices):
        frame = frames[frame_idx]
        card_img = _autocrop_card_frame(frame)
        card_path = out / f"card_{i:02d}.jpg"
        cv2.imwrite(str(card_path), card_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        card_paths.append(str(card_path))
        logger.info("Card %d: frame %d / %d -> %s", i, frame_idx, len(frames), card_path.name)

    # ------------------------------------------------------------------
    # Step 4: Save debug visualization
    # ------------------------------------------------------------------
    _save_debug_plot(brightness, peak_indices, out / "debug_brightness.png")

    return card_paths


def _find_card_peaks(
    brightness: np.ndarray,
    fps: float,
    num_cards: int,
) -> list[int]:
    """Find frame indices where cards are centered in the field of view.

    Uses scipy peak detection with adaptive prominence. Falls back to
    uniform time-based sampling if peak detection produces wrong count.
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d

    n = len(brightness)

    # Smooth with a window of ~0.15 seconds to remove high-freq noise
    smooth_window = max(3, int(fps * 0.15))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smoothed = uniform_filter1d(brightness, size=smooth_window)

    # Minimum distance between peaks: at least 0.4 seconds
    min_distance = max(3, int(fps * 0.4))

    # Try multiple prominence thresholds to get exactly num_cards peaks
    signal_range = smoothed.max() - smoothed.min()
    if signal_range < 1.0:
        # Extremely flat signal — fall back to uniform sampling
        logger.warning("Flat brightness signal (range=%.2f), using uniform sampling", signal_range)
        return _uniform_sample(n, num_cards)

    # Start with high prominence, decrease until we get enough peaks
    for prom_frac in [0.25, 0.15, 0.10, 0.05, 0.02]:
        prominence = signal_range * prom_frac
        peaks, properties = find_peaks(
            smoothed,
            distance=min_distance,
            prominence=prominence,
        )

        if len(peaks) >= num_cards:
            break

    if len(peaks) < num_cards:
        logger.warning(
            "Only found %d peaks (need %d), using uniform sampling",
            len(peaks), num_cards,
        )
        return _uniform_sample(n, num_cards)

    if len(peaks) > num_cards:
        # Too many peaks — take the num_cards most prominent
        prominences = properties["prominences"]
        top_indices = np.argsort(prominences)[-num_cards:]
        top_indices = np.sort(top_indices)  # maintain temporal order
        peaks = peaks[top_indices]
        logger.info("Filtered %d peaks to top %d by prominence", len(peaks) + num_cards, num_cards)

    logger.info("Peak frames: %s", list(peaks))
    return list(peaks)


def _uniform_sample(n: int, num_cards: int) -> list[int]:
    """Uniformly sample num_cards frame indices, avoiding first/last 10%."""
    start = int(n * 0.10)
    end = int(n * 0.90)
    indices = np.linspace(start, end, num_cards, dtype=int).tolist()
    logger.info("Uniform sample frames: %s", indices)
    return indices


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

    # Score candidates: prefer central, card-shaped, large
    best = None
    best_score = float("inf")

    for pts in candidates:
        area = cv2.contourArea(pts)
        if area < min_area:
            continue

        # Aspect ratio of bounding rect
        rect = cv2.minAreaRect(pts)
        rw, rh = sorted(rect[1])
        if rh == 0:
            continue
        ratio = rw / rh
        ratio_err = abs(ratio - CARD_RATIO)
        if ratio_err > 0.20:
            continue

        # Distance of centroid from image center
        centroid = pts.mean(axis=0)
        dist = np.linalg.norm(centroid - center) / max(w, h)

        # Score: lower is better (prefer centered, correct ratio, large)
        score = ratio_err * 3.0 + dist * 2.0 - (area / (w * h))
        if score < best_score:
            best_score = score
            best = pts

    if best is not None:
        return _perspective_crop(frame, best)
    else:
        # Fallback: center crop with card aspect ratio
        return _center_crop(frame)


def _perspective_crop(frame: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Perspective-warp a 4-point quadrilateral to a clean rectangle."""
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = _order_points(pts)
    dst = np.array(
        [[0, 0], [OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]],
        dtype=np.float32,
    )

    # Determine if the card is landscape (rotated 90 degrees)
    w_top = np.linalg.norm(rect[1] - rect[0])
    h_left = np.linalg.norm(rect[3] - rect[0])
    if w_top > h_left * 1.1:
        # Landscape — rotate destination
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
    rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum
    rect[1] = pts[np.argmin(d)]   # top-right has smallest difference
    rect[3] = pts[np.argmax(d)]   # bottom-left has largest difference
    return rect


def _center_crop(frame: np.ndarray) -> np.ndarray:
    """Center-crop a frame to card aspect ratio and resize."""
    h, w = frame.shape[:2]
    # Fit the largest card-ratio rectangle in the center
    target_ratio = CARD_RATIO  # w/h
    frame_ratio = w / h

    if frame_ratio > target_ratio:
        # Frame is wider — crop width
        crop_w = int(h * target_ratio)
        x0 = (w - crop_w) // 2
        cropped = frame[:, x0:x0 + crop_w]
    else:
        # Frame is taller — crop height
        crop_h = int(w / target_ratio)
        y0 = (h - crop_h) // 2
        cropped = frame[y0:y0 + crop_h, :]

    return cv2.resize(cropped, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)


def _save_debug_plot(
    brightness: np.ndarray,
    peak_indices: list[int],
    path: Path,
) -> None:
    """Save a brightness signal plot with peak markers for debugging."""
    try:
        # Draw the plot using OpenCV (no matplotlib dependency)
        plot_w, plot_h = 800, 200
        plot = np.ones((plot_h, plot_w, 3), dtype=np.uint8) * 30  # dark bg

        n = len(brightness)
        if n < 2:
            return

        # Normalize brightness to plot height
        b_min, b_max = brightness.min(), brightness.max()
        b_range = b_max - b_min if b_max > b_min else 1.0

        # Draw brightness curve
        for i in range(1, n):
            x0 = int((i - 1) / (n - 1) * (plot_w - 1))
            x1 = int(i / (n - 1) * (plot_w - 1))
            y0 = plot_h - 1 - int((brightness[i - 1] - b_min) / b_range * (plot_h - 20))
            y1 = plot_h - 1 - int((brightness[i] - b_min) / b_range * (plot_h - 20))
            cv2.line(plot, (x0, y0), (x1, y1), (100, 200, 100), 1)

        # Draw peak markers
        for idx in peak_indices:
            x = int(idx / (n - 1) * (plot_w - 1))
            y = plot_h - 1 - int((brightness[idx] - b_min) / b_range * (plot_h - 20))
            cv2.circle(plot, (x, y), 5, (0, 0, 255), -1)
            cv2.line(plot, (x, 0), (x, plot_h), (0, 0, 200), 1)

        cv2.imwrite(str(path), plot)
        logger.info("Debug plot saved: %s", path)
    except Exception as e:
        logger.warning("Could not save debug plot: %s", e)
