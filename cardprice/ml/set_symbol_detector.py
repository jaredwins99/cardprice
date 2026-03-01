"""Set symbol extraction from Pokemon card images.

Pokemon cards have a set symbol (small icon) in the bottom-right area of the
artwork box.  This module crops that region so it can be used as an additional
feature for card matching (e.g. narrowing candidate sets before full-card
comparison).

Typical location on a standard Pokemon card:
    - ~73-80 % from left edge
    - ~56-64 % from top edge
    - ~6 % of card width square

The crop is intentionally generous to tolerate slight variation in card layout,
rotation, and photo framing.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default region (fraction of card dimensions).
# These are tuned for standard Pokemon card layouts.
_DEFAULT_X_CENTER = 0.76   # 76 % from left
_DEFAULT_Y_CENTER = 0.60   # 60 % from top
_DEFAULT_SIZE = 0.06       # crop box is 6 % of card width (square)


def extract_set_symbol(
    image_path,
    *,
    x_center=_DEFAULT_X_CENTER,
    y_center=_DEFAULT_Y_CENTER,
    size=_DEFAULT_SIZE,
    output_size=(64, 64),
):
    """Crop the set-symbol region from a Pokemon card image.

    Parameters
    ----------
    image_path : str or Path
        Path to the card image (JPEG/PNG).
    x_center, y_center : float
        Fractional position of the symbol centre (0-1).
    size : float
        Fractional width/height of the crop square (0-1).
    output_size : tuple[int, int]
        Resize the crop to this (width, height) for consistent downstream use.

    Returns
    -------
    numpy.ndarray
        BGR crop of the set-symbol region, resized to *output_size*.

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    ValueError
        If the image cannot be decoded.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    h, w = img.shape[:2]

    half = size / 2.0
    x1 = int(max(0, (x_center - half) * w))
    y1 = int(max(0, (y_center - half) * h))
    x2 = int(min(w, (x_center + half) * w))
    y2 = int(min(h, (y_center + half) * h))

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError(
            f"Empty crop region ({x1},{y1})-({x2},{y2}) for image {w}x{h}"
        )

    if output_size is not None:
        crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)

    logger.debug(
        "Extracted set symbol from %s: region (%d,%d)-(%d,%d) of %dx%d",
        image_path.name, x1, y1, x2, y2, w, h,
    )
    return crop
