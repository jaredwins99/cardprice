"""Image format utilities — handles HEIC/HEIF from iPhones."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_compatible(image_path: str) -> str:
    """Convert HEIC/HEIF images to JPEG. Returns path to compatible image.

    If already JPG/PNG/WebP, returns the original path unchanged.
    If HEIC/HEIF, converts to JPEG and returns the new path.
    """
    path = Path(image_path)
    if path.suffix.lower() not in (".heic", ".heif"):
        return str(path)

    output = path.with_suffix(".jpg")
    if output.exists():
        return str(output)

    try:
        # Try pillow-heif first (pip install pillow-heif)
        import pillow_heif
        pillow_heif.register_heif_opener()
        from PIL import Image
        img = Image.open(path)
        img.save(output, "JPEG", quality=95)
        logger.info("Converted %s -> %s", path.name, output.name)
        return str(output)
    except ImportError:
        pass

    try:
        # Fallback: use ImageMagick if available
        import subprocess
        subprocess.run(
            ["convert", str(path), str(output)],
            check=True, capture_output=True,
        )
        logger.info("Converted %s -> %s (ImageMagick)", path.name, output.name)
        return str(output)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    logger.warning(
        "Cannot convert HEIC: install pillow-heif (pip install pillow-heif) "
        "or ImageMagick (apt install imagemagick)"
    )
    return str(path)
