"""Image format utilities — handles HEIC/HEIF from iPhones."""

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_heif_registered = False


def _register_heif_once():
    """Register the pillow-heif opener exactly once."""
    global _heif_registered
    if not _heif_registered:
        import pillow_heif
        pillow_heif.register_heif_opener()
        _heif_registered = True


def ensure_compatible(image_path: str) -> str:
    """Convert HEIC/HEIF images to JPEG. Returns path to compatible image.

    If already JPG/PNG/WebP, returns the original path unchanged.
    If HEIC/HEIF, converts to JPEG and returns the new path.

    Thread-safe: uses atomic write-and-rename so concurrent callers
    never see a partially-written output file.
    """
    path = Path(image_path)
    if path.suffix.lower() not in (".heic", ".heif"):
        return str(path)

    output = path.with_suffix(".jpg")
    if output.exists():
        return str(output)

    # Try pillow-heif first (pip install pillow-heif)
    try:
        _register_heif_once()
        from PIL import Image
        img = Image.open(path)
        _atomic_save_jpeg(img, output)
        logger.info("Converted %s -> %s (pillow-heif)", path.name, output.name)
        return str(output)
    except ImportError:
        logger.debug("pillow-heif not installed, trying ImageMagick")
    except Exception as e:
        logger.debug("pillow-heif failed: %s, trying ImageMagick", e)

    # Fallback: use ImageMagick if available
    try:
        import subprocess
        # Write to a temp file, then atomically rename
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg", dir=str(output.parent))
        os.close(fd)
        try:
            subprocess.run(
                ["convert", str(path), tmp_path],
                check=True, capture_output=True,
            )
            os.replace(tmp_path, str(output))
            logger.info("Converted %s -> %s (ImageMagick)", path.name, output.name)
            return str(output)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except FileNotFoundError:
        logger.debug("ImageMagick 'convert' not found")
    except subprocess.CalledProcessError as e:
        logger.debug("ImageMagick conversion failed: %s", e)

    logger.warning(
        "Cannot convert HEIC: install pillow-heif (pip install pillow-heif) "
        "or ImageMagick (apt install imagemagick)"
    )
    return str(path)


def _atomic_save_jpeg(img, output: Path):
    """Save a PIL Image to JPEG via a temp file + atomic rename."""
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg", dir=str(output.parent))
    os.close(fd)
    try:
        img.save(tmp_path, "JPEG", quality=95)
        os.replace(tmp_path, str(output))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
