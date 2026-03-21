"""Overlay variant indicators onto card reference images.

Renders professional-looking badges and coloured borders on reference card
images to indicate detected physical variants (1st Edition, Reverse Holo,
Promo, etc.).  Inspired by grading-company slab aesthetics.

Visual elements:
  1. Thin coloured border around the entire card image.
  2. Small icon/label badges in the bottom-right corner, stacked upward
     when multiple variants are present.

Uses Pillow (PIL) for compositing.  All drawing is vector-based (no
raster assets required).

Usage::

    from cardprice.ml.image_overlay import overlay_variant_indicator

    png_bytes = overlay_variant_indicator(
        "data/card_images/base1/base1-4_normal.png",
        ["1st_edition"],
    )

Legacy compatibility wrapper ``apply_variant_overlay`` is retained for
existing call-sites.
"""

from __future__ import annotations

import io
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Union

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variant visual definitions
# ---------------------------------------------------------------------------
# Each variant maps to:
#   label   -- short text rendered inside the badge
#   bg      -- badge background colour (RGBA)
#   fg      -- badge text colour (RGBA)
#   border  -- thin border colour around the entire card (RGB)
#   icon_fn -- optional key into _ICON_FNS for a small drawn icon

_VARIANT_STYLES: dict[str, dict] = {
    "1st_edition": {
        "label": "1ST",
        "bg": (218, 165, 32, 230),       # gold
        "fg": (30, 30, 30, 255),
        "border": (218, 165, 32),
        "icon_fn": "_draw_1st_edition_icon",
    },
    "reverse_holo": {
        "label": "RH",
        "bg": (180, 190, 210, 220),      # silver
        "fg": (20, 20, 40, 255),
        "border": (160, 170, 195),
    },
    "reverse_holofoil": {
        "label": "RH",
        "bg": (180, 190, 210, 220),
        "fg": (20, 20, 40, 255),
        "border": (160, 170, 195),
    },
    "holofoil": {
        "label": "HOLO",
        "bg": (100, 200, 255, 200),      # light blue
        "fg": (10, 10, 40, 255),
        "border": (100, 180, 240),
    },
    "promo": {
        "label": "",
        "bg": (30, 30, 30, 230),
        "fg": (255, 215, 0, 255),        # gold star on black
        "border": (40, 40, 40),
        "icon_fn": "_draw_star_icon",
    },
    "black_star_promo": {
        "label": "",
        "bg": (30, 30, 30, 230),
        "fg": (255, 215, 0, 255),
        "border": (40, 40, 40),
        "icon_fn": "_draw_star_icon",
    },
    "promo_stamp": {
        "label": "",
        "bg": (30, 30, 30, 230),
        "fg": (255, 215, 0, 255),
        "border": (40, 40, 40),
        "icon_fn": "_draw_star_icon",
    },
    "modern_promo": {
        "label": "",
        "bg": (30, 30, 30, 230),
        "fg": (255, 215, 0, 255),
        "border": (40, 40, 40),
        "icon_fn": "_draw_star_icon",
    },
    "stamped": {
        "label": "STAMP",
        "bg": (128, 60, 180, 220),       # purple
        "fg": (255, 255, 255, 255),
        "border": (128, 60, 180),
    },
    "ex_set_stamp": {
        "label": "EX",
        "bg": (128, 60, 180, 220),
        "fg": (255, 255, 255, 255),
        "border": (128, 60, 180),
    },
    "prerelease": {
        "label": "PR",
        "bg": (40, 100, 200, 220),       # blue
        "fg": (255, 255, 255, 255),
        "border": (40, 100, 200),
    },
    "staff": {
        "label": "STAFF",
        "bg": (218, 165, 32, 230),       # gold bg
        "fg": (30, 60, 140, 255),        # blue text
        "border": (218, 165, 32),
    },
    "shadowless": {
        "label": "SL",
        "bg": (140, 140, 150, 200),      # grey
        "fg": (255, 255, 255, 255),
        "border": (140, 140, 150),
    },
    "grey_stamp": {
        "label": "GS",
        "bg": (140, 140, 140, 200),
        "fg": (255, 255, 255, 255),
        "border": (140, 140, 140),
    },
    "pokemon_center": {
        "label": "PC",
        "bg": (200, 30, 30, 220),        # red
        "fg": (255, 255, 255, 255),
        "border": (200, 30, 30),
        "icon_fn": "_draw_pokeball_icon",
    },
    "build_battle": {
        "label": "B&B",
        "bg": (200, 40, 40, 220),        # red
        "fg": (255, 255, 255, 255),
        "border": (200, 40, 40),
    },
    "full_art": {
        "label": "FA",
        "bg": (60, 60, 60, 200),
        "fg": (255, 200, 50, 255),
        "border": (80, 80, 80),
    },
    "gold": {
        "label": "GOLD",
        "bg": (200, 160, 30, 230),
        "fg": (40, 20, 0, 255),
        "border": (200, 160, 30),
    },
    "rainbow_rare": {
        "label": "RR",
        "bg": (180, 80, 200, 220),       # purple-pink
        "fg": (255, 255, 255, 255),
        "border": (180, 80, 200),
    },
}

_DEFAULT_STYLE = {
    "label": "?",
    "bg": (100, 100, 100, 200),
    "fg": (255, 255, 255, 255),
    "border": (100, 100, 100),
}


# ---------------------------------------------------------------------------
# Icon drawing helpers
# ---------------------------------------------------------------------------

def _draw_1st_edition_icon(
    draw: ImageDraw.ImageDraw, bbox: tuple[int, int, int, int], fg: tuple,
) -> None:
    """Circle with '1' -- mimics the 1st Edition stamp."""
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    r = min(x1 - x0, y1 - y0) // 2 - 1
    if r < 2:
        return
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        outline=fg, width=max(1, r // 5),
    )
    font = _get_font(int(r * 1.3))
    draw.text((cx, cy), "1", fill=fg, font=font, anchor="mm")


def _draw_star_icon(
    draw: ImageDraw.ImageDraw, bbox: tuple[int, int, int, int], fg: tuple,
) -> None:
    """Five-pointed star (promo indicator)."""
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    r_outer = min(x1 - x0, y1 - y0) / 2 - 1
    r_inner = r_outer * 0.40
    points = []
    for i in range(10):
        angle = math.radians(i * 36 - 90)
        r = r_outer if i % 2 == 0 else r_inner
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(points, fill=fg)


def _draw_pokeball_icon(
    draw: ImageDraw.ImageDraw, bbox: tuple[int, int, int, int], fg: tuple,
) -> None:
    """Simplified pokeball icon."""
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    r = min(x1 - x0, y1 - y0) // 2 - 1
    if r < 3:
        return
    lw = max(1, r // 6)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fg, width=lw)
    draw.line([(cx - r, cy), (cx + r, cy)], fill=fg, width=lw)
    cr = max(2, r // 3)
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], outline=fg, width=lw)
    draw.ellipse(
        [cx - cr + lw, cy - cr + lw, cx + cr - lw, cy + cr - lw],
        fill=(255, 255, 255, 255),
    )


_ICON_FNS = {
    "_draw_1st_edition_icon": _draw_1st_edition_icon,
    "_draw_star_icon": _draw_star_icon,
    "_draw_pokeball_icon": _draw_pokeball_icon,
}


# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a bold font at *size* px, falling back to Pillow default."""
    size = max(8, size)
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Core overlay renderer
# ---------------------------------------------------------------------------

def overlay_variant_indicator(
    image_source: Union[str, bytes, Image.Image],
    variants: Sequence[str],
    *,
    badge_scale: float = 0.14,
    border_width_frac: float = 0.012,
) -> bytes:
    """Add variant indicators to a card reference image.

    Renders:
      1. Thin coloured border around the entire image.
      2. Small badge(s) in the bottom-right corner (icon + label),
         stacked upward when multiple variants are present.

    Args:
        image_source: file path ``str``, raw image ``bytes``, or ``PIL.Image``.
        variants: detected variants, e.g. ``["1st_edition"]``.
        badge_scale: badge height as fraction of card height.
        border_width_frac: border thickness as fraction of card width.

    Returns:
        PNG-encoded image bytes with overlays applied.
    """
    # --- fast exit when nothing to draw ---
    active = [v for v in (variants or []) if v and v != "normal"]
    if not active:
        return _raw_bytes(image_source)

    # --- load image ---
    if isinstance(image_source, (str, Path)):
        img = Image.open(image_source).convert("RGBA")
    elif isinstance(image_source, bytes):
        img = Image.open(io.BytesIO(image_source)).convert("RGBA")
    else:
        img = image_source.convert("RGBA")

    w, h = img.size

    # ------------------------------------------------------------------
    # 1. Coloured border
    # ------------------------------------------------------------------
    first_style = _VARIANT_STYLES.get(active[0], _DEFAULT_STYLE)
    border_rgb = first_style["border"]
    bw = max(2, int(w * border_width_frac))

    bordered = Image.new("RGBA", (w + 2 * bw, h + 2 * bw), border_rgb + (255,))
    bordered.paste(img, (bw, bw))

    # Dual-colour inner stripe when 2+ variants
    if len(active) > 1:
        second_style = _VARIANT_STYLES.get(active[1], _DEFAULT_STYLE)
        inner_rgb = second_style["border"]
        inner_w = max(1, bw // 2)
        bd = ImageDraw.Draw(bordered)
        bd.rectangle(
            [bw - inner_w, bw - inner_w,
             w + bw + inner_w - 1, h + bw + inner_w - 1],
            outline=inner_rgb + (255,), width=inner_w,
        )

    # ------------------------------------------------------------------
    # 2. Badge overlays (bottom-right, stacked upward)
    # ------------------------------------------------------------------
    overlay = Image.new("RGBA", bordered.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bimg_w, bimg_h = bordered.size
    badge_h = max(16, int(h * badge_scale))
    pad_x = int(badge_h * 0.45)
    pad_y = int(badge_h * 0.18)
    badge_r = badge_h // 3
    font_size = int(badge_h * 0.48)
    icon_size = int(badge_h * 0.55)
    font = _get_font(font_size)
    margin = int(bimg_w * 0.03)

    y_cursor = bimg_h - margin  # bottom edge start

    for variant in reversed(active):  # stack bottom-up
        style = _VARIANT_STYLES.get(variant, _DEFAULT_STYLE)
        label = style.get("label") or variant.upper().replace("_", " ")
        bg = style["bg"]
        fg = style["fg"]

        icon_fn_key = style.get("icon_fn")
        icon_fn = _ICON_FNS.get(icon_fn_key) if icon_fn_key else None

        # Measure content width
        text_w = 0
        if label:
            tb = font.getbbox(label)
            text_w = tb[2] - tb[0]
        content_w = text_w
        if icon_fn:
            content_w += icon_size + (int(pad_x * 0.35) if label else 0)

        total_w = content_w + 2 * pad_x
        total_h = max(badge_h, font_size + 2 * pad_y + 4)

        x1 = bimg_w - margin
        x0 = x1 - total_w
        y1 = y_cursor
        y0 = y1 - total_h

        # Badge background
        draw.rounded_rectangle([x0, y0, x1, y1], radius=badge_r, fill=bg)

        # Subtle outline for definition against busy card art
        outline_a = (min(255, fg[0]), min(255, fg[1]), min(255, fg[2]), 80)
        draw.rounded_rectangle(
            [x0, y0, x1, y1], radius=badge_r,
            outline=outline_a, width=max(1, badge_h // 22),
        )

        # Draw icon + label
        cx = x0 + pad_x
        cy = (y0 + y1) // 2

        if icon_fn:
            iy0 = cy - icon_size // 2
            icon_fn(draw, (cx, iy0, cx + icon_size, iy0 + icon_size), fg)
            if label:
                cx += icon_size + int(pad_x * 0.35)

        if label:
            draw.text((cx, cy), label, fill=fg, font=font, anchor="lm")

        y_cursor = y0 - int(margin * 0.4)

    result = Image.alpha_composite(bordered, overlay).convert("RGB")

    buf = io.BytesIO()
    result.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Legacy API (retained for backward compatibility with existing call-sites)
# ---------------------------------------------------------------------------

def apply_variant_overlay(image_data: bytes, variants: list[str]) -> bytes:
    """Apply variant indicator badges to a card image.

    Legacy wrapper around :func:`overlay_variant_indicator`.  Accepts raw
    image bytes and a list of variant strings, returns PNG bytes.
    """
    if not variants:
        return image_data
    try:
        return overlay_variant_indicator(image_data, variants)
    except Exception:
        logger.warning("overlay_variant_indicator failed, returning original", exc_info=True)
        return image_data


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _raw_bytes(source: Union[str, bytes, Path, Image.Image]) -> bytes:
    """Return image as raw bytes without modification."""
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    buf = io.BytesIO()
    source.save(buf, format="PNG")
    return buf.getvalue()


def get_variant_border_color(variant: str) -> tuple[int, int, int] | None:
    """Return the RGB border colour for a variant, or ``None``."""
    style = _VARIANT_STYLES.get(variant)
    return style["border"] if style else None


def supported_variants() -> list[str]:
    """Return list of all variant names with defined visual styles."""
    return list(_VARIANT_STYLES.keys())
