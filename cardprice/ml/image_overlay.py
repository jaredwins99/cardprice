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

import json

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stamp set name lookup (card_id → set name for EX-era stamp overlay)
# ---------------------------------------------------------------------------
_STAMP_MAP = None

def _get_stamp_set_name(card_id: str) -> str | None:
    """Get the set name for stamp overlay from card_id (e.g. 'ex15-72/normal' → 'Dragon Frontiers')."""
    global _STAMP_MAP
    if _STAMP_MAP is None:
        p = Path(__file__).resolve().parent.parent.parent / "data" / "stamp_logos" / "stamp_map.json"
        try:
            _STAMP_MAP = json.loads(p.read_text())
        except Exception:
            _STAMP_MAP = {}
    set_id = card_id.split("-")[0] if card_id else ""
    entry = _STAMP_MAP.get(set_id)
    if isinstance(entry, dict):
        return entry.get("name", entry.get("full_name"))
    elif isinstance(entry, str):
        return entry
    return None

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
        # Left side, just below artwork frame
        "stamp_pos": (0.06, 0.58),
        "stamp_text": "1ST EDITION",
        "stamp_rotation": 0,
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
        # Replaces set symbol, bottom-right area
        "stamp_pos": (0.78, 0.89),
        "stamp_icon": "_draw_star_icon",
        "stamp_rotation": 0,
    },
    "black_star_promo": {
        "label": "",
        "bg": (30, 30, 30, 230),
        "fg": (255, 215, 0, 255),
        "border": (40, 40, 40),
        "icon_fn": "_draw_star_icon",
        "stamp_pos": (0.78, 0.89),
        "stamp_icon": "_draw_star_icon",
        "stamp_rotation": 0,
    },
    "promo_stamp": {
        "label": "",
        "bg": (30, 30, 30, 230),
        "fg": (255, 215, 0, 255),
        "border": (40, 40, 40),
        "icon_fn": "_draw_star_icon",
        "stamp_pos": (0.78, 0.89),
        "stamp_icon": "_draw_star_icon",
        "stamp_rotation": 0,
    },
    "modern_promo": {
        "label": "",
        "bg": (30, 30, 30, 230),
        "fg": (255, 215, 0, 255),
        "border": (40, 40, 40),
        "icon_fn": "_draw_star_icon",
        "stamp_pos": (0.78, 0.89),
        "stamp_icon": "_draw_star_icon",
        "stamp_rotation": 0,
    },
    "stamped": {
        "label": "STAMP",
        "bg": (128, 60, 180, 220),       # purple
        "fg": (255, 255, 255, 255),
        "border": (128, 60, 180),
        # Center of card text area
        "stamp_pos": (0.50, 0.62),
        "stamp_text": "STAMPED",
        "stamp_rotation": -15,
    },
    "ex_set_stamp": {
        "label": "EX",
        "bg": (218, 165, 32, 220),          # gold
        "fg": (255, 255, 255, 255),
        "border": (218, 165, 32),            # gold
        # Bottom-right of artwork area — real stamp position
        "stamp_pos": (0.82, 0.50),
        "stamp_text": "STAMPED",
        "stamp_rotation": -15,
        "use_set_name": True,  # flag: replace stamp_text with actual set name
    },
    "prerelease": {
        "label": "PR",
        "bg": (40, 100, 200, 220),       # blue
        "fg": (255, 255, 255, 255),
        "border": (40, 100, 200),
        # Bottom-right of artwork area
        "stamp_pos": (0.72, 0.50),
        "stamp_text": "PRERELEASE",
        "stamp_rotation": -20,
    },
    "staff": {
        "label": "STAFF",
        "bg": (218, 165, 32, 230),       # gold bg
        "fg": (30, 60, 140, 255),        # blue text
        "border": (218, 165, 32),
        # Same position as prerelease but with STAFF text
        "stamp_pos": (0.72, 0.50),
        "stamp_text": "STAFF",
        "stamp_rotation": -20,
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
        "stamp_pos": (0.50, 0.62),
        "stamp_text": "GREY\nSTAMP",
        "stamp_rotation": -10,
    },
    "pokemon_center": {
        "label": "PC",
        "bg": (200, 30, 30, 220),        # red
        "fg": (255, 255, 255, 255),
        "border": (200, 30, 30),
        "icon_fn": "_draw_pokeball_icon",
        "stamp_pos": (0.50, 0.62),
        "stamp_icon": "_draw_pokeball_icon",
        "stamp_rotation": 0,
    },
    "build_battle": {
        "label": "B&B",
        "bg": (200, 40, 40, 220),        # red
        "fg": (255, 255, 255, 255),
        "border": (200, 40, 40),
        "stamp_pos": (0.72, 0.50),
        "stamp_text": "BUILD &\nBATTLE",
        "stamp_rotation": -15,
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

def _draw_positioned_stamp(
    overlay: Image.Image,
    draw: ImageDraw.ImageDraw,
    style: dict,
    card_w: int,
    card_h: int,
    border_w: int,
    card_id: str | None = None,
) -> None:
    """Draw a stamp/watermark at its real position on the card.

    Uses stamp_pos (fractional x, y on the card), stamp_text or stamp_icon,
    and optional stamp_rotation.  For EX-era stamps with a logo PNG file,
    pastes the actual set logo image.
    """
    fx, fy = style["stamp_pos"]
    rotation = style.get("stamp_rotation", 0)
    bg = style["bg"]
    fg = style["fg"]

    # Position relative to the card image (offset by border)
    cx = int(fx * card_w) + border_w
    cy = int(fy * card_h) + border_w

    # For EX-era stamps: overlay the official stylized set name logo in gold.
    # These are the actual logo designs used on real stamped reverse holo cards,
    # rendered as gold/monochrome foil in the bottom-right of the artwork area.
    if style.get("use_set_name") and card_id:
        set_id = card_id.split("-")[0] if card_id else ""
        # Map set_id to stylized logo filename
        _STYLIZED_LOGOS = {
            "ex1": "ex01_ruby_sapphire_logo.png", "ex2": "ex02_sandstorm_logo.png",
            "ex3": "ex03_dragon_logo.png", "ex4": "ex04_team_magma_vs_team_aqua_logo.png",
            "ex5": "ex05_hidden_legends_logo.png", "ex6": "ex06_firered_leafgreen_logo.png",
            "ex7": "ex07_team_rocket_returns_logo.png", "ex8": "ex08_deoxys_logo.png",
            "ex9": "ex09_emerald_logo.png", "ex10": "ex10_unseen_forces_logo.png",
            "ex11": "ex11_delta_species_logo.png", "ex12": "ex12_legend_maker_logo.png",
            "ex13": "ex13_holon_phantoms_logo.png", "ex14": "ex14_crystal_guardians_logo.png",
            "ex15": "ex15_dragon_frontiers_logo.png", "ex16": "ex16_power_keepers_logo.png",
        }
        logo_name = _STYLIZED_LOGOS.get(set_id)
        if logo_name:
            logo_dir = Path(__file__).resolve().parent.parent.parent / "data" / "stamp_logos" / "stylized"
            logo_path = logo_dir / logo_name
            if logo_path.exists():
                try:
                    logo = Image.open(logo_path).convert("RGBA")
                    # Scale to fit ~25% of card width, maintaining aspect ratio
                    target_w = int(card_w * 0.25)
                    scale = target_w / logo.width
                    target_h = int(logo.height * scale)
                    logo = logo.resize((target_w, target_h), Image.LANCZOS)
                    # Tint to gold monochrome (like real foil stamp)
                    r, g, b, a = logo.split()
                    # Convert to grayscale luminance, then apply gold tint
                    import numpy as np
                    arr = np.array(logo)
                    gray = (0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]).astype(np.uint8)
                    gold_arr = np.zeros_like(arr)
                    gold_arr[:,:,0] = (gray * 218 / 255).astype(np.uint8)  # R
                    gold_arr[:,:,1] = (gray * 185 / 255).astype(np.uint8)  # G
                    gold_arr[:,:,2] = (gray * 52 / 255).astype(np.uint8)   # B
                    gold_arr[:,:,3] = (arr[:,:,3] * 0.6).astype(np.uint8)  # alpha at 60%
                    logo = Image.fromarray(gold_arr, 'RGBA')
                    # Paste at stamp position (bottom-right of artwork)
                    paste_x = cx - target_w // 2
                    paste_y = cy - target_h // 2
                    overlay.paste(logo, (paste_x, paste_y), logo)
                    return
                except Exception as e:
                    logger.warning("Failed to load stylized stamp logo %s: %s", logo_path, e)
        return  # skip if no logo found — no overlay is better than wrong overlay

    stamp_text = style.get("stamp_text")
    stamp_icon_key = style.get("stamp_icon")

    if stamp_text:
        # Text stamp — rendered as a rotated watermark
        font_size = max(12, int(card_h * 0.045))
        font = _get_font(font_size)

        # Render text onto a temporary image for rotation
        # Generous canvas to avoid clipping after rotation
        tmp_size = int(card_w * 0.6)
        tmp = Image.new("RGBA", (tmp_size, tmp_size), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp)
        tcx, tcy = tmp_size // 2, tmp_size // 2

        # Measure text for background pill
        lines = stamp_text.split("\n")
        line_height = font_size + 4
        total_text_h = line_height * len(lines)
        max_text_w = 0
        for line in lines:
            tb = font.getbbox(line)
            max_text_w = max(max_text_w, tb[2] - tb[0])

        pad = int(font_size * 0.5)
        pill_x0 = tcx - max_text_w // 2 - pad
        pill_y0 = tcy - total_text_h // 2 - pad
        pill_x1 = tcx + max_text_w // 2 + pad
        pill_y1 = tcy + total_text_h // 2 + pad
        pill_r = int(font_size * 0.4)

        # Semi-transparent background
        stamp_bg = (bg[0], bg[1], bg[2], min(bg[3], 180))
        tmp_draw.rounded_rectangle(
            [pill_x0, pill_y0, pill_x1, pill_y1],
            radius=pill_r, fill=stamp_bg,
        )
        # Border
        stamp_outline = (fg[0], fg[1], fg[2], 120)
        tmp_draw.rounded_rectangle(
            [pill_x0, pill_y0, pill_x1, pill_y1],
            radius=pill_r, outline=stamp_outline,
            width=max(1, font_size // 12),
        )

        # Draw text lines centered
        for i, line in enumerate(lines):
            ly = tcy - total_text_h // 2 + i * line_height + line_height // 2
            tmp_draw.text((tcx, ly), line, fill=fg, font=font, anchor="mm")

        if rotation:
            tmp = tmp.rotate(rotation, resample=Image.BICUBIC, expand=False)

        # Paste onto overlay centered at (cx, cy)
        paste_x = cx - tmp_size // 2
        paste_y = cy - tmp_size // 2
        overlay.paste(tmp, (paste_x, paste_y), tmp)

    elif stamp_icon_key:
        # Icon stamp (promo star, pokeball, etc.)
        icon_fn = _ICON_FNS.get(stamp_icon_key)
        if icon_fn:
            icon_r = max(8, int(card_h * 0.04))
            # Draw a circular background
            stamp_bg = (bg[0], bg[1], bg[2], min(bg[3], 180))
            draw.ellipse(
                [cx - icon_r - 4, cy - icon_r - 4,
                 cx + icon_r + 4, cy + icon_r + 4],
                fill=stamp_bg,
            )
            icon_fn(draw, (cx - icon_r, cy - icon_r,
                           cx + icon_r, cy + icon_r), fg)


def overlay_variant_indicator(
    image_source: Union[str, bytes, Image.Image],
    variants: Sequence[str],
    *,
    card_id: str | None = None,
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
    # 2. Positioned stamp overlays on the card image itself
    # ------------------------------------------------------------------
    overlay = Image.new("RGBA", bordered.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bimg_w, bimg_h = bordered.size
    margin = int(bimg_w * 0.03)

    # Variants with stamp_pos get rendered at their real card position.
    # Variants without stamp_pos get a small corner badge (bottom-right).
    badge_variants = []

    for variant in active:
        style = _VARIANT_STYLES.get(variant, _DEFAULT_STYLE)

        if "stamp_pos" in style:
            # For EX-era stamps, replace generic text with actual set name
            if style.get("use_set_name") and card_id:
                set_name = _get_stamp_set_name(card_id)
                if set_name:
                    style = dict(style)
                    style["stamp_text"] = set_name.upper()
            _draw_positioned_stamp(overlay, draw, style, w, h, bw, card_id=card_id)
        else:
            badge_variants.append(variant)

    # Corner badges for variants without positioned stamps (holo, reverse, etc.)
    if badge_variants:
        badge_h = max(16, int(h * badge_scale))
        pad_x = int(badge_h * 0.45)
        pad_y = int(badge_h * 0.18)
        badge_r = badge_h // 3
        font_size = int(badge_h * 0.48)
        icon_size = int(badge_h * 0.55)
        font = _get_font(font_size)

        y_cursor = bimg_h - margin

        for variant in reversed(badge_variants):
            style = _VARIANT_STYLES.get(variant, _DEFAULT_STYLE)
            label = style.get("label") or variant.upper().replace("_", " ")
            bg = style["bg"]
            fg = style["fg"]

            icon_fn_key = style.get("icon_fn")
            icon_fn = _ICON_FNS.get(icon_fn_key) if icon_fn_key else None

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

            draw.rounded_rectangle([x0, y0, x1, y1], radius=badge_r, fill=bg)
            outline_a = (min(255, fg[0]), min(255, fg[1]), min(255, fg[2]), 80)
            draw.rounded_rectangle(
                [x0, y0, x1, y1], radius=badge_r,
                outline=outline_a, width=max(1, badge_h // 22),
            )

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

def apply_variant_overlay(image_data: bytes, variants: list[str], card_id: str | None = None) -> bytes:
    """Apply variant indicator badges to a card image.

    Legacy wrapper around :func:`overlay_variant_indicator`.  Accepts raw
    image bytes and a list of variant strings, returns PNG bytes.
    """
    if not variants:
        return image_data
    try:
        return overlay_variant_indicator(image_data, variants, card_id=card_id)
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
