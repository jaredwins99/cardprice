#!/usr/bin/env python3
"""Generate professional variant overlay icons for Pokemon card reference images.

Creates 128x128 RGBA PNGs saved to data/variant_icons/.
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

_BOLD_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_REGULAR_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    path = _BOLD_FONT if bold else _REGULAR_FONT
    return ImageFont.truetype(path, size)


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    """Return (width, height) of rendered text."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _center_text(draw, text, font, cx, cy, fill):
    """Draw text centered on (cx, cy)."""
    w, h = _text_bbox(draw, text, font)
    bbox = draw.textbbox((0, 0), text, font=font)
    x_off = bbox[0]
    y_off = bbox[1]
    draw.text((cx - w / 2 - x_off, cy - h / 2 - y_off), text, font=font, fill=fill)


def _draw_star(draw, cx, cy, outer_r, inner_r, points=5, fill=(0, 0, 0, 230)):
    """Draw a filled star polygon."""
    coords = []
    for i in range(points * 2):
        angle = math.pi / 2 + i * math.pi / points
        r = outer_r if i % 2 == 0 else inner_r
        coords.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
    draw.polygon(coords, fill=fill)


def _draw_circle_text(draw, text, font, cx, cy, radius, fill, start_angle=-90):
    """Draw text along a circular arc, centered on start_angle."""
    # Compute total angular span
    char_angle = 18  # degrees per character
    total = char_angle * (len(text) - 1)
    angle_start = start_angle - total / 2

    for i, ch in enumerate(text):
        angle_deg = angle_start + i * char_angle
        angle_rad = math.radians(angle_deg)
        x = cx + radius * math.cos(angle_rad)
        y = cy + radius * math.sin(angle_rad)
        # Draw each character centered at the position
        bbox = draw.textbbox((0, 0), ch, font=font)
        cw = bbox[2] - bbox[0]
        ch_h = bbox[3] - bbox[1]
        draw.text((x - cw / 2, y - ch_h / 2), ch, font=font, fill=fill)


# ---------------------------------------------------------------------------
# Gold / metallic color palette
# ---------------------------------------------------------------------------

GOLD = (218, 175, 50, 240)
GOLD_LIGHT = (245, 215, 100, 220)
GOLD_DARK = (170, 130, 20, 250)
SILVER = (190, 195, 205, 240)
SILVER_LIGHT = (220, 225, 235, 220)
SILVER_DARK = (140, 145, 155, 250)
BLACK = (30, 30, 30, 240)
WHITE = (255, 255, 255, 240)
RED = (210, 45, 45, 240)
RED_DARK = (160, 25, 25, 250)
BLUE = (50, 100, 200, 240)
BLUE_DARK = (30, 65, 150, 250)
PURPLE = (130, 60, 180, 240)
PURPLE_DARK = (90, 35, 130, 250)
GREY = (130, 135, 140, 240)
GREY_DARK = (90, 95, 100, 250)


# ---------------------------------------------------------------------------
# Icon generators
# ---------------------------------------------------------------------------

def create_1st_edition_icon(size=128) -> Image.Image:
    """1st Edition stamp: gold circle with '1' and 'EDITION' text."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size / 2, size / 2
    margin = size // 8

    # Outer circle — double ring for premium look
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        outline=GOLD_DARK, width=max(3, size // 32),
    )
    inner_gap = size // 20
    draw.ellipse(
        [margin + inner_gap, margin + inner_gap,
         size - margin - inner_gap, size - margin - inner_gap],
        outline=GOLD, width=max(2, size // 48),
    )

    # Large "1" numeral
    f_big = _font(size // 3)
    _center_text(draw, "1", f_big, cx, cy - size // 14, GOLD)

    # "EDITION" below the 1
    f_small = _font(size // 10)
    _center_text(draw, "EDITION", f_small, cx, cy + size // 5, GOLD_DARK)

    # Small "ST" superscript next to the 1
    f_tiny = _font(size // 8)
    # position it to upper-right of the "1"
    one_w, one_h = _text_bbox(draw, "1", f_big)
    st_x = cx + one_w / 2 + size // 40
    st_y = cy - size // 14 - one_h / 3
    draw.text((st_x, st_y), "ST", font=f_tiny, fill=GOLD_LIGHT)

    return img


def create_reverse_holo_icon(size=128) -> Image.Image:
    """Reverse holo: silver rounded rectangle with 'RH' monogram."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = size // 6
    r = size // 6  # corner radius

    # Background rounded rect with subtle gradient simulation
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=r, fill=(200, 210, 225, 160), outline=SILVER_DARK, width=max(2, size // 40),
    )

    # Diagonal shimmer line
    draw.line(
        [(margin + size // 5, size - margin),
         (size - margin - size // 5, margin)],
        fill=(255, 255, 255, 80), width=size // 8,
    )

    # "RH" text
    f = _font(size // 3)
    cx, cy = size / 2, size / 2
    _center_text(draw, "RH", f, cx, cy, SILVER_DARK)

    # Thin inner border
    inner = size // 20
    draw.rounded_rectangle(
        [margin + inner, margin + inner,
         size - margin - inner, size - margin - inner],
        radius=r - inner // 2, outline=SILVER, width=max(1, size // 64),
    )

    return img


def create_promo_icon(size=128) -> Image.Image:
    """Promo: black star with gold outline, like the official promo symbol."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size / 2, size / 2
    outer_r = size * 0.40
    inner_r = outer_r * 0.42

    # Shadow star slightly offset
    _draw_star(draw, cx + 2, cy + 2, outer_r, inner_r,
               fill=(0, 0, 0, 100))

    # Main black star
    _draw_star(draw, cx, cy, outer_r, inner_r, fill=BLACK)

    # Gold outline star (slightly larger, drawn under would be better but
    # we just draw a thin outline on top)
    star_coords = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        r = outer_r if i % 2 == 0 else inner_r
        star_coords.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
    draw.polygon(star_coords, outline=GOLD, width=max(2, size // 48))

    # "PROMO" text at bottom
    f_small = _font(size // 10)
    _center_text(draw, "PROMO", f_small, cx, cy + size * 0.38, GOLD_DARK)

    return img


def create_stamped_icon(size=128) -> Image.Image:
    """Stamped/set stamp indicator: purple hexagonal badge."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size / 2, size / 2
    r = size * 0.38

    # Hexagon
    hex_pts = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3
        hex_pts.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
    draw.polygon(hex_pts, fill=(130, 60, 180, 170), outline=PURPLE_DARK, width=max(2, size // 40))

    # Inner hexagon border
    r_inner = r * 0.82
    hex_inner = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3
        hex_inner.append((cx + r_inner * math.cos(angle), cy - r_inner * math.sin(angle)))
    draw.polygon(hex_inner, outline=PURPLE, width=max(1, size // 64))

    # "SET" text on top line, star below
    f = _font(size // 6)
    _center_text(draw, "SET", f, cx, cy - size // 10, WHITE)

    # Small star beneath
    _draw_star(draw, cx, cy + size // 7, size // 10, size // 22,
               fill=(255, 255, 255, 200))

    return img


def create_prerelease_icon(size=128) -> Image.Image:
    """Prerelease: blue badge with 'PRE' and 'RELEASE' text."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size / 2, size / 2
    margin = size // 7

    # Rounded rectangle background
    draw.rounded_rectangle(
        [margin, margin + size // 8, size - margin, size - margin - size // 8],
        radius=size // 8, fill=(50, 100, 200, 170), outline=BLUE_DARK,
        width=max(2, size // 40),
    )

    # "PRE" large
    f_big = _font(size // 4)
    _center_text(draw, "PRE", f_big, cx, cy - size // 10, WHITE)

    # "RELEASE" smaller below
    f_small = _font(size // 9)
    _center_text(draw, "RELEASE", f_small, cx, cy + size // 7, (200, 220, 255, 230))

    # Thin inner border
    inner = size // 18
    draw.rounded_rectangle(
        [margin + inner, margin + size // 8 + inner,
         size - margin - inner, size - margin - size // 8 - inner],
        radius=size // 10, outline=(150, 180, 255, 150), width=max(1, size // 64),
    )

    return img


def create_staff_icon(size=128) -> Image.Image:
    """Staff promo: gold shield with 'STAFF' text."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size / 2, size / 2

    # Shield shape (pointed bottom)
    top = size // 7
    side = size // 6
    bottom_y = size - size // 8
    mid_y = cy + size // 10

    shield = [
        (side, top),                     # top-left
        (size - side, top),              # top-right
        (size - side, mid_y),            # right shoulder
        (cx, bottom_y),                  # bottom point
        (side, mid_y),                   # left shoulder
    ]
    draw.polygon(shield, fill=(218, 175, 50, 170), outline=GOLD_DARK,
                 width=max(2, size // 36))

    # Inner shield border
    scale = 0.85
    shield_inner = [
        (cx + (x - cx) * scale, cy + (y - cy) * scale * 0.95)
        for x, y in shield
    ]
    draw.polygon(shield_inner, outline=GOLD_LIGHT, width=max(1, size // 56))

    # "STAFF" text
    f = _font(size // 6)
    _center_text(draw, "STAFF", f, cx, cy - size // 16, (60, 30, 0, 240))

    # Small star below text
    _draw_star(draw, cx, cy + size // 6, size // 12, size // 26,
               fill=(60, 30, 0, 200))

    return img


def create_shadowless_icon(size=128) -> Image.Image:
    """Shadowless: grey circle with 'SL' and dashed outline suggesting no shadow."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size / 2, size / 2
    margin = size // 6

    # Outer circle (solid)
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=(160, 165, 175, 140), outline=GREY_DARK, width=max(2, size // 40),
    )

    # Dashed inner ring (simulate with arc segments)
    inner_m = margin + size // 14
    segments = 12
    arc_span = 360 / segments
    gap = arc_span * 0.3
    for i in range(segments):
        start = i * arc_span + gap / 2
        end = (i + 1) * arc_span - gap / 2
        draw.arc(
            [inner_m, inner_m, size - inner_m, size - inner_m],
            start, end, fill=GREY, width=max(1, size // 56),
        )

    # "SL" text
    f = _font(size // 3)
    _center_text(draw, "SL", f, cx, cy - size // 14, GREY_DARK)

    # "SHADOWLESS" tiny text below
    f_tiny = _font(size // 14)
    _center_text(draw, "SHADOWLESS", f_tiny, cx, cy + size // 5, (90, 95, 100, 200))

    return img


def _draw_pokeball(draw, cx, cy, radius, top_color, bottom_color=(255, 255, 255, 230),
                   outline_color=(40, 40, 40, 240), band_width_ratio=0.12):
    """Draw a Pokeball at given center and radius."""
    r = radius
    lw = max(2, int(r * 0.08))

    # Bottom half (white)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=bottom_color,
                 outline=outline_color, width=lw)

    # Top half (colored) — draw a filled chord for the upper half
    draw.pieslice([cx - r, cy - r, cx + r, cy + r], 180, 360,
                  fill=top_color, outline=outline_color, width=lw)

    # Horizontal band across the middle
    band_h = max(3, int(r * band_width_ratio))
    draw.rectangle([cx - r, cy - band_h, cx + r, cy + band_h],
                   fill=outline_color)

    # Center button
    btn_r = max(3, int(r * 0.22))
    draw.ellipse([cx - btn_r, cy - btn_r, cx + btn_r, cy + btn_r],
                 fill=(255, 255, 255, 240), outline=outline_color,
                 width=max(2, lw))
    inner_btn = max(2, int(btn_r * 0.55))
    draw.ellipse([cx - inner_btn, cy - inner_btn, cx + inner_btn, cy + inner_btn],
                 fill=outline_color)

    # Redraw outer circle outline cleanly
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=outline_color, width=lw)


def create_pokemon_center_icon(size=128) -> Image.Image:
    """Pokemon Center exclusive: red Pokeball with 'PC' text."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size / 2, size / 2
    r = int(size * 0.38)

    _draw_pokeball(draw, cx, cy, r, top_color=RED)

    # "PC" text below the ball
    f = _font(size // 8)
    _center_text(draw, "PC", f, cx, cy + r + size // 10, RED_DARK)

    return img


def create_build_battle_icon(size=128) -> Image.Image:
    """Build & Battle / prerelease kit: Pokeball with construction motif."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size / 2, size / 2 - size // 14
    r = int(size * 0.30)

    _draw_pokeball(draw, cx, cy, r, top_color=RED)

    # Crossed tools (wrench + hammer) behind the ball — simplified as an X
    tool_len = size // 4
    tool_w = max(2, size // 24)
    arm = tool_len * 0.7
    # Upper-left to lower-right
    draw.line([(cx - arm, cy - arm), (cx + arm, cy + arm)],
              fill=GREY_DARK, width=tool_w)
    # Upper-right to lower-left
    draw.line([(cx + arm, cy - arm), (cx - arm, cy + arm)],
              fill=GREY_DARK, width=tool_w)

    # Re-draw pokeball on top
    _draw_pokeball(draw, cx, cy, r, top_color=RED)

    # "BUILD" and "BATTLE" text
    f = _font(size // 10)
    text_y = cy + r + size // 14
    _center_text(draw, "BUILD &", f, cx, text_y, RED_DARK)
    _center_text(draw, "BATTLE", f, cx, text_y + size // 8, RED_DARK)

    return img


# ---------------------------------------------------------------------------
# Registry & main
# ---------------------------------------------------------------------------

ICONS = {
    "1st_edition": create_1st_edition_icon,
    "reverse_holo": create_reverse_holo_icon,
    "promo": create_promo_icon,
    "stamped": create_stamped_icon,
    "prerelease": create_prerelease_icon,
    "staff": create_staff_icon,
    "shadowless": create_shadowless_icon,
    "pokemon_center": create_pokemon_center_icon,
    "build_battle": create_build_battle_icon,
}


def generate_all(output_dir: str = "data/variant_icons", size: int = 128):
    """Generate all variant icons and save to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, func in ICONS.items():
        icon = func(size)
        path = out / f"{name}.png"
        icon.save(str(path), "PNG")
        print(f"  Saved {path} ({icon.size[0]}x{icon.size[1]})")

    print(f"\nGenerated {len(ICONS)} icons in {out}/")


if __name__ == "__main__":
    generate_all()
