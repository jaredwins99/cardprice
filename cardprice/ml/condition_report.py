"""Generate visual HTML condition reports for Pokemon card assessments.

Creates a self-contained HTML file with base64-encoded images, inline CSS,
and detailed breakdowns of each condition sub-score. No external dependencies
are required to view the output -- it works standalone in any browser.

Usage::

    from cardprice.ml.condition_report import generate_condition_report

    html = generate_condition_report(
        image_path="photo.jpg",
        card_id="base1-4/holofoil",
        assessment=assess_condition("photo.jpg", card_id="base1-4/holofoil"),
    )

    # Or save to file:
    generate_condition_report(
        "photo.jpg", "base1-4/holofoil", assessment, output_path="report.html"
    )
"""

import base64
import io
import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grade colors and styling
# ---------------------------------------------------------------------------

_GRADE_COLORS = {
    "NM":  ("#22c55e", "#166534"),  # green bg, dark text
    "LP":  ("#facc15", "#713f12"),  # yellow bg, dark text
    "MP":  ("#fb923c", "#7c2d12"),  # orange bg, dark text
    "HP":  ("#ef4444", "#ffffff"),  # red bg, white text
    "DMG": ("#7f1d1d", "#ffffff"),  # dark red bg, white text
}

_GRADE_LABELS = {
    "NM":  "Near Mint",
    "LP":  "Lightly Played",
    "MP":  "Moderately Played",
    "HP":  "Heavily Played",
    "DMG": "Damaged",
}

_CORNER_GRADE_COLORS = {
    "Gem":      "#22c55e",
    "Mint":     "#86efac",
    "Light":    "#facc15",
    "Moderate": "#fb923c",
    "Heavy":    "#ef4444",
}


# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------

def _image_to_base64(
    image: Union[str, Path, np.ndarray, Image.Image],
    fmt: str = "JPEG",
    quality: int = 85,
) -> str:
    """Convert an image to a base64-encoded data URI string."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            return ""
        pil_img = Image.open(path).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            rgb = image
        pil_img = Image.fromarray(rgb)
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        return ""

    # Resize large images to keep HTML file size manageable
    max_dim = 800
    if max(pil_img.size) > max_dim:
        ratio = max_dim / max(pil_img.size)
        new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def _render_annotated_card(
    image_path: str,
    assessment: dict,
) -> str:
    """Render the card image with condition overlays and return as base64 data URI.

    Draws:
    - Red rectangles on edges with detected whitening
    - Corner grade labels at each corner
    - Centering guide lines (cross-hair at center offset)
    - Surface defect heatmap overlay (from anomaly_map if available)
    """
    img = cv2.imread(image_path)
    if img is None:
        return ""

    h, w = img.shape[:2]
    overlay = img.copy()
    sub_scores = assessment.get("sub_scores", {})

    # --- Surface defect heatmap overlay ---
    # We don't have the raw anomaly_map in the assessment dict (it's not
    # JSON-serializable), so we render a synthetic heatmap from the defect
    # score if surface data is available.
    surface = sub_scores.get("surface")
    if surface and surface.get("details"):
        defect_ratio = surface["details"].get("defect_ratio", 0)
        if defect_ratio > 0:
            # Create a subtle red tint overlay to indicate surface issues
            # Intensity proportional to defect score
            intensity = min(int(surface["score"] * 100), 80)
            red_overlay = np.zeros_like(overlay)
            red_overlay[:, :, 2] = intensity  # Red channel
            overlay = cv2.add(overlay, red_overlay)

    # --- Edge whitening highlights ---
    edges = sub_scores.get("edges")
    if edges and edges.get("details", {}).get("per_edge"):
        per_edge = edges["details"]["per_edge"]
        strip_w = max(10, int(30 * min(w, h) / 1008))

        for side, info in per_edge.items():
            ratio = info.get("whitening_ratio", 0)
            if ratio <= 0.0:
                continue

            # Red intensity proportional to whitening severity
            red_intensity = min(int(ratio * 5000), 200)
            color = (0, 0, red_intensity)
            thickness = max(2, strip_w // 3)

            if side == "top":
                cv2.rectangle(overlay, (0, 0), (w, strip_w), color, thickness)
            elif side == "bottom":
                cv2.rectangle(overlay, (0, h - strip_w), (w, h), color, thickness)
            elif side == "left":
                cv2.rectangle(overlay, (0, 0), (strip_w, h), color, thickness)
            elif side == "right":
                cv2.rectangle(overlay, (w - strip_w, 0), (w, h), color, thickness)

    # --- Corner grade annotations ---
    corners = sub_scores.get("corners")
    if corners and corners.get("details", {}).get("per_corner"):
        per_corner = corners["details"]["per_corner"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(h, w) / 1500.0)
        thickness = max(1, int(font_scale * 2))
        pad = int(min(w, h) * 0.02)

        corner_positions = {
            "top_left":     (pad, int(pad + font_scale * 30)),
            "top_right":    (w - int(font_scale * 100) - pad, int(pad + font_scale * 30)),
            "bottom_left":  (pad, h - pad),
            "bottom_right": (w - int(font_scale * 100) - pad, h - pad),
        }

        for corner_name, pos in corner_positions.items():
            if corner_name in per_corner:
                grade = per_corner[corner_name].get("grade", "?")
                color_hex = _CORNER_GRADE_COLORS.get(grade, "#ffffff")
                # Convert hex to BGR
                r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
                color = (b, g, r)

                # Draw background rectangle for readability
                text_size = cv2.getTextSize(grade, font, font_scale, thickness)[0]
                x, y = pos
                cv2.rectangle(
                    overlay,
                    (x - 2, y - text_size[1] - 4),
                    (x + text_size[0] + 4, y + 4),
                    (0, 0, 0), -1,
                )
                cv2.putText(overlay, grade, pos, font, font_scale, color, thickness, cv2.LINE_AA)

    # --- Centering cross-hair lines ---
    centering = sub_scores.get("centering")
    if centering and centering.get("details"):
        details = centering["details"]
        lr_str = details.get("front_lr", "50/50")
        tb_str = details.get("front_tb", "50/50")

        try:
            lr_parts = lr_str.split("/")
            tb_parts = tb_str.split("/")
            lr_left = float(lr_parts[0])
            tb_top = float(tb_parts[0])

            # Draw centering lines showing where the center actually is
            cx = int(w * lr_left / 100.0)
            cy = int(h * tb_top / 100.0)

            # Dashed effect via short line segments
            dash_len = max(10, min(w, h) // 30)
            # Vertical line
            for y_pos in range(0, h, dash_len * 2):
                cv2.line(overlay, (cx, y_pos), (cx, min(y_pos + dash_len, h)),
                         (0, 255, 255), 1, cv2.LINE_AA)
            # Horizontal line
            for x_pos in range(0, w, dash_len * 2):
                cv2.line(overlay, (x_pos, cy), (min(x_pos + dash_len, w), cy),
                         (0, 255, 255), 1, cv2.LINE_AA)

            # Perfect center reference (dim gray)
            mid_x, mid_y = w // 2, h // 2
            cv2.drawMarker(overlay, (mid_x, mid_y), (128, 128, 128),
                           cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)
        except (ValueError, IndexError):
            pass

    return _image_to_base64(overlay)


def _get_ref_image_path(card_id: str) -> Optional[str]:
    """Resolve card_id to reference image path."""
    if not card_id:
        return None

    parts = card_id.split("/", 1)
    base_id = parts[0]
    variant = parts[1] if len(parts) > 1 else "normal"

    dash_idx = base_id.rfind("-")
    set_id = base_id[:dash_idx] if dash_idx > 0 else base_id

    filename = f"{base_id}_{variant}.png"

    for images_dir in (Path("data/card_images_hires"), Path("data/card_images")):
        candidate = images_dir / set_id / filename
        if candidate.exists():
            return str(candidate)

    if variant != "normal":
        fallback = f"{base_id}_normal.png"
        for images_dir in (Path("data/card_images_hires"), Path("data/card_images")):
            candidate = images_dir / set_id / fallback
            if candidate.exists():
                return str(candidate)

    return None


# ---------------------------------------------------------------------------
# HTML template pieces
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a; color: #e2e8f0; padding: 24px;
    max-width: 1100px; margin: 0 auto;
}
h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: 8px; color: #f8fafc; }
h2 { font-size: 1.1rem; font-weight: 600; margin: 20px 0 10px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
.header { display: flex; align-items: center; gap: 16px; margin-bottom: 24px; }
.header-text { flex: 1; }
.card-id { color: #64748b; font-size: 0.9rem; font-family: monospace; }

.grade-badge {
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 2.5rem; font-weight: 900; letter-spacing: 0.05em;
    width: 120px; height: 120px; border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.grade-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; text-align: center; }
.confidence { font-size: 0.75rem; color: #64748b; text-align: center; }

.images-row { display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap; }
.image-panel {
    flex: 1; min-width: 280px; background: #1e293b; border-radius: 12px;
    padding: 12px; text-align: center;
}
.image-panel img { max-width: 100%; border-radius: 8px; }
.image-panel .label { font-size: 0.8rem; color: #94a3b8; margin-bottom: 8px; }

table {
    width: 100%; border-collapse: collapse; margin: 8px 0;
    background: #1e293b; border-radius: 12px; overflow: hidden;
}
th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; }
th { background: #0f172a; color: #94a3b8; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; }
td { font-size: 0.9rem; }
tr:last-child td { border-bottom: none; }

.sub-grade {
    display: inline-block; padding: 2px 10px; border-radius: 6px;
    font-weight: 700; font-size: 0.8rem;
}

.meter-track {
    width: 100%; height: 8px; background: #334155; border-radius: 4px;
    overflow: hidden; margin-top: 4px;
}
.meter-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }

.price-box {
    background: #1e293b; border-radius: 12px; padding: 16px; margin: 16px 0;
    display: flex; gap: 24px; flex-wrap: wrap; align-items: center;
}
.price-item { text-align: center; flex: 1; min-width: 120px; }
.price-value { font-size: 1.4rem; font-weight: 700; color: #f8fafc; }
.price-label { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
.multiplier { font-size: 1.8rem; font-weight: 900; }

.corner-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
    max-width: 400px; margin: 8px 0;
}
.corner-cell {
    background: #1e293b; border-radius: 8px; padding: 10px;
    text-align: center; border: 2px solid #334155;
}
.corner-name { font-size: 0.7rem; color: #64748b; text-transform: uppercase; margin-bottom: 4px; }
.corner-grade { font-size: 1.1rem; font-weight: 700; }
.corner-conf { font-size: 0.7rem; color: #64748b; }

.centering-visual {
    background: #1e293b; border-radius: 12px; padding: 16px;
    display: flex; gap: 24px; align-items: center; flex-wrap: wrap;
}
.centering-box {
    width: 160px; height: 220px; border: 2px solid #475569;
    border-radius: 8px; position: relative; flex-shrink: 0;
}
.centering-inner {
    position: absolute; border: 2px dashed #22c55e; border-radius: 4px;
}
.centering-stats { flex: 1; min-width: 200px; }

.modules-bar {
    display: flex; gap: 8px; margin: 8px 0; flex-wrap: wrap;
}
.module-chip {
    padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
}
.module-run { background: #166534; color: #bbf7d0; }
.module-skip { background: #7f1d1d; color: #fecaca; }

.footer { margin-top: 24px; padding-top: 16px; border-top: 1px solid #334155; font-size: 0.7rem; color: #475569; }
"""


def _grade_badge_html(grade: str, confidence: float) -> str:
    """Render the large grade badge."""
    bg, fg = _GRADE_COLORS.get(grade, ("#64748b", "#ffffff"))
    label = _GRADE_LABELS.get(grade, grade)
    conf_pct = f"{confidence * 100:.0f}%" if confidence > 0 else "N/A"

    return f"""
    <div style="text-align: center;">
        <div class="grade-badge" style="background: {bg}; color: {fg};">{grade}</div>
        <div class="grade-label">{label}</div>
        <div class="confidence">Confidence: {conf_pct}</div>
    </div>
    """


def _meter_html(score: float, color: str = "#22c55e") -> str:
    """Render a horizontal meter bar."""
    pct = max(0, min(score * 100, 100))
    # Color transitions: green -> yellow -> orange -> red
    if pct < 20:
        bar_color = "#22c55e"
    elif pct < 40:
        bar_color = "#facc15"
    elif pct < 60:
        bar_color = "#fb923c"
    else:
        bar_color = "#ef4444"

    return f"""
    <div class="meter-track">
        <div class="meter-fill" style="width: {pct:.1f}%; background: {bar_color};"></div>
    </div>
    """


def _sub_grade_chip(grade: str) -> str:
    """Render a small grade chip."""
    bg, fg = _GRADE_COLORS.get(grade, ("#64748b", "#ffffff"))
    return f'<span class="sub-grade" style="background: {bg}; color: {fg};">{grade}</span>'


def _surface_section(surface: Optional[dict]) -> str:
    """Render the surface defects section."""
    if surface is None:
        return '<tr><td>Surface</td><td colspan="3"><em style="color:#64748b;">No reference image available</em></td></tr>'

    score = surface.get("score", 0)
    grade = surface.get("grade", "?")
    details = surface.get("details", {})

    return f"""
    <tr>
        <td><strong>Surface</strong></td>
        <td>{_sub_grade_chip(grade)}</td>
        <td>{score:.4f} {_meter_html(score)}</td>
        <td>
            Defect ratio: {details.get('defect_ratio', 0):.4f}<br>
            Flagged patches: {details.get('defect_count', 0)}/256<br>
            Mean similarity: {details.get('mean_similarity', 0):.4f}<br>
            Min similarity: {details.get('min_similarity', 0):.4f}
        </td>
    </tr>
    """


def _edges_section(edges: Optional[dict]) -> str:
    """Render the edge whitening section."""
    if edges is None:
        return '<tr><td>Edges</td><td colspan="3"><em style="color:#64748b;">Detection failed</em></td></tr>'

    score = edges.get("score", 0)
    grade = edges.get("grade", "?")
    details = edges.get("details", {})
    worst_edge = details.get("worst_edge", "?")
    worst_ratio = details.get("worst_ratio", 0)
    per_edge = details.get("per_edge", {})

    edge_detail_rows = ""
    for side in ("top", "right", "bottom", "left"):
        info = per_edge.get(side, {})
        r = info.get("whitening_ratio", 0)
        run = info.get("max_white_run", 0)
        bar_w = min(r * 10000, 100)  # scale for visibility
        is_worst = side == worst_edge
        marker = " (worst)" if is_worst else ""
        edge_detail_rows += f"""
            <span style="display:inline-block; width:65px; text-transform:capitalize;">{side}{marker}</span>
            <span style="font-family:monospace;">{r:.6f}</span>
            <span style="color:#64748b;"> run={run}</span><br>
        """

    return f"""
    <tr>
        <td><strong>Edges</strong></td>
        <td>{_sub_grade_chip(grade)}</td>
        <td>{score:.4f} {_meter_html(score)}</td>
        <td>
            Overall ratio: {details.get('overall_ratio', 0):.6f}<br>
            Clusters: {details.get('cluster_count', 0)}<br>
            Max run: {details.get('max_white_run', 0)}px<br>
            <details style="margin-top:4px;">
                <summary style="cursor:pointer; color:#94a3b8; font-size:0.8rem;">Per-edge details</summary>
                <div style="font-size:0.8rem; margin-top:4px; padding:6px; background:#0f172a; border-radius:6px;">
                    {edge_detail_rows}
                </div>
            </details>
        </td>
    </tr>
    """


def _centering_section(centering: Optional[dict]) -> str:
    """Render the centering section."""
    if centering is None:
        return '<tr><td>Centering</td><td colspan="3"><em style="color:#64748b;">Detection failed</em></td></tr>'

    grade = centering.get("grade", "?")
    psa_grade = centering.get("psa_grade", "?")
    details = centering.get("details", {})
    front_lr = details.get("front_lr", "50/50")
    front_tb = details.get("front_tb", "50/50")
    centering_score = centering.get("centering_score", 0)

    # Compute a severity score for the meter (centering_score is 1-10, 10=perfect)
    severity = max(0, 1.0 - centering_score / 10.0)

    return f"""
    <tr>
        <td><strong>Centering</strong></td>
        <td>{_sub_grade_chip(grade)}</td>
        <td>PSA {psa_grade} {_meter_html(severity)}</td>
        <td>
            L/R: <strong>{front_lr}</strong><br>
            T/B: <strong>{front_tb}</strong><br>
            Score: {centering_score:.1f}/10
        </td>
    </tr>
    """


def _centering_visual_html(centering: Optional[dict]) -> str:
    """Render a visual centering diagram."""
    if centering is None:
        return ""

    details = centering.get("details", {})
    front_lr = details.get("front_lr", "50/50")
    front_tb = details.get("front_tb", "50/50")

    try:
        lr_parts = front_lr.split("/")
        tb_parts = front_tb.split("/")
        left_pct = float(lr_parts[0])
        right_pct = float(lr_parts[1])
        top_pct = float(tb_parts[0])
        bottom_pct = float(tb_parts[1])
    except (ValueError, IndexError):
        left_pct = right_pct = top_pct = bottom_pct = 50.0

    # Normalize to the visual box dimensions (160x220)
    box_w, box_h = 160, 220
    # Inner rectangle position (proportional to border ratios)
    total_lr = left_pct + right_pct
    total_tb = top_pct + bottom_pct
    if total_lr == 0:
        total_lr = 100
    if total_tb == 0:
        total_tb = 100

    border_scale = 0.12  # borders are ~12% of card dimension
    inner_left = int(left_pct / total_lr * box_w * border_scale * 2)
    inner_top = int(top_pct / total_tb * box_h * border_scale * 2)
    inner_right = int(right_pct / total_lr * box_w * border_scale * 2)
    inner_bottom = int(bottom_pct / total_tb * box_h * border_scale * 2)

    inner_w = box_w - inner_left - inner_right
    inner_h = box_h - inner_top - inner_bottom

    psa_grade = centering.get("psa_grade", "?")
    centering_score = centering.get("centering_score", 0)

    return f"""
    <div class="centering-visual">
        <div class="centering-box">
            <div class="centering-inner" style="
                left: {inner_left}px; top: {inner_top}px;
                width: {inner_w}px; height: {inner_h}px;
            "></div>
            <div style="position:absolute; top:-18px; left:50%; transform:translateX(-50%); font-size:0.7rem; color:#94a3b8;">
                {top_pct:.0f}
            </div>
            <div style="position:absolute; bottom:-18px; left:50%; transform:translateX(-50%); font-size:0.7rem; color:#94a3b8;">
                {bottom_pct:.0f}
            </div>
            <div style="position:absolute; left:-22px; top:50%; transform:translateY(-50%); font-size:0.7rem; color:#94a3b8;">
                {left_pct:.0f}
            </div>
            <div style="position:absolute; right:-22px; top:50%; transform:translateY(-50%); font-size:0.7rem; color:#94a3b8;">
                {right_pct:.0f}
            </div>
        </div>
        <div class="centering-stats">
            <div style="font-size:0.9rem; margin-bottom:8px;">
                <strong>L/R:</strong> {front_lr} &nbsp;&nbsp;
                <strong>T/B:</strong> {front_tb}
            </div>
            <div style="font-size:0.9rem; margin-bottom:4px;">
                <strong>PSA Centering Grade:</strong> {psa_grade}
            </div>
            <div style="font-size:0.9rem;">
                <strong>Score:</strong> {centering_score:.1f}/10
            </div>
        </div>
    </div>
    """


def _corners_section(corners: Optional[dict]) -> str:
    """Render the corners row in the sub-grade table."""
    if corners is None:
        return '<tr><td>Corners</td><td colspan="3"><em style="color:#64748b;">No trained model available</em></td></tr>'

    score = corners.get("score", 0)
    grade = corners.get("grade", "?")
    corner_grade = corners.get("corner_grade", "?")

    return f"""
    <tr>
        <td><strong>Corners</strong></td>
        <td>{_sub_grade_chip(grade)}</td>
        <td>{score:.4f} {_meter_html(score)}</td>
        <td>Overall corner: <strong>{corner_grade}</strong></td>
    </tr>
    """


def _corners_grid_html(corners: Optional[dict]) -> str:
    """Render the per-corner visual grid."""
    if corners is None or not corners.get("details", {}).get("per_corner"):
        return ""

    per_corner = corners["details"]["per_corner"]

    cells = ""
    for name in ("top_left", "top_right", "bottom_left", "bottom_right"):
        info = per_corner.get(name, {})
        grade = info.get("grade", "?")
        conf = info.get("confidence", 0)
        color = _CORNER_GRADE_COLORS.get(grade, "#64748b")
        display_name = name.replace("_", " ").title()

        cells += f"""
        <div class="corner-cell" style="border-color: {color}40;">
            <div class="corner-name">{display_name}</div>
            <div class="corner-grade" style="color: {color};">{grade}</div>
            <div class="corner-conf">{conf:.0%}</div>
        </div>
        """

    return f'<div class="corner-grid">{cells}</div>'


def _price_section(assessment: dict, card_id: str) -> str:
    """Render the price impact section."""
    grade = assessment.get("overall_grade", "NM")
    multiplier = assessment.get("price_multiplier", 1.0)
    price_range = assessment.get("price_range", (0.9, 1.1))

    bg, fg = _GRADE_COLORS.get(grade, ("#64748b", "#ffffff"))

    # Try to look up the actual NM market price
    nm_price = None
    try:
        from cardprice.models import db
        # Attempt a quick price lookup if DB is available
        pass
    except Exception:
        pass

    if nm_price is not None:
        adjusted = nm_price * multiplier
        low = nm_price * price_range[0]
        high = nm_price * price_range[1]
        price_html = f"""
        <div class="price-item">
            <div class="price-value">${nm_price:.2f}</div>
            <div class="price-label">NM Market Price</div>
        </div>
        <div class="price-item">
            <div class="multiplier" style="color: {bg};">x{multiplier:.2f}</div>
            <div class="price-label">Condition Multiplier</div>
        </div>
        <div class="price-item">
            <div class="price-value" style="color: {bg};">${adjusted:.2f}</div>
            <div class="price-label">Adjusted Price ({grade})</div>
        </div>
        <div class="price-item">
            <div class="price-value" style="font-size:1rem;">${low:.2f} - ${high:.2f}</div>
            <div class="price-label">Confidence Interval</div>
        </div>
        """
    else:
        price_html = f"""
        <div class="price-item">
            <div class="multiplier" style="color: {bg};">x{multiplier:.2f}</div>
            <div class="price-label">Condition Multiplier ({grade})</div>
        </div>
        <div class="price-item">
            <div class="price-value" style="font-size:1rem;">x{price_range[0]:.2f} - x{price_range[1]:.2f}</div>
            <div class="price-label">Multiplier Range</div>
        </div>
        <div class="price-item" style="max-width:300px;">
            <div style="font-size:0.8rem; color:#94a3b8; text-align:left;">
                To calculate adjusted price:<br>
                <strong style="color:#e2e8f0;">NM Price x {multiplier:.2f} = {grade} Price</strong>
            </div>
        </div>
        """

    return f'<div class="price-box">{price_html}</div>'


def _modules_bar(assessment: dict) -> str:
    """Render the modules status bar."""
    run = assessment.get("modules_run", [])
    skipped = assessment.get("modules_skipped", [])

    chips = ""
    for m in run:
        chips += f'<span class="module-chip module-run">{m}</span>'
    for m in skipped:
        chips += f'<span class="module-chip module-skip">{m} (skipped)</span>'

    return f'<div class="modules-bar">{chips}</div>'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_condition_report(
    image_path: str,
    card_id: str,
    assessment: dict,
    output_path: str = None,
) -> str:
    """Generate an HTML condition report.

    Creates a self-contained HTML document with base64-encoded images,
    inline CSS, annotated card overlays, sub-grade breakdowns, centering
    visuals, corner grids, and price impact information.

    Parameters
    ----------
    image_path : str
        Path to the card photograph / scan being assessed.
    card_id : str
        Card identifier (e.g. ``"base1-4/holofoil"``).
    assessment : dict
        Output from ``assess_condition()``.  Expected keys:
        ``overall_grade``, ``overall_confidence``, ``sub_scores``,
        ``modules_run``, ``modules_skipped``, ``price_multiplier``,
        ``price_range``.
    output_path : str, optional
        If provided, saves the HTML to this file path.

    Returns
    -------
    str
        The complete HTML document as a string.
    """
    image_path = str(image_path)
    grade = assessment.get("overall_grade", "NM")
    confidence = assessment.get("overall_confidence", 0.0)
    sub_scores = assessment.get("sub_scores", {})

    # --- Encode images ---
    annotated_b64 = _render_annotated_card(image_path, assessment)

    ref_path = _get_ref_image_path(card_id)
    ref_b64 = _image_to_base64(ref_path) if ref_path else ""

    # --- Build HTML sections ---
    grade_badge = _grade_badge_html(grade, confidence)
    modules = _modules_bar(assessment)

    # Sub-grade table rows
    surface_row = _surface_section(sub_scores.get("surface"))
    edges_row = _edges_section(sub_scores.get("edges"))
    centering_row = _centering_section(sub_scores.get("centering"))
    corners_row = _corners_section(sub_scores.get("corners"))

    # Detail sections
    centering_viz = _centering_visual_html(sub_scores.get("centering"))
    corners_grid = _corners_grid_html(sub_scores.get("corners"))
    price_section = _price_section(assessment, card_id)

    # Images row
    images_html = '<div class="images-row">'
    if annotated_b64:
        images_html += f"""
        <div class="image-panel">
            <div class="label">Assessed Card (annotated)</div>
            <img src="{annotated_b64}" alt="Annotated card">
        </div>
        """
    if ref_b64:
        images_html += f"""
        <div class="image-panel">
            <div class="label">Reference Image</div>
            <img src="{ref_b64}" alt="Reference card">
        </div>
        """
    images_html += '</div>'

    # --- Assemble full HTML ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Condition Report - {card_id}</title>
<style>{_CSS}</style>
</head>
<body>

<div class="header">
    <div class="header-text">
        <h1>Card Condition Report</h1>
        <div class="card-id">{card_id}</div>
    </div>
    {grade_badge}
</div>

{modules}

{images_html}

<h2>Sub-Grade Breakdown</h2>
<table>
    <thead>
        <tr>
            <th style="width:110px;">Category</th>
            <th style="width:80px;">Grade</th>
            <th style="width:180px;">Score</th>
            <th>Details</th>
        </tr>
    </thead>
    <tbody>
        {surface_row}
        {edges_row}
        {centering_row}
        {corners_row}
    </tbody>
</table>

<h2>Centering Analysis</h2>
{centering_viz if centering_viz else '<div style="color:#64748b; padding:12px;">Centering data not available</div>'}

<h2>Corner Analysis</h2>
{corners_grid if corners_grid else '<div style="color:#64748b; padding:12px;">Corner classifier not available</div>'}

<h2>Price Impact</h2>
{price_section}

<div class="footer">
    Generated by cardprice condition assessment pipeline.
    Grade weights: Surface 35%, Edges 30%, Corners 20%, Centering 15%.
    Multipliers based on TCGPlayer market data analysis.
</div>

</body>
</html>"""

    # Save if output_path provided
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        logger.info("Condition report saved to %s", out)

    return html


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if len(sys.argv) < 3:
        print(
            "Usage: python -m cardprice.ml.condition_report <image> <card_id> "
            "[--assessment JSON_FILE] [--output report.html]"
        )
        print()
        print("If --assessment is not provided, runs assess_condition() first.")
        sys.exit(1)

    img_path = sys.argv[1]
    cid = sys.argv[2]
    assessment_file = None
    output = None

    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--assessment" and i + 1 < len(sys.argv):
            assessment_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    if assessment_file:
        with open(assessment_file) as f:
            assess = json.load(f)
    else:
        from cardprice.ml.condition_assessor import assess_condition
        assess = assess_condition(img_path, card_id=cid)

    if output is None:
        output = f"condition_report_{cid.replace('/', '_')}.html"

    html = generate_condition_report(img_path, cid, assess, output_path=output)
    print(f"Report generated: {output} ({len(html)} bytes)")
