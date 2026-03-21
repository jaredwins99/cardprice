"""Multi-card scanner UI for binder page and single card scanning.

Exports MULTI_CARD_HTML: a self-contained HTML page (no external deps)
that provides:
- Mode toggle: Single Card vs Binder Page
- Grid display of identified cards with segment thumbnails + reference images
- Re-scan button per card slot for corrections
- Total page value summary
- Grid position labels (Row N, Col N)
- Tap-to-detail modal for individual cards
- Pending/queued status with spinner animation
- Mobile-first dark theme matching the existing scanner style
"""

MULTI_CARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>Card Scanner</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg-primary: #1a1a2e;
    --bg-card: #16213e;
    --bg-card-hover: #1c2a4a;
    --bg-modal: #0f1629;
    --accent: #e94560;
    --accent-dark: #c23152;
    --green: #4ecca3;
    --green-dim: #3ba882;
    --text: #eee;
    --text-dim: #888;
    --text-faint: #555;
    --radius: 12px;
    --radius-sm: 8px;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-primary);
    color: var(--text);
    min-height: 100vh;
    -webkit-tap-highlight-color: transparent;
}

.container {
    max-width: 600px;
    margin: 0 auto;
    padding: 16px 12px 100px;
}

/* Header */
.header {
    text-align: center;
    padding: 8px 0 16px;
}
.header h1 {
    color: var(--accent);
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
}

/* QR Section */
.qr-section {
    text-align: center;
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 15px;
    margin: 0 0 16px;
}
.qr-section p { margin: 5px 0 10px; color: var(--text-dim); font-size: 14px; }
.qr-section .url { font-family: monospace; color: var(--green); font-size: 13px; word-break: break-all; }
#qrCanvas { image-rendering: pixelated; border-radius: 4px; }

/* Mode Toggle */
.mode-toggle {
    display: flex;
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 3px;
    margin-bottom: 12px;
    gap: 3px;
}
.mode-btn {
    flex: 1;
    padding: 10px 8px;
    border: none;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--text-dim);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}
.mode-btn.active {
    background: var(--accent);
    color: #fff;
}

/* Upload Buttons */
.btn-row {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
}
.btn-row .upload-btn {
    flex: 1;
    margin: 0;
}
.scan-row {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
    align-items: center;
}
.scan-row .upload-btn { flex: 1; }
.scan-row .toggle-switch { flex-shrink: 0; }
.upload-area { margin-bottom: 12px; }
.upload-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 14px 10px;
    font-size: 14px;
    font-weight: 600;
    border: none;
    border-radius: var(--radius);
    cursor: pointer;
    transition: background 0.15s;
}
.upload-btn.primary {
    background: var(--accent);
    color: #fff;
}
.upload-btn.primary:active { background: var(--accent-dark); }
.upload-btn.secondary {
    background: var(--bg-card);
    color: var(--text);
    border: 2px solid var(--accent);
}
.upload-btn.secondary:active { background: var(--bg-card-hover); }
input[type=file] { display: none; }

/* Continuous scan toggle */
.continuous-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
    cursor: pointer;
    user-select: none;
    -webkit-user-select: none;
}
.continuous-toggle .toggle-label {
    font-size: 11px;
    color: var(--text-dim);
    white-space: nowrap;
    line-height: 1.1;
}
.toggle-switch {
    position: relative;
    width: 38px;
    height: 22px;
    background: var(--bg-card);
    border-radius: 11px;
    border: 2px solid var(--text-faint);
    transition: all 0.2s;
    flex-shrink: 0;
}
.toggle-switch::after {
    content: '';
    position: absolute;
    top: 2px;
    left: 2px;
    width: 14px;
    height: 14px;
    background: var(--text-dim);
    border-radius: 50%;
    transition: all 0.2s;
}
.toggle-switch.active {
    background: var(--green);
    border-color: var(--green);
}
.toggle-switch.active::after {
    left: 18px;
    background: #fff;
}

/* Preview */
#preview {
    width: 100%;
    max-height: 300px;
    object-fit: contain;
    border-radius: var(--radius-sm);
    margin-bottom: 16px;
    display: none;
}

/* Summary Bar */
.summary-bar {
    display: none;
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 14px 16px;
    margin-bottom: 16px;
    justify-content: space-between;
    align-items: center;
}
.summary-bar.show { display: flex; }
.summary-label { color: var(--text-dim); font-size: 13px; }
.summary-value { font-size: 22px; font-weight: 700; color: var(--green); }
.summary-count { font-size: 14px; color: var(--text-dim); }

/* Session Action Buttons */
.session-actions {
    display: none;
    gap: 8px;
    margin-bottom: 16px;
}
.session-actions.show { display: flex; }
.session-btn {
    flex: 1;
    padding: 14px 12px;
    font-size: 15px;
    font-weight: 600;
    border: none;
    border-radius: var(--radius);
    cursor: pointer;
    transition: background 0.15s;
}
.session-btn.scan-next {
    background: var(--accent);
    color: #fff;
}
.session-btn.scan-next:active { background: var(--accent-dark); }
.session-btn.new-session {
    background: var(--bg-card);
    color: var(--text-dim);
    border: 2px solid var(--text-faint);
}
.session-btn.new-session:active { background: var(--bg-card-hover); }

/* Page Divider */
.page-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 0 6px;
    color: var(--text-dim);
    font-size: 13px;
    font-weight: 600;
}
.page-divider .divider-line {
    flex: 1;
    height: 1px;
    background: var(--text-faint);
}
.page-divider .page-subtotal {
    color: var(--green-dim);
    font-size: 12px;
    font-weight: 700;
}

/* Scanning Indicator */
.scanning-indicator {
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 32px 24px;
    color: var(--text-dim);
    font-size: 15px;
    gap: 12px;
}
.scanning-indicator.show { display: flex; }
.scanning-indicator .scan-spinner-wrap {
    position: relative;
    width: 96px; height: 96px;
}
.scanning-indicator .scan-spinner-ring {
    width: 96px; height: 96px;
    border: 4px solid var(--text-faint);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
.scanning-indicator .scan-timer {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 24px;
    font-weight: 700;
    color: #fff;
    font-variant-numeric: tabular-nums;
}
.scanning-indicator .scan-label .dots::after {
    content: '';
    animation: dots 1.5s steps(4, end) infinite;
}
@keyframes dots {
    0%   { content: ''; }
    25%  { content: '.'; }
    50%  { content: '..'; }
    75%  { content: '...'; }
    100% { content: ''; }
}

/* Spinner */
@keyframes spin { to { transform: rotate(360deg); } }
.spinner-ring {
    display: inline-block;
    width: 20px; height: 20px;
    border: 2.5px solid var(--text-faint);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    vertical-align: middle;
    margin-right: 8px;
}

/* Binder Card List (replaces grid for richer per-card display) */
.binder-results {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

/* Individual binder card row */
.binder-card-row {
    background: var(--bg-card);
    border-radius: var(--radius);
    overflow: hidden;
    transition: transform 0.15s;
}
.binder-card-row:active {
    transform: scale(0.99);
}

/* Grid position label */
.binder-pos-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px 0;
}
.binder-pos-label .pos-tag {
    font-size: 11px;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.binder-pos-label .rescan-btn {
    background: none;
    border: 1px solid var(--text-faint);
    border-radius: 6px;
    color: var(--text-dim);
    font-size: 11px;
    padding: 3px 10px;
    cursor: pointer;
    transition: all 0.15s;
}
.binder-pos-label .rescan-btn:hover,
.binder-pos-label .rescan-btn:active {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(233, 69, 96, 0.08);
}

/* Image comparison area */
.binder-images {
    display: flex;
    gap: 8px;
    padding: 8px 12px;
    align-items: stretch;
}

.binder-img-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 0;
}
.binder-img-col .img-label {
    font-size: 10px;
    color: var(--text-faint);
    text-transform: uppercase;
    letter-spacing: 0.3px;
    margin-bottom: 4px;
    font-weight: 600;
}
.binder-img-col .img-wrap {
    width: 100%;
    aspect-ratio: 3/4.2;
    background: #0d1321;
    border-radius: 6px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
}
.binder-img-col .img-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.binder-img-col .img-wrap .no-img {
    color: var(--text-faint);
    font-size: 11px;
    text-align: center;
    padding: 8px;
}

/* Card info section */
.binder-card-info {
    padding: 6px 12px 10px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}
.binder-card-info .info-left {
    flex: 1;
    min-width: 0;
}
.binder-card-info .info-name {
    font-size: 14px;
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 2px;
}
.binder-card-info .info-set {
    font-size: 12px;
    color: var(--text-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.binder-card-info .info-meta {
    font-size: 10px;
    color: var(--text-faint);
    margin-top: 2px;
}
.binder-card-info .info-price {
    font-size: 18px;
    font-weight: 700;
    color: var(--green);
    white-space: nowrap;
    margin-left: 8px;
}
.binder-card-info .info-price.no-price {
    color: var(--text-dim);
    font-size: 14px;
}

/* Condition prices row */
.cond-prices {
    display: flex;
    gap: 2px;
    padding: 0 12px 8px;
    font-size: 10px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}
.cond-prices .cp {
    flex: 1;
    text-align: center;
    padding: 3px 0;
    border-radius: 3px;
    background: rgba(255,255,255,0.04);
}
.cond-prices .cp .cl { opacity: 0.5; font-size: 9px; display: block; }
.cond-prices .cp.nm { color: #4ecca3; }
.cond-prices .cp.lp { color: #a8d8a8; }
.cond-prices .cp.mp { color: #f1c40f; }
.cond-prices .cp.hp { color: #e67e22; }
.cond-prices .cp.dmg { color: #e74c3c; }
.cond-prices .cp.blank { color: var(--text-faint); }

/* Action Buttons (Add to Inventory / Cart) */
.action-btns {
    display: flex;
    gap: 4px;
    padding: 4px 12px 8px;
}
.action-btn {
    flex: 1;
    padding: 6px;
    border: none;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s;
}
.action-btn:active { opacity: 0.7; }
.action-btn.inv { background: var(--green); color: #1a1a2e; }
.action-btn.cart { background: #3498db; color: #fff; }
.action-btn.done { opacity: 0.5; pointer-events: none; }
.add-all-btn {
    padding: 8px 14px;
    border: none;
    border-radius: var(--radius-sm);
    background: var(--green);
    color: #1a1a2e;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    white-space: nowrap;
    transition: opacity 0.15s;
}
.add-all-btn:active { opacity: 0.7; }

/* Toast notification */
.toast {
    position: fixed;
    bottom: 80px;
    left: 50%;
    transform: translateX(-50%) translateY(20px);
    background: #333;
    color: #fff;
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.25s, transform 0.25s;
    z-index: 2000;
    max-width: 90vw;
    text-align: center;
}
.toast.show {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
}

/* Card name as TCGPlayer link */
.binder-card-info .info-name a {
    color: inherit;
    text-decoration: none;
}
.binder-card-info .info-name a:active {
    color: var(--accent);
}

/* Queued state for binder rows */
.binder-card-row.queued .info-name { color: var(--text-dim); }
.binder-card-row.queued .img-wrap {
    position: relative;
}
.binder-card-row.queued .img-wrap::after {
    content: '';
    position: absolute;
    width: 24px; height: 24px;
    border: 3px solid var(--text-faint);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

/* Hidden rescan file input */
.rescan-input { display: none; }

/* Single Card Result */
.single-result {
    display: none;
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 20px 16px;
    text-align: center;
}
.single-result.show { display: block; }
.single-result .sr-image {
    max-width: 200px;
    border-radius: var(--radius-sm);
    margin: 0 auto 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.single-result .sr-name {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 4px;
}
.single-result .sr-set {
    font-size: 14px;
    color: var(--text-dim);
    margin-bottom: 8px;
}
.single-result .sr-price {
    font-size: 28px;
    font-weight: 700;
    color: var(--green);
    margin-bottom: 6px;
}
.single-result .sr-meta {
    font-size: 12px;
    color: var(--text-dim);
}

/* Detail Modal */
.modal-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.75);
    z-index: 1000;
    align-items: flex-end;
    justify-content: center;
}
.modal-overlay.show {
    display: flex;
}
.modal-sheet {
    background: var(--bg-modal);
    width: 100%;
    max-width: 600px;
    max-height: 85vh;
    border-radius: 20px 20px 0 0;
    padding: 8px 20px 32px;
    overflow-y: auto;
    animation: slideUp 0.25s ease-out;
}
@keyframes slideUp {
    from { transform: translateY(100%); }
    to { transform: translateY(0); }
}
.modal-handle {
    width: 40px;
    height: 4px;
    background: var(--text-faint);
    border-radius: 2px;
    margin: 8px auto 20px;
}

/* Modal side-by-side images */
.modal-images {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
    justify-content: center;
}
.modal-img-col {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.modal-img-col .img-label {
    font-size: 10px;
    color: var(--text-faint);
    text-transform: uppercase;
    letter-spacing: 0.3px;
    margin-bottom: 4px;
    font-weight: 600;
}
.modal-img-col img {
    max-width: 150px;
    max-height: 210px;
    border-radius: var(--radius-sm);
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    object-fit: contain;
}

.modal-name {
    font-size: 20px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 4px;
}
.modal-set {
    font-size: 14px;
    color: var(--text-dim);
    text-align: center;
    margin-bottom: 12px;
}
.modal-price {
    font-size: 32px;
    font-weight: 700;
    color: var(--green);
    text-align: center;
    margin-bottom: 6px;
}
.modal-meta-row {
    display: flex;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 14px;
}
.modal-meta-row .label { color: var(--text-dim); }
.modal-meta-row .value { font-weight: 600; }

.variant-badge {
    display: inline-block;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    padding: 2px 6px;
    border-radius: 3px;
    margin-left: 6px;
    vertical-align: middle;
}
.variant-badge.stamped { background: #9b59b6; color: #fff; }
.variant-badge.first-edition { background: #f1c40f; color: #333; }
.variant-badge.holo { background: linear-gradient(135deg, #e74c3c, #f1c40f, #2ecc71, #3498db); color: #fff; }
.variant-badge.reverse-holo { background: #95a5a6; color: #fff; }
.modal-close {
    display: block;
    width: 100%;
    margin-top: 20px;
    padding: 14px;
    font-size: 16px;
    font-weight: 600;
    border: none;
    border-radius: var(--radius);
    background: var(--accent);
    color: #fff;
    cursor: pointer;
}
.modal-close:active { background: var(--accent-dark); }

/* Modal action buttons */
.modal-actions {
    display: flex;
    gap: 8px;
    margin-top: 16px;
    padding: 0;
}
.modal-actions .action-btn {
    flex: 1;
    padding: 12px 8px;
    font-size: 13px;
    font-weight: 600;
    border: none;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: opacity 0.15s;
}
.modal-actions .action-btn:active { opacity: 0.7; }
.modal-actions .btn-inventory {
    background: var(--green);
    color: #1a1a2e;
}
.modal-actions .btn-cart {
    background: #3498db;
    color: #fff;
}
.modal-actions .action-msg {
    font-size: 11px;
    text-align: center;
    padding-top: 4px;
    min-height: 18px;
}

/* TCGPlayer external link */
.modal-tcg-link {
    display: block;
    text-align: center;
    margin: 10px 0 0;
    font-size: 13px;
    color: var(--text-dim);
    text-decoration: none;
}
.modal-tcg-link:hover { color: var(--text); }
.modal-tcg-link svg {
    width: 12px;
    height: 12px;
    vertical-align: -1px;
    margin-left: 4px;
    fill: currentColor;
}

/* Modal variant badge (reuse card list badge styles) */
.modal-variant-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 4px;
    margin-top: 6px;
}
.modal-variant-badge.stamped { background: #9b59b6; color: #fff; }
.modal-variant-badge.first-edition { background: #f1c40f; color: #333; }
.modal-variant-badge.holo { background: linear-gradient(135deg, #e74c3c, #f1c40f, #2ecc71, #3498db); color: #fff; }
.modal-variant-badge.reverse-holo { background: #95a5a6; color: #fff; }
</style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Pokemon Card Scanner</h1>
    </div>

    <!-- QR Code (desktop only) -->
    <div class="qr-section" id="qrSection">
        <p>Scan QR code to open on your phone</p>
        <canvas id="qrCanvas"></canvas>
        <br>
        <span class="url" id="serverUrl"></span>
    </div>

    <!-- Mode Toggle -->
    <div class="mode-toggle">
        <button class="mode-btn active" data-mode="single" onclick="setMode('single')">Single Card</button>
        <button class="mode-btn" data-mode="binder" onclick="setMode('binder')">Binder Page</button>
    </div>

    <!-- Single Card Buttons -->
    <div class="upload-area" id="singleButtons">
        <div class="btn-row">
            <label class="upload-btn primary" for="camera">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/><circle cx="12" cy="13" r="4"/></svg>
                Take Photo
            </label>
            <input type="file" id="camera" accept="image/*" capture="environment">
            <label class="upload-btn secondary" for="gallery">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                Gallery
            </label>
            <input type="file" id="gallery" accept="image/*">
        </div>
    </div>

    <!-- Binder Page Buttons -->
    <div class="upload-area" id="binderButtons" style="display:none">
        <div class="scan-row">
            <label class="upload-btn primary" for="binderCamera">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/><circle cx="12" cy="13" r="4"/></svg>
                Scan Binder Page
            </label>
            <input type="file" id="binderCamera" accept="image/*" capture="environment">
            <label class="upload-btn secondary" for="binderGallery">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                Page from Gallery
            </label>
            <input type="file" id="binderGallery" accept="image/*">
            <div class="continuous-toggle" onclick="toggleContinuous()" title="Auto-open camera after each scan">
                <div class="toggle-switch" id="continuousToggle"></div>
                <span class="toggle-label">Auto</span>
            </div>
        </div>
    </div>

    <!-- Image Preview -->
    <img id="preview">

    <!-- Scanning Indicator -->
    <div class="scanning-indicator" id="scanningIndicator">
        <div class="scan-spinner-wrap">
            <div class="scan-spinner-ring"></div>
            <div class="scan-timer" id="scanTimer">0s</div>
        </div>
        <span class="scan-label">Scanning<span class="dots"></span></span>
    </div>

    <!-- Single Card Result -->
    <div class="single-result" id="singleResult">
        <img class="sr-image" id="srImage">
        <div class="sr-name" id="srName"></div>
        <div class="sr-set" id="srSet"></div>
        <div class="sr-price" id="srPrice"></div>
        <div class="sr-meta" id="srMeta"></div>
    </div>

    <!-- Binder Page Summary -->
    <div class="summary-bar" id="summaryBar">
        <div>
            <div class="summary-label" id="summaryLabel">Page Total</div>
            <div class="summary-value" id="summaryTotal">$0.00</div>
        </div>
        <div style="text-align:right; display:flex; align-items:center; gap:10px;">
            <button class="add-all-btn" id="addAllInvBtn" onclick="addAllToInventory()" style="display:none">Add All to Inventory</button>
            <div>
                <div class="summary-count" id="summaryCount">0 cards</div>
                <div class="summary-label" id="summaryStatus"></div>
            </div>
        </div>
    </div>

    <!-- Toast notification -->
    <div class="toast" id="toast"></div>

    <!-- Session Action Buttons (shown after binder scan completes) -->
    <div class="session-actions" id="sessionActions">
        <button class="session-btn scan-next" id="scanNextBtn" onclick="scanNextPage()">Scan Next Page</button>
        <button class="session-btn new-session" onclick="newSession()">New Session</button>
    </div>

    <!-- Binder Results (list of card rows with thumbnails) -->
    <div class="binder-results" id="binderResults"></div>
</div>

<!-- Detail Modal -->
<div class="modal-overlay" id="detailModal">
    <div class="modal-sheet" id="modalSheet">
        <div class="modal-handle"></div>
        <div class="modal-images" id="modalImages"></div>
        <div class="modal-name" id="modalName"></div>
        <div class="modal-set" id="modalSet"></div>
        <div id="modalVariantBadge" style="text-align:center"></div>
        <div class="modal-price" id="modalPrice"></div>
        <div class="cond-prices" id="modalCondPrices" style="justify-content:center;padding:0 20px 12px;"></div>
        <a id="modalTcgLink" class="modal-tcg-link" href="#" target="_blank" rel="noopener" style="display:none">View on TCGPlayer <svg viewBox="0 0 24 24"><path d="M14 3h7v7h-2V6.41l-9.29 9.3-1.42-1.42L17.59 5H14V3zM5 5h5v2H7v10h10v-3h2v5H5V5z"/></svg></a>
        <div class="modal-actions">
            <button class="action-btn btn-inventory" id="modalAddInventory">Add to Inventory</button>
            <button class="action-btn btn-cart" id="modalAddCart">Add to Cart</button>
        </div>
        <div class="action-msg" id="modalActionMsg"></div>
        <div id="modalMeta">
            <div class="modal-meta-row">
                <span class="label">Position</span>
                <span class="value" id="modalPosition">--</span>
            </div>
            <div class="modal-meta-row">
                <span class="label">Card ID</span>
                <span class="value" id="modalCardId">--</span>
            </div>
            <div class="modal-meta-row">
                <span class="label">Method</span>
                <span class="value" id="modalMethod">--</span>
            </div>
            <div class="modal-meta-row">
                <span class="label">Confidence</span>
                <span class="value" id="modalConfidence">--</span>
            </div>
            <div class="modal-meta-row" id="modalVariantRow" style="display:none">
                <span class="label">Variant</span>
                <span class="value" id="modalVariant">--</span>
            </div>
        </div>
        <button class="modal-close" onclick="closeModal()">Close</button>
    </div>
</div>

<!-- QR Code Generator (same as existing, self-contained) -->
<script>
var QRGen=(function(){
"use strict";
var EXP=new Array(256),LOG=new Array(256);
(function(){var v=1;for(var i=0;i<255;i++){EXP[i]=v;LOG[v]=i;v<<=1;if(v>=256)v^=0x11d;}EXP[255]=EXP[0];})();
function gfMul(a,b){return a===0||b===0?0:EXP[(LOG[a]+LOG[b])%255];}
function polyMul(a,b){var r=new Array(a.length+b.length-1).fill(0);for(var i=0;i<a.length;i++)for(var j=0;j<b.length;j++)r[i+j]^=gfMul(a[i],b[j]);return r;}
function ecBytes(data,ecLen){
    var gen=[1];for(var i=0;i<ecLen;i++)gen=polyMul(gen,[1,EXP[i]]);
    var msg=new Array(data.length+ecLen).fill(0);for(var i=0;i<data.length;i++)msg[i]=data[i];
    for(var i=0;i<data.length;i++){var c=msg[i];if(c!==0)for(var j=0;j<gen.length;j++)msg[i+j]^=gfMul(gen[j],c);}
    return msg.slice(data.length);
}
var VERSIONS=[
    null,
    {total:26,ec:10,cap:16},{total:44,ec:16,cap:28},{total:70,ec:26,cap:44},
    {total:100,ec:18,cap:82},{total:134,ec:26,cap:108},{total:172,ec:18,cap:154}
];
var ALIGN=[null,null,[6,18],[6,22],[6,26],[6,30],[6,34]];
function chooseVersion(len){for(var v=1;v<=6;v++){if(len<=VERSIONS[v].cap)return v;}return 6;}
function makeMatrix(sz){var m=[];for(var i=0;i<sz;i++){var r=[];for(var j=0;j<sz;j++)r.push(null);m.push(r);}return m;}
function addFinder(m,row,col){
    for(var r=-1;r<=7;r++)for(var c=-1;c<=7;c++){
        var rr=row+r,cc=col+c;if(rr<0||rr>=m.length||cc<0||cc>=m.length)continue;
        m[rr][cc]=((r>=0&&r<=6&&(c===0||c===6))||(c>=0&&c<=6&&(r===0||r===6))||(r>=2&&r<=4&&c>=2&&c<=4))?1:0;
    }
}
function addAlignment(m,row,col){
    for(var r=-2;r<=2;r++)for(var c=-2;c<=2;c++)
        m[row+r][col+c]=(Math.abs(r)===2||Math.abs(c)===2||(r===0&&c===0))?1:0;
}
function addTimingPatterns(m){var sz=m.length;for(var i=8;i<sz-8;i++){if(m[6][i]===null)m[6][i]=(i%2===0)?1:0;if(m[i][6]===null)m[i][6]=(i%2===0)?1:0;}}
function reserveFormatInfo(m){
    var sz=m.length;
    for(var i=0;i<8;i++){if(m[8][i]===null)m[8][i]=0;if(m[i][8]===null)m[i][8]=0;if(m[8][sz-1-i]===null)m[8][sz-1-i]=0;if(m[sz-1-i][8]===null)m[sz-1-i][8]=0;}
    if(m[8][8]===null)m[8][8]=0;m[sz-8][8]=1;
}
function placeData(m,bits){
    var sz=m.length,idx=0;
    for(var col=sz-1;col>=1;col-=2){
        if(col===6)col=5;
        for(var row=0;row<sz;row++){for(var c=0;c<2;c++){
            var cc=col-c,goUp=((Math.floor((sz-1-col)/2))%2===0),rr=goUp?(sz-1-row):row;
            if(m[rr][cc]===null){m[rr][cc]=(idx<bits.length)?bits[idx]:0;idx++;}
        }}
    }
}
function isReserved(m,r,c,sz){
    if(r<9&&c<9)return true;if(r<9&&c>=sz-8)return true;if(r>=sz-8&&c<9)return true;
    if(r===6||c===6)return true;return false;
}
function applyMask0(m,sz){for(var r=0;r<sz;r++)for(var c=0;c<sz;c++){if(!isReserved(m,r,c,sz)&&(r+c)%2===0)m[r][c]^=1;}}
function writeFormatInfo(m){
    var sz=m.length;var bits=[1,0,1,0,1,0,0,0,0,0,1,0,0,1,0];
    var hpos=[0,1,2,3,4,5,7,8];
    for(var i=0;i<8;i++)m[8][hpos[i]]=bits[i];
    for(var i=0;i<7;i++)m[8][sz-7+i]=bits[8+i];
    for(var i=0;i<8;i++)m[hpos[7-i]][8]=bits[i];
    for(var i=0;i<7;i++)m[sz-1-i][8]=bits[8+i];
}
function pushBitsTo(arr,val,len){for(var i=len-1;i>=0;i--)arr.push((val>>i)&1);}
function encode(text){
    var bytes=[];for(var i=0;i<text.length;i++){var cp=text.charCodeAt(i);if(cp<128)bytes.push(cp);else if(cp<0x800){bytes.push(0xc0|(cp>>6));bytes.push(0x80|(cp&0x3f));}else{bytes.push(0xe0|(cp>>12));bytes.push(0x80|((cp>>6)&0x3f));bytes.push(0x80|(cp&0x3f));}}
    var version=chooseVersion(bytes.length);var vi=VERSIONS[version];var sz=17+version*4;
    var dataBits=[];
    pushBitsTo(dataBits,4,4);pushBitsTo(dataBits,bytes.length,version<=9?8:16);
    for(var i=0;i<bytes.length;i++)pushBitsTo(dataBits,bytes[i],8);
    var totalBits=vi.cap*8;var tl=Math.min(4,totalBits-dataBits.length);pushBitsTo(dataBits,0,tl);
    while(dataBits.length%8!==0)dataBits.push(0);
    var pad=[0xEC,0x11],pi=0;while(dataBits.length<totalBits){pushBitsTo(dataBits,pad[pi],8);pi^=1;}
    var dataBytes=[];for(var i=0;i<dataBits.length;i+=8){var b=0;for(var j=0;j<8;j++)b=(b<<1)|dataBits[i+j];dataBytes.push(b);}
    var ecCW=ecBytes(dataBytes,vi.ec);
    var allBits=[];for(var i=0;i<dataBytes.length;i++)pushBitsTo(allBits,dataBytes[i],8);
    for(var i=0;i<ecCW.length;i++)pushBitsTo(allBits,ecCW[i],8);
    var m=makeMatrix(sz);addFinder(m,0,0);addFinder(m,0,sz-7);addFinder(m,sz-7,0);
    if(ALIGN[version]){var ap=ALIGN[version];for(var i=0;i<ap.length;i++)for(var j=0;j<ap.length;j++){if(i===0&&j===0)continue;if(i===0&&j===ap.length-1)continue;if(i===ap.length-1&&j===0)continue;addAlignment(m,ap[i],ap[j]);}}
    addTimingPatterns(m);reserveFormatInfo(m);placeData(m,allBits);applyMask0(m,sz);writeFormatInfo(m);
    return m;
}
return{encode:encode};
})();

function drawQR(canvasId,text,cellSize){
    cellSize=cellSize||6;var matrix=QRGen.encode(text);var sz=matrix.length;
    var canvas=document.getElementById(canvasId);canvas.width=sz*cellSize;canvas.height=sz*cellSize;
    var ctx=canvas.getContext('2d');ctx.fillStyle='#1a1a2e';ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='#ffffff';
    for(var r=0;r<sz;r++)for(var c=0;c<sz;c++)if(matrix[r][c])ctx.fillRect(c*cellSize,r*cellSize,cellSize,cellSize);
}
(function(){
    var url='http://'+location.host;
    document.getElementById('serverUrl').textContent=url;
    drawQR('qrCanvas',url,6);
    if(/Mobi|Android|iPhone|iPad/i.test(navigator.userAgent))
        document.getElementById('qrSection').style.display='none';
})();
</script>

<script>
(function() {
    "use strict";

    // ---- State ----
    var currentMode = 'single';  // 'single' or 'binder'
    var cards = [];               // Array of card result objects for binder mode
    var activePolls = {};         // scan_id -> intervalId
    var pageCount = 0;            // Number of binder pages scanned in this session
    var pageBoundaries = [];      // Array of {startIdx, endIdx, pageNum} for page dividers
    var continuousScan = false;   // Auto-reopen camera after binder scan

    // ---- Mode Toggle ----
    window.setMode = function(mode) {
        currentMode = mode;
        var btns = document.querySelectorAll('.mode-btn');
        for (var i = 0; i < btns.length; i++) {
            btns[i].classList.toggle('active', btns[i].getAttribute('data-mode') === mode);
        }
        // Show/hide correct button area
        document.getElementById('singleButtons').style.display = (mode === 'single') ? '' : 'none';
        document.getElementById('binderButtons').style.display = (mode === 'binder') ? '' : 'none';
        // Reset display
        document.getElementById('singleResult').classList.remove('show');
        document.getElementById('summaryBar').classList.remove('show');
        document.getElementById('sessionActions').classList.remove('show');
        document.getElementById('binderResults').innerHTML = '';
        document.getElementById('preview').style.display = 'none';
        document.getElementById('scanningIndicator').classList.remove('show');
        cards = [];
        pageCount = 0;
        pageBoundaries = [];
        clearAllPolls();
    };

    window.toggleContinuous = function() {
        continuousScan = !continuousScan;
        document.getElementById('continuousToggle').classList.toggle('active', continuousScan);
    };

    function clearAllPolls() {
        for (var id in activePolls) {
            clearInterval(activePolls[id]);
        }
        activePolls = {};
    }

    // ---- File Handling ----
    function handleFile(file) {
        if (!file) return;
        var preview = document.getElementById('preview');
        preview.src = URL.createObjectURL(file);
        preview.style.display = 'block';

        if (currentMode === 'single') {
            handleSingleScan(file);
        } else {
            handleBinderScan(file);
        }
    }

    // ---- Single Card Mode ----
    function handleSingleScan(file) {
        var indicator = document.getElementById('scanningIndicator');
        var result = document.getElementById('singleResult');
        var summary = document.getElementById('summaryBar');
        var binder = document.getElementById('binderResults');

        indicator.classList.add('show');
        startScanTimer();
        result.classList.remove('show');
        summary.classList.remove('show');
        binder.innerHTML = '';

        var fd = new FormData();
        fd.append('image', file);

        fetch('/scan', { method: 'POST', body: fd })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                stopScanTimer();
                indicator.classList.remove('show');
                if (data.status === 'pending') {
                    showSinglePending(data.scan_id);
                } else if (data.error) {
                    showSingleError(data.error);
                } else {
                    showSingleResult(data);
                }
            })
            .catch(function(e) {
                stopScanTimer();
                indicator.classList.remove('show');
                showSingleError(String(e));
            });
    }

    function showSingleResult(data) {
        var el = document.getElementById('singleResult');
        el.classList.add('show');

        var img = document.getElementById('srImage');
        if (data.image_url) {
            img.src = data.image_url;
            img.style.display = 'block';
        } else {
            img.style.display = 'none';
        }

        document.getElementById('srName').textContent = data.card_name || 'Unknown Card';
        document.getElementById('srSet').textContent = data.set_name || '';
        document.getElementById('srPrice').textContent = data.market_price ? '$' + Number(data.market_price).toFixed(2) : 'No price data';
        var metaParts = [];
        if (data.confidence) metaParts.push(Math.round(data.confidence * 100) + '% confidence');
        if (data.method) metaParts.push('via ' + data.method);
        if (data.card_id) metaParts.push(data.card_id);
        document.getElementById('srMeta').textContent = metaParts.join(' \u2022 ');
    }

    function showSinglePending(scanId) {
        var el = document.getElementById('singleResult');
        el.classList.add('show');
        document.getElementById('srImage').style.display = 'none';
        document.getElementById('srName').textContent = 'Queued for identification...';
        document.getElementById('srSet').textContent = '';
        document.getElementById('srPrice').innerHTML = '<span class="spinner-ring"></span> Checking';
        document.getElementById('srMeta').textContent = 'Polling every 3s';

        var poll = setInterval(function() {
            fetch('/result/' + scanId)
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.status === 'resolved') {
                        clearInterval(poll);
                        showSingleResult(data);
                    }
                });
        }, 3000);
        activePolls[scanId] = poll;
    }

    function showSingleError(msg) {
        var el = document.getElementById('singleResult');
        el.classList.add('show');
        document.getElementById('srImage').style.display = 'none';
        document.getElementById('srName').textContent = 'Error';
        document.getElementById('srSet').textContent = msg;
        document.getElementById('srPrice').textContent = '';
        document.getElementById('srMeta').textContent = '';
    }

    // ---- Binder Page Mode ----
    // Posts to /scan-page which segments the binder page and identifies each card.
    // Response: { status, cards: [{position, row, col, card_id, card_name, ...}], total_value }

    var scanTimerInterval = null;

    function startScanTimer() {
        var start = Date.now();
        var timerEl = document.getElementById('scanTimer');
        timerEl.textContent = '0s';
        scanTimerInterval = setInterval(function() {
            var elapsed = Math.round((Date.now() - start) / 1000);
            timerEl.textContent = elapsed + 's';
        }, 1000);
    }
    function stopScanTimer() {
        if (scanTimerInterval) {
            clearInterval(scanTimerInterval);
            scanTimerInterval = null;
        }
    }

    function handleBinderScan(file) {
        var indicator = document.getElementById('scanningIndicator');
        indicator.classList.add('show');
        startScanTimer();
        document.getElementById('singleResult').classList.remove('show');
        document.getElementById('sessionActions').classList.remove('show');

        // Track page boundary: new page starts at current end of cards array
        pageCount++;
        var pageStartIdx = cards.length;

        var fd = new FormData();
        fd.append('image', file);

        fetch('/scan-page', { method: 'POST', body: fd })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                stopScanTimer();
                indicator.classList.remove('show');

                if (data.cards && data.cards.length > 0) {
                    for (var i = 0; i < data.cards.length; i++) {
                        cards.push(normalizeCard(data.cards[i]));
                    }
                } else if (data.status === 'pending') {
                    cards.push({
                        status: 'pending',
                        scan_id: data.scan_id,
                        card_name: 'Page queued...',
                        row: 0,
                        col: 0,
                    });
                    startPollForTile(cards.length - 1, data.scan_id);
                } else if (data.error) {
                    cards.push({
                        status: 'error',
                        card_name: 'Error',
                        set_name: data.error,
                        row: 0,
                        col: 0,
                    });
                }

                // Record this page's boundary
                pageBoundaries.push({
                    startIdx: pageStartIdx,
                    endIdx: cards.length,
                    pageNum: pageCount,
                });

                renderBinderResults();
                updateSummary();
                document.getElementById('sessionActions').classList.add('show');
                if (continuousScan) {
                    setTimeout(function() { document.getElementById('binderCamera').click(); }, 500);
                }
            })
            .catch(function(e) {
                stopScanTimer();
                indicator.classList.remove('show');
                cards.push({
                    status: 'error',
                    card_name: 'Error',
                    set_name: String(e),
                    row: 0,
                    col: 0,
                });

                pageBoundaries.push({
                    startIdx: pageStartIdx,
                    endIdx: cards.length,
                    pageNum: pageCount,
                });

                renderBinderResults();
                updateSummary();
                document.getElementById('sessionActions').classList.add('show');
            });
    }

    function normalizeCard(data) {
        return {
            status: 'resolved',
            position: data.position != null ? data.position : null,
            row: data.row != null ? data.row : null,
            col: data.col != null ? data.col : null,
            card_id: data.card_id || null,
            card_name: data.card_name || 'Unknown Card',
            set_name: data.set_name || '',
            market_price: data.market_price || null,
            image_url: data.image_url || null,
            local_image_url: data.local_image_url || null,
            segment_image_url: data.segment_image_url || null,
            confidence: data.confidence || null,
            method: data.method || null,
            condition_prices: data.condition_prices || null,
            tcgplayer_url: data.tcgplayer_url || null,
            variant: data.variant || null,
            variant_confidence: data.variant_confidence || null,
        };
    }

    function startPollForTile(idx, scanId) {
        var poll = setInterval(function() {
            fetch('/result/' + scanId)
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.status === 'resolved') {
                        clearInterval(poll);
                        delete activePolls[scanId];
                        cards[idx] = normalizeCard(data);
                        renderBinderResults();
                        updateSummary();
                    }
                });
        }, 3000);
        activePolls[scanId] = poll;
    }

    // ---- Re-scan individual card slot ----
    function rescanSlot(idx) {
        // Create a hidden file input, trigger it, and re-scan just that slot
        var input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.capture = 'environment';
        input.className = 'rescan-input';
        input.onchange = function() {
            var file = input.files[0];
            if (!file) return;

            // Mark slot as scanning
            cards[idx] = { status: 'scanning', card_name: 'Re-scanning...', row: cards[idx].row, col: cards[idx].col };
            renderBinderResults();
            updateSummary();

            var fd = new FormData();
            fd.append('image', file);

            fetch('/scan', { method: 'POST', body: fd })
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.status === 'pending') {
                        cards[idx] = {
                            status: 'pending',
                            scan_id: data.scan_id,
                            card_name: 'Queued...',
                            row: cards[idx].row,
                            col: cards[idx].col,
                        };
                        startPollForTile(idx, data.scan_id);
                    } else if (data.error) {
                        cards[idx] = {
                            status: 'error',
                            card_name: 'Error',
                            set_name: data.error,
                            row: cards[idx].row,
                            col: cards[idx].col,
                        };
                    } else {
                        var c = normalizeCard(data);
                        c.row = cards[idx].row;
                        c.col = cards[idx].col;
                        cards[idx] = c;
                    }
                    renderBinderResults();
                    updateSummary();
                })
                .catch(function(e) {
                    cards[idx] = {
                        status: 'error',
                        card_name: 'Error',
                        set_name: String(e),
                        row: cards[idx].row,
                        col: cards[idx].col,
                    };
                    renderBinderResults();
                    updateSummary();
                });

            input.remove();
        };
        document.body.appendChild(input);
        input.click();
    }

    // ---- Binder Results Rendering ----
    function getPageForIndex(idx) {
        for (var p = 0; p < pageBoundaries.length; p++) {
            if (idx >= pageBoundaries[p].startIdx && idx < pageBoundaries[p].endIdx) {
                return pageBoundaries[p];
            }
        }
        return null;
    }

    function calcPageSubtotal(page) {
        var total = 0;
        for (var i = page.startIdx; i < page.endIdx; i++) {
            if (cards[i] && cards[i].status === 'resolved' && cards[i].market_price) {
                total += Number(cards[i].market_price);
            }
        }
        return total;
    }

    function renderBinderResults() {
        var container = document.getElementById('binderResults');
        container.innerHTML = '';

        for (var i = 0; i < cards.length; i++) {
            // Insert page divider at start of each page
            var page = getPageForIndex(i);
            if (page && i === page.startIdx && pageCount > 1) {
                var divider = document.createElement('div');
                divider.className = 'page-divider';
                var line1 = document.createElement('span');
                line1.className = 'divider-line';
                var label = document.createElement('span');
                var st = calcPageSubtotal(page);
                label.textContent = 'Page ' + page.pageNum + ' \u00b7 $' + st.toFixed(2);
                var line2 = document.createElement('span');
                line2.className = 'divider-line';
                divider.appendChild(line1);
                divider.appendChild(label);
                divider.appendChild(line2);
                container.appendChild(divider);
            }

            var card = cards[i];
            var row = document.createElement('div');
            row.className = 'binder-card-row';
            if (card.status === 'scanning' || card.status === 'pending') {
                row.className += ' queued';
            }

            // Position label + Re-scan button
            var posLabel = document.createElement('div');
            posLabel.className = 'binder-pos-label';

            var posTag = document.createElement('span');
            posTag.className = 'pos-tag';
            if (card.row != null && card.col != null) {
                posTag.textContent = 'Row ' + (card.row + 1) + ', Col ' + (card.col + 1);
            } else if (card.position != null) {
                posTag.textContent = 'Slot ' + (card.position + 1);
            } else {
                posTag.textContent = 'Card ' + (i + 1);
            }
            posLabel.appendChild(posTag);

            var rescanBtn = document.createElement('button');
            rescanBtn.className = 'rescan-btn';
            rescanBtn.textContent = 'Re-scan';
            (function(idx) {
                rescanBtn.addEventListener('click', function(e) {
                    e.stopPropagation();
                    rescanSlot(idx);
                });
            })(i);
            posLabel.appendChild(rescanBtn);
            row.appendChild(posLabel);

            // Side-by-side images: scanned segment vs reference
            var images = document.createElement('div');
            images.className = 'binder-images';

            // Left: scanned segment thumbnail
            var segCol = document.createElement('div');
            segCol.className = 'binder-img-col';
            var segLabel = document.createElement('div');
            segLabel.className = 'img-label';
            segLabel.textContent = 'Scanned';
            segCol.appendChild(segLabel);
            var segWrap = document.createElement('div');
            segWrap.className = 'img-wrap';
            if (card.segment_image_url) {
                var segImg = document.createElement('img');
                segImg.src = card.segment_image_url;
                segImg.alt = 'Scanned card';
                segImg.loading = 'lazy';
                segWrap.appendChild(segImg);
            } else {
                var noSeg = document.createElement('div');
                noSeg.className = 'no-img';
                noSeg.textContent = card.status === 'scanning' || card.status === 'pending' ? '' : 'No segment';
                segWrap.appendChild(noSeg);
            }
            segCol.appendChild(segWrap);
            images.appendChild(segCol);

            // Right: reference image
            var refCol = document.createElement('div');
            refCol.className = 'binder-img-col';
            var refLabel = document.createElement('div');
            refLabel.className = 'img-label';
            refLabel.textContent = 'Reference';
            refCol.appendChild(refLabel);
            var refWrap = document.createElement('div');
            refWrap.className = 'img-wrap';
            // Prefer local_image_url, fall back to image_url (remote)
            var refSrc = card.local_image_url || card.image_url;
            if (refSrc) {
                var refImg = document.createElement('img');
                refImg.src = refSrc;
                refImg.alt = card.card_name || 'Reference';
                refImg.loading = 'lazy';
                refWrap.appendChild(refImg);
            } else {
                var noRef = document.createElement('div');
                noRef.className = 'no-img';
                noRef.textContent = card.status === 'scanning' || card.status === 'pending' ? '' : 'No match';
                refWrap.appendChild(noRef);
            }
            refCol.appendChild(refWrap);
            images.appendChild(refCol);

            row.appendChild(images);

            // Card info: name, set, meta, price
            var info = document.createElement('div');
            info.className = 'binder-card-info';

            var infoLeft = document.createElement('div');
            infoLeft.className = 'info-left';

            var nameDiv = document.createElement('div');
            nameDiv.className = 'info-name';
            if (card.status === 'scanning') {
                nameDiv.textContent = 'Scanning...';
            } else if (card.status === 'pending') {
                nameDiv.textContent = 'Queued...';
            } else if (card.tcgplayer_url) {
                var nameLink = document.createElement('a');
                nameLink.href = card.tcgplayer_url;
                nameLink.target = '_blank';
                nameLink.rel = 'noopener';
                nameLink.textContent = card.card_name || 'Unknown';
                nameDiv.appendChild(nameLink);
            } else {
                nameDiv.textContent = card.card_name || 'Unknown';
            }
            if (card.variant && card.variant !== 'normal') {
                var badge = document.createElement('span');
                badge.className = 'variant-badge ' + card.variant.toLowerCase().replace(/\s+/g, '-').replace(/1st-edition/, 'first-edition');
                badge.textContent = card.variant;
                nameDiv.appendChild(badge);
            }
            infoLeft.appendChild(nameDiv);

            var setDiv = document.createElement('div');
            setDiv.className = 'info-set';
            setDiv.textContent = card.set_name || '';
            infoLeft.appendChild(setDiv);

            if (card.confidence || card.method) {
                var metaDiv = document.createElement('div');
                metaDiv.className = 'info-meta';
                var parts = [];
                if (card.confidence) parts.push(Math.round(card.confidence * 100) + '%');
                if (card.method) parts.push(card.method);
                metaDiv.textContent = parts.join(' / ');
                infoLeft.appendChild(metaDiv);
            }

            info.appendChild(infoLeft);

            var priceDiv = document.createElement('div');
            priceDiv.className = 'info-price';
            if (card.status === 'scanning' || card.status === 'pending') {
                priceDiv.innerHTML = '<span class="spinner-ring" style="width:16px;height:16px;border-width:2px"></span>';
            } else if (card.market_price) {
                priceDiv.textContent = '$' + Number(card.market_price).toFixed(2);
            } else {
                priceDiv.textContent = '--';
                priceDiv.className += ' no-price';
            }
            info.appendChild(priceDiv);

            row.appendChild(info);

            // Condition prices row (5 conditions, inline)
            if (card.condition_prices || card.market_price) {
                var cpRow = document.createElement('div');
                cpRow.className = 'cond-prices';
                var conds = ['NM', 'LP', 'MP', 'HP', 'DMG'];
                var condClasses = ['nm', 'lp', 'mp', 'hp', 'dmg'];
                for (var ci = 0; ci < conds.length; ci++) {
                    var cpEl = document.createElement('div');
                    var cp = card.condition_prices && card.condition_prices[conds[ci]];
                    if (cp && cp.price != null) {
                        cpEl.className = 'cp ' + condClasses[ci];
                        cpEl.innerHTML = '<span class="cl">' + conds[ci] + '</span>$' + Number(cp.price).toFixed(cp.price >= 10 ? 0 : 2);
                    } else {
                        cpEl.className = 'cp blank';
                        cpEl.innerHTML = '<span class="cl">' + conds[ci] + '</span>—';
                    }
                    cpRow.appendChild(cpEl);
                }
                row.appendChild(cpRow);
            }

            // Action buttons (Add to Inventory / Add to Cart)
            if (card.status === 'resolved' && card.card_id) {
                var btnRow = document.createElement('div');
                btnRow.className = 'action-btns';

                var invBtn = document.createElement('button');
                invBtn.className = 'action-btn inv';
                invBtn.textContent = 'Add to Inventory';
                (function(c, btn) {
                    btn.addEventListener('click', function(e) {
                        e.stopPropagation();
                        addCardTo('/inventory/add', c, btn, 'Inventory');
                    });
                })(card, invBtn);
                btnRow.appendChild(invBtn);

                var cartBtn = document.createElement('button');
                cartBtn.className = 'action-btn cart';
                cartBtn.textContent = 'Add to Cart';
                (function(c, btn) {
                    btn.addEventListener('click', function(e) {
                        e.stopPropagation();
                        addCardTo('/cart/add', c, btn, 'Cart');
                    });
                })(card, cartBtn);
                btnRow.appendChild(cartBtn);

                row.appendChild(btnRow);
            }

            // Tap handler for detail modal (resolved/error cards)
            if (card.status === 'resolved' || card.status === 'error') {
                (function(c) {
                    row.addEventListener('click', function() { openModal(c); });
                })(card);
            }

            container.appendChild(row);
        }
    }

    function updateSummary() {
        var bar = document.getElementById('summaryBar');
        bar.classList.add('show');

        var total = 0;
        var resolved = 0;
        var pending = 0;

        for (var i = 0; i < cards.length; i++) {
            if (cards[i].status === 'resolved') {
                resolved++;
                if (cards[i].market_price) {
                    total += Number(cards[i].market_price);
                }
            } else {
                pending++;
            }
        }

        document.getElementById('summaryTotal').textContent = '$' + total.toFixed(2);

        // Multi-page summary: "X pages . Y cards"
        var countParts = [];
        if (pageCount > 1) {
            countParts.push(pageCount + ' page' + (pageCount !== 1 ? 's' : ''));
        }
        countParts.push(cards.length + ' card' + (cards.length !== 1 ? 's' : ''));
        document.getElementById('summaryCount').textContent = countParts.join(' \u00b7 ');

        // Update label text
        document.getElementById('summaryLabel').textContent = pageCount > 1 ? 'Session Total' : 'Page Total';

        var statusEl = document.getElementById('summaryStatus');
        if (pending > 0) {
            statusEl.innerHTML = '<span class="spinner-ring" style="width:12px;height:12px;border-width:2px"></span> ' + pending + ' pending';
        } else {
            statusEl.textContent = resolved + ' identified';
        }

        // Show "Add All to Inventory" when there are resolved cards
        var addAllBtn = document.getElementById('addAllInvBtn');
        if (addAllBtn) {
            addAllBtn.style.display = resolved > 0 ? 'inline-block' : 'none';
        }
    }

    // ---- Toast ----
    var toastTimer = null;
    function showToast(msg) {
        var el = document.getElementById('toast');
        el.textContent = msg;
        el.classList.add('show');
        if (toastTimer) clearTimeout(toastTimer);
        toastTimer = setTimeout(function() { el.classList.remove('show'); }, 2000);
    }

    // ---- Add Card to Inventory/Cart ----
    function addCardTo(endpoint, card, btn, label) {
        var payload = { card_id: card.card_id, quantity: 1 };
        btn.disabled = true;
        btn.textContent = '...';
        fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        }).then(function(r) { return r.json(); }).then(function(data) {
            if (data.error) {
                showToast('Error: ' + data.error);
                btn.disabled = false;
                btn.textContent = label === 'Inventory' ? 'Add to Inventory' : 'Add to Cart';
            } else {
                btn.textContent = 'Added!';
                btn.classList.add('done');
                var name = card.card_name || card.card_id;
                showToast(name + ' added to ' + label.toLowerCase());
                setTimeout(function() {
                    btn.disabled = false;
                    btn.classList.remove('done');
                    btn.textContent = label === 'Inventory' ? 'Add to Inventory' : 'Add to Cart';
                }, 1500);
            }
        }).catch(function(err) {
            showToast('Failed: ' + err.message);
            btn.disabled = false;
            btn.textContent = label === 'Inventory' ? 'Add to Inventory' : 'Add to Cart';
        });
    }

    // ---- Add All to Inventory ----
    window.addAllToInventory = function() {
        var resolved = [];
        for (var i = 0; i < cards.length; i++) {
            if (cards[i].status === 'resolved' && cards[i].card_id) {
                resolved.push(cards[i]);
            }
        }
        if (resolved.length === 0) return;

        var btn = document.getElementById('addAllInvBtn');
        btn.disabled = true;
        btn.textContent = 'Adding...';

        var done = 0;
        var errors = 0;
        for (var j = 0; j < resolved.length; j++) {
            (function(card) {
                fetch('/inventory/add', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ card_id: card.card_id, quantity: 1 })
                }).then(function(r) { return r.json(); }).then(function(data) {
                    if (data.error) errors++;
                    done++;
                    checkDone();
                }).catch(function() {
                    errors++;
                    done++;
                    checkDone();
                });
            })(resolved[j]);
        }

        function checkDone() {
            if (done < resolved.length) return;
            btn.disabled = false;
            if (errors === 0) {
                btn.textContent = 'Added All!';
                showToast(resolved.length + ' card' + (resolved.length !== 1 ? 's' : '') + ' added to inventory');
            } else {
                showToast((done - errors) + ' added, ' + errors + ' failed');
            }
            setTimeout(function() { btn.textContent = 'Add All to Inventory'; }, 2000);
        }
    };

    // ---- Session Controls ----
    window.scanNextPage = function() {
        // Hide session actions and show upload buttons for next photo
        document.getElementById('sessionActions').classList.remove('show');
        document.getElementById('preview').style.display = 'none';
        // Scroll to top so user sees upload buttons
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    window.newSession = function() {
        // Full reset: clear all cards, pages, and UI
        cards = [];
        pageCount = 0;
        pageBoundaries = [];
        clearAllPolls();
        document.getElementById('summaryBar').classList.remove('show');
        document.getElementById('sessionActions').classList.remove('show');
        document.getElementById('binderResults').innerHTML = '';
        document.getElementById('preview').style.display = 'none';
        document.getElementById('scanningIndicator').classList.remove('show');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    // ---- Detail Modal ----
    window.openModal = function(card) {
        var overlay = document.getElementById('detailModal');
        overlay.classList.add('show');

        // Build modal images (segment + reference side-by-side)
        var imagesDiv = document.getElementById('modalImages');
        imagesDiv.innerHTML = '';

        if (card.segment_image_url) {
            var segCol = document.createElement('div');
            segCol.className = 'modal-img-col';
            var segLabel = document.createElement('div');
            segLabel.className = 'img-label';
            segLabel.textContent = 'Scanned';
            segCol.appendChild(segLabel);
            var segImg = document.createElement('img');
            segImg.src = card.segment_image_url;
            segCol.appendChild(segImg);
            imagesDiv.appendChild(segCol);
        }

        var refSrc = card.local_image_url || card.image_url;
        if (refSrc) {
            var refCol = document.createElement('div');
            refCol.className = 'modal-img-col';
            var refLabel = document.createElement('div');
            refLabel.className = 'img-label';
            refLabel.textContent = 'Reference';
            refCol.appendChild(refLabel);
            var refImg = document.createElement('img');
            refImg.src = refSrc;
            refCol.appendChild(refImg);
            imagesDiv.appendChild(refCol);
        }

        var modalNameEl = document.getElementById('modalName');
        if (card.tcgplayer_url) {
            modalNameEl.innerHTML = '<a href="' + card.tcgplayer_url + '" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;text-decoration-color:var(--text-faint)">' + (card.card_name || 'Unknown Card') + '</a>';
        } else {
            modalNameEl.textContent = card.card_name || 'Unknown Card';
        }
        document.getElementById('modalSet').textContent = card.set_name || '';

        // Variant badge (visual, below set name)
        var badgeDiv = document.getElementById('modalVariantBadge');
        badgeDiv.innerHTML = '';
        if (card.variant && card.variant !== 'normal') {
            var badge = document.createElement('span');
            var vClass = card.variant.replace(/[\s_]/g, '-').toLowerCase();
            badge.className = 'modal-variant-badge ' + vClass;
            badge.textContent = card.variant;
            badgeDiv.appendChild(badge);
        }

        document.getElementById('modalPrice').textContent = card.market_price ? '$' + Number(card.market_price).toFixed(2) : 'No price data';

        // TCGPlayer external link
        var tcgLink = document.getElementById('modalTcgLink');
        if (card.tcgplayer_url) {
            tcgLink.href = card.tcgplayer_url;
            tcgLink.style.display = '';
        } else {
            tcgLink.style.display = 'none';
        }

        // Action buttons
        var actionMsg = document.getElementById('modalActionMsg');
        actionMsg.textContent = '';
        var invBtn = document.getElementById('modalAddInventory');
        var cartBtn = document.getElementById('modalAddCart');

        // Clone buttons to remove old listeners
        var newInvBtn = invBtn.cloneNode(true);
        invBtn.parentNode.replaceChild(newInvBtn, invBtn);
        var newCartBtn = cartBtn.cloneNode(true);
        cartBtn.parentNode.replaceChild(newCartBtn, cartBtn);

        if (card.card_id) {
            newInvBtn.style.display = '';
            newCartBtn.style.display = '';
            newInvBtn.onclick = function() {
                newInvBtn.disabled = true;
                newInvBtn.textContent = 'Adding...';
                fetch('/inventory/add', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({card_id: card.card_id, quantity: 1})
                }).then(function(r) { return r.json(); }).then(function(d) {
                    newInvBtn.disabled = false;
                    newInvBtn.textContent = 'Add to Inventory';
                    actionMsg.textContent = d.error ? d.error : 'Added to inventory';
                    actionMsg.style.color = d.error ? '#e74c3c' : 'var(--green)';
                }).catch(function() {
                    newInvBtn.disabled = false;
                    newInvBtn.textContent = 'Add to Inventory';
                    actionMsg.textContent = 'Network error';
                    actionMsg.style.color = '#e74c3c';
                });
            };
            newCartBtn.onclick = function() {
                newCartBtn.disabled = true;
                newCartBtn.textContent = 'Adding...';
                fetch('/cart/add', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        card_id: card.card_id,
                        card_name: card.card_name || '',
                        set_name: card.set_name || '',
                        market_price: card.market_price || 0,
                        image_url: card.image_url || '',
                        tcgplayer_url: card.tcgplayer_url || ''
                    })
                }).then(function(r) { return r.json(); }).then(function(d) {
                    newCartBtn.disabled = false;
                    newCartBtn.textContent = 'Add to Cart';
                    actionMsg.textContent = d.error ? d.error : 'Added to cart';
                    actionMsg.style.color = d.error ? '#e74c3c' : '#3498db';
                }).catch(function() {
                    newCartBtn.disabled = false;
                    newCartBtn.textContent = 'Add to Cart';
                    actionMsg.textContent = 'Network error';
                    actionMsg.style.color = '#e74c3c';
                });
            };
        } else {
            newInvBtn.style.display = 'none';
            newCartBtn.style.display = 'none';
        }

        // Position
        var posText = '--';
        if (card.row != null && card.col != null) {
            posText = 'Row ' + (card.row + 1) + ', Col ' + (card.col + 1);
        }
        document.getElementById('modalPosition').textContent = posText;
        document.getElementById('modalCardId').textContent = card.card_id || '--';
        document.getElementById('modalMethod').textContent = card.method || '--';
        document.getElementById('modalConfidence').textContent = card.confidence ? (Math.round(card.confidence * 100) + '%') : '--';

        // Condition prices in modal
        var mcpDiv = document.getElementById('modalCondPrices');
        mcpDiv.innerHTML = '';
        if (card.condition_prices || card.market_price) {
            var conds = ['NM', 'LP', 'MP', 'HP', 'DMG'];
            var condClasses = ['nm', 'lp', 'mp', 'hp', 'dmg'];
            for (var ci = 0; ci < conds.length; ci++) {
                var cpEl = document.createElement('div');
                var cp = card.condition_prices && card.condition_prices[conds[ci]];
                if (cp && cp.price != null) {
                    cpEl.className = 'cp ' + condClasses[ci];
                    cpEl.innerHTML = '<span class="cl">' + conds[ci] + '</span>$' + Number(cp.price).toFixed(cp.price >= 10 ? 0 : 2);
                } else {
                    cpEl.className = 'cp blank';
                    cpEl.innerHTML = '<span class="cl">' + conds[ci] + '</span>\u2014';
                }
                mcpDiv.appendChild(cpEl);
            }
        }

        // Variant
        var variantRow = document.getElementById('modalVariantRow');
        if (card.variant && card.variant !== 'normal') {
            variantRow.style.display = '';
            var varText = card.variant;
            if (card.variant_confidence) varText += ' (' + Math.round(card.variant_confidence * 100) + '%)';
            document.getElementById('modalVariant').textContent = varText;
        } else {
            variantRow.style.display = 'none';
        }
    };

    window.closeModal = function() {
        document.getElementById('detailModal').classList.remove('show');
    };

    // Close modal on overlay click
    document.getElementById('detailModal').addEventListener('click', function(e) {
        if (e.target === this) closeModal();
    });

    // Swipe down to close modal
    (function() {
        var sheet = document.getElementById('modalSheet');
        var startY = 0;
        sheet.addEventListener('touchstart', function(e) {
            startY = e.touches[0].clientY;
        }, { passive: true });
        sheet.addEventListener('touchend', function(e) {
            var dy = e.changedTouches[0].clientY - startY;
            if (dy > 80) closeModal();
        }, { passive: true });
    })();

    // ---- Wire up file inputs ----
    document.getElementById('camera').onchange = function() { handleFile(this.files[0]); this.value = ''; };
    document.getElementById('gallery').onchange = function() { handleFile(this.files[0]); this.value = ''; };
    document.getElementById('binderCamera').onchange = function() { handleFile(this.files[0]); this.value = ''; };
    document.getElementById('binderGallery').onchange = function() { handleFile(this.files[0]); this.value = ''; };

})();
</script>
</body>
</html>
"""
