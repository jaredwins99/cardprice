"""Record cardprice README demo GIFs.

Two modes:
  search -- home -> type "charizard" -> autocomplete -> tap result list ->
            tap a vintage Charizard -> slow scroll through prices
  scan   -- home -> upload a 100% WOTC binder page -> show 9-card grid

Usage:
  python3 scripts/demo/record_demo.py search
  python3 scripts/demo/record_demo.py scan

Output: data/demo/<mode>_raw.webm and (after ffmpeg) <mode>.gif.
The server must be running at http://127.0.0.1:8888.
"""
from pathlib import Path
import argparse
import time
from playwright.sync_api import sync_playwright

OUT_DIR = Path("/home/godli/cardprice/data/demo")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIEW_W, VIEW_H = 414, 896  # iPhone-ish portrait

# Demo inputs
SCAN_PATH = Path("/home/godli/cardprice/data/inbox/page_20260307_120653.jpg")
# Base Set Charizard — vintage card with rich price history (1st Edition,
# Shadowless, Unlimited variants + lots of sales).
DETAIL_CARD_ID = "base1-4/normal"


def _hide_qr(page):
    """Hide the cloudflared tunnel QR section before recording."""
    page.evaluate("""
        () => {
            const qr = document.getElementById('qrSection');
            if (qr) qr.style.display = 'none';
        }
    """)


def _smooth_scroll(page, total_px, duration_s=2.5):
    """Animate window.scrollBy continuously over duration_s. Renders as
    smooth motion in the captured video — stepped JS scrolls look choppy
    once compressed to 8-10fps GIF."""
    page.evaluate(f"""
        () => new Promise(resolve => {{
            const total = {total_px};
            const dur = {duration_s * 1000};
            const start = performance.now();
            const startY = window.scrollY;
            function tick(t) {{
                const k = Math.min(1, (t - start) / dur);
                // ease-in-out
                const e = k < 0.5 ? 2*k*k : 1 - Math.pow(-2*k+2, 2)/2;
                window.scrollTo(0, startY + total * e);
                if (k < 1) requestAnimationFrame(tick);
                else resolve();
            }}
            requestAnimationFrame(tick);
        }})
    """)
    time.sleep(duration_s + 0.3)


def record_search(page):
    """Type charizard -> autocomplete -> click result -> detail page -> scroll."""
    page.goto("http://127.0.0.1:8888/", wait_until="networkidle")
    _hide_qr(page)
    time.sleep(1.0)

    # Pre-warm the names index so autocomplete fires on first keystroke
    page.evaluate("fetch('/names/all?lang=en').then(r=>r.json())")
    time.sleep(0.8)

    # Type "charizard" at a brisk pace so the autocomplete narrative reads quick
    page.click('#searchName')
    page.keyboard.type("charizard", delay=85)
    page.wait_for_selector('#searchNameDropdown .ac-item', timeout=8000)
    time.sleep(1.6)  # let the autocomplete dropdown breathe

    # Tap the first autocomplete entry -> runs the full search
    page.locator('#searchNameDropdown .ac-item').first.click()
    page.wait_for_selector('#searchResults .sr-card', timeout=8000)
    time.sleep(1.5)  # let user see all 27 charizards

    # Smooth scroll through results
    _smooth_scroll(page, 250, duration_s=1.8)

    # Tap the first result -> /card/<id>
    page.locator('#searchResults .sr-card').first.click()
    page.wait_for_load_state("networkidle")
    time.sleep(1.5)  # detail page settle

    # Smooth scroll to prices/sales section
    _smooth_scroll(page, 700, duration_s=3.5)
    time.sleep(1.0)
    _smooth_scroll(page, 500, duration_s=2.5)
    time.sleep(1.5)


def record_scan(page):
    """Home -> upload WOTC page -> wait for scan -> show 9-card grid."""
    page.goto("http://127.0.0.1:8888/", wait_until="networkidle")
    _hide_qr(page)
    time.sleep(1.0)  # short home idle

    # Trigger upload
    assert SCAN_PATH.exists(), f"missing scan: {SCAN_PATH}"
    page.set_input_files("#invPageGallery", str(SCAN_PATH))

    # Wait for scan to complete
    try:
        page.wait_for_function(
            "() => document.querySelectorAll('[id*=\"invPageTile_\"]').length >= 9",
            timeout=45000,
        )
    except Exception:
        print("warn: tiles never reached 9; capturing whatever rendered")

    # Bring the result grid into view (the scan-result section is below the
    # initial fold; smooth-scroll there immediately after results render).
    page.evaluate("""
        () => {
            const tile = document.querySelector('[id*="invPageTile_"]');
            if (tile) tile.scrollIntoView({behavior: 'smooth', block: 'start'});
        }
    """)
    time.sleep(2.0)

    # Smooth scroll through the result grid so all 9 tiles + prices read
    _smooth_scroll(page, 350, duration_s=2.2)
    time.sleep(1.2)
    _smooth_scroll(page, 250, duration_s=1.8)
    # Hold on the final state long enough for viewers to read it
    time.sleep(3.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["search", "scan"])
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": VIEW_W, "height": VIEW_H},
            device_scale_factor=2,
            record_video_dir=str(OUT_DIR),
            record_video_size={"width": VIEW_W, "height": VIEW_H},
        )
        page = ctx.new_page()

        if args.mode == "search":
            record_search(page)
        else:
            record_scan(page)

        ctx.close()
        browser.close()

    vids = sorted(OUT_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime)
    if vids:
        latest = vids[-1]
        target = OUT_DIR / f"{args.mode}_raw.webm"
        latest.rename(target)
        print(f"recorded: {target}")
    else:
        print("ERROR: no video produced")


if __name__ == "__main__":
    main()
