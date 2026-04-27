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


def _slow_scroll(page, total_px, step_px=80, delay=0.45):
    """Smooth scroll using small increments. Slower than wheel events."""
    direction = 1 if total_px > 0 else -1
    remaining = abs(total_px)
    while remaining > 0:
        step = min(step_px, remaining)
        page.evaluate(f"window.scrollBy({{top: {step*direction}, behavior: 'instant'}})")
        time.sleep(delay)
        remaining -= step


def record_search(page):
    """Type charizard -> autocomplete -> click result -> detail page -> scroll."""
    page.goto("http://127.0.0.1:8888/", wait_until="networkidle")
    _hide_qr(page)
    time.sleep(1.0)

    # Pre-warm the names index so autocomplete fires on first keystroke
    page.evaluate("fetch('/names/all?lang=en').then(r=>r.json())")
    time.sleep(1.0)

    # Click search input + type slowly enough to see autocomplete populate
    page.click('#searchName')
    page.keyboard.type("char", delay=120)
    time.sleep(0.8)
    page.keyboard.type("izard", delay=120)
    # Wait for dropdown to actually render
    page.wait_for_selector('#searchNameDropdown .ac-item', timeout=8000)
    time.sleep(2.5)  # let user "see" the dropdown of charizards

    # Click the first autocomplete entry ("Charizard")
    first_item = page.locator('#searchNameDropdown .ac-item').first
    first_item.click()
    time.sleep(2.5)  # results render below; let user see list

    # Scroll a bit so the result tiles are visible
    _slow_scroll(page, 350, step_px=70, delay=0.4)
    time.sleep(1.5)

    # Click the first result tile (top of search results) -> /card/<id>
    page.wait_for_selector('#searchResults .sr-card', timeout=8000)
    page.locator('#searchResults .sr-card').first.click()

    # Detail page
    page.wait_for_load_state("networkidle")
    time.sleep(2.0)

    # Slow scroll down through the detail page (image, prices, sales)
    _slow_scroll(page, 800, step_px=60, delay=0.4)
    time.sleep(1.5)
    _slow_scroll(page, 600, step_px=60, delay=0.4)
    time.sleep(2.0)


def record_scan(page):
    """Home -> upload WOTC page -> wait for scan -> show 9-card grid."""
    page.goto("http://127.0.0.1:8888/", wait_until="networkidle")
    _hide_qr(page)
    time.sleep(1.5)

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

    time.sleep(1.5)

    # Slow scroll to bring results into view
    page.evaluate("""
        () => {
            const tile = document.querySelector('[id*="invPageTile_"]');
            if (tile) tile.scrollIntoView({behavior: 'smooth', block: 'start'});
        }
    """)
    time.sleep(2.5)

    # Slow scroll through the result grid
    _slow_scroll(page, 350, step_px=60, delay=0.5)
    time.sleep(2.0)
    _slow_scroll(page, 350, step_px=60, delay=0.5)
    time.sleep(2.0)


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
