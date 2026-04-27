"""Record a README demo GIF of the cardprice scanner.

Drives the live server at :8888 with Playwright, uploads a known good
binder page, and captures the flow as MP4 -> we convert to GIF separately.
"""
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

SCAN_PATH = Path("/home/godli/cardprice/data/inbox/page_20260422_170426.jpg")
OUT_DIR = Path("/home/godli/cardprice/data/demo")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIEW_W, VIEW_H = 414, 896  # iPhone-ish portrait

def main():
    assert SCAN_PATH.exists(), f"missing scan: {SCAN_PATH}"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": VIEW_W, "height": VIEW_H},
            device_scale_factor=2,
            record_video_dir=str(OUT_DIR),
            record_video_size={"width": VIEW_W, "height": VIEW_H},
        )
        page = ctx.new_page()

        # 1. Land on home
        page.goto("http://127.0.0.1:8888/", wait_until="networkidle")
        # Hide the QR code section (exposes a private cloudflared tunnel URL)
        # and tweak the title block for demo polish.
        page.evaluate("""
            () => {
                const qr = document.getElementById('qrSection');
                if (qr) qr.style.display = 'none';
            }
        """)
        time.sleep(1.2)  # show home page

        # 2. Show search + autocomplete
        search_input = page.locator('#searchName').first
        search_input.click()
        page.keyboard.type("charizard", delay=70)
        time.sleep(1.8)  # autocomplete renders
        page.keyboard.press("Escape")
        search_input.evaluate("el => el.value = ''")
        time.sleep(0.4)

        # 3. Upload binder page via the gallery file input
        page.set_input_files("#invPageGallery", str(SCAN_PATH))

        # 4. Wait for scan to complete
        try:
            page.wait_for_function(
                "() => document.querySelectorAll('[id*=\"invPageTile_\"]').length >= 9",
                timeout=40000,
            )
        except Exception:
            print("warn: tiles never reached 9; capturing whatever rendered")

        time.sleep(1.2)

        # 5. Smooth scroll to bring the scan results into view, JS-driven
        # so it works in headless mode.
        page.evaluate("""
            () => {
                const tile = document.querySelector('[id*="invPageTile_"]');
                if (tile) tile.scrollIntoView({behavior: 'smooth', block: 'start'});
            }
        """)
        time.sleep(2.5)

        # Scroll down through the result grid
        page.evaluate("() => window.scrollBy({top: 350, left: 0, behavior: 'smooth'})")
        time.sleep(2.0)
        page.evaluate("() => window.scrollBy({top: 350, left: 0, behavior: 'smooth'})")
        time.sleep(2.0)
        # Hold on results
        time.sleep(1.0)

        # 5. End
        ctx.close()
        browser.close()

    # Find the captured webm and rename it
    vids = sorted(OUT_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime)
    if vids:
        latest = vids[-1]
        target = OUT_DIR / "demo_raw.webm"
        latest.rename(target)
        print(f"recorded: {target}")
    else:
        print("ERROR: no video produced")

if __name__ == "__main__":
    main()
