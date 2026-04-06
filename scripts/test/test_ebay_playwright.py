#!/usr/bin/env python3
"""Test Playwright's ability to bypass eBay bot detection for sold listings.

Findings from testing (2026-03-07):

BOT DETECTION BYPASS:
  - Playwright + playwright-stealth + homepage warmup WORKS
  - Homepage visit is REQUIRED -- sets JS-generated cookies that enable search
  - Direct navigation to search URLs (without warmup) is always blocked
  - Regular HTTP clients (requests, urllib, curl_cffi) cannot bypass because
    eBay's anti-bot requires JavaScript execution for cookie generation

IP-LEVEL THROTTLING:
  - eBay applies IP-level rate limiting after repeated automated access
  - After ~10-20 automated requests, the IP gets temporarily banned (minutes)
  - The "Pardon Our Interruption" page runs heavy JS that crashes headless Chromium
  - Production use needs: delays between requests, rotating IPs, or proxy rotation

WHAT WORKS (when not IP-banned):
  - Homepage warmup -> sold listings search: OK
  - Pagination (page 2, 3, etc.): OK
  - Rapid sequential queries (5+): OK within a single session
  - Listing titles and prices: extractable via .s-item selectors
  - Both .com and .co.uk behave identically

WHAT DOESN'T WORK:
  - Direct URL navigation without homepage warmup
  - curl_cffi TLS impersonation (eBay checks JS cookies, not just TLS)
  - requests/urllib with session cookies (no JS execution)
  - Firefox headless (same bot detection)

REQUIRED STRATEGY FOR PRODUCTION:
  1. Launch Chromium with stealth plugin
  2. Visit ebay.com homepage first (sets ~30+ cookies via JS)
  3. Wait 1-2s for JS to complete
  4. Navigate to search URLs
  5. Add delays (2-5s) between pages to avoid IP ban
  6. Handle "Pardon" page by waiting/retrying (catches temporary blocks)

Usage:
    python scripts/test_ebay_playwright.py
"""

import sys
import time
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from playwright_stealth import Stealth

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

SEARCH_URL = "https://www.ebay.com/sch/i.html?_nkw=pokemon+charizard+base+set&LH_Complete=1&LH_Sold=1"


def is_blocked(title):
    return any(kw in title.lower() for kw in
               ["pardon", "security", "captcha", "robot", "interruption"])


def extract_via_js(page, max_items=5):
    """Extract listing data via JS evaluation (avoids pulling full DOM)."""
    return page.evaluate(f"""() => {{
        const items = [];
        document.querySelectorAll('.s-item').forEach((el, i) => {{
            if (i >= {max_items + 2}) return;
            const titleEl = el.querySelector('.s-item__title');
            const priceEl = el.querySelector('.s-item__price');

            let soldDate = null;
            const posSel = el.querySelector(
                '.s-item__caption .POSITIVE, .s-item__title--tag .POSITIVE');
            if (posSel) soldDate = posSel.textContent.trim();
            if (!soldDate) {{
                el.querySelectorAll('span').forEach(s => {{
                    const t = s.textContent.trim();
                    if (t.includes('Sold') && t.length < 40) soldDate = t;
                }});
            }}

            const title = titleEl ? titleEl.textContent.trim() : null;
            if (title && title !== 'Shop on eBay'
                && title !== 'Results matching fewer words') {{
                items.push({{
                    title: title.substring(0, 120),
                    price: priceEl ? priceEl.textContent.trim() : null,
                    soldDate,
                }});
            }}
        }});
        return items.slice(0, {max_items});
    }}""")


def run_test():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1366, "height": 768},
            user_agent=UA,
            locale="en-US",
            timezone_id="America/New_York",
        )
        page = context.new_page()
        Stealth().apply_stealth_sync(page)

        # ====== STEP 1: Homepage warmup ======
        print("=" * 60)
        print("STEP 1: Homepage warmup (sets required JS cookies)")
        print("=" * 60)
        t0 = time.time()
        try:
            page.goto("https://www.ebay.com",
                       wait_until="domcontentloaded", timeout=20000)
            time.sleep(2)
        except Exception as e:
            print(f"  FAIL: {e}")
            browser.close()
            return False

        cookies = context.cookies()
        print(f"  Title: {page.title()}")
        print(f"  Cookies: {len(cookies)} ({time.time() - t0:.1f}s)")

        # ====== STEP 2: Sold listings search ======
        print(f"\n{'=' * 60}")
        print("STEP 2: Sold listings search")
        print("=" * 60)
        try:
            page.goto(SEARCH_URL,
                       wait_until="domcontentloaded", timeout=20000)
            time.sleep(3)
        except Exception as e:
            print(f"  Navigation error: {e}")
            browser.close()
            return False

        try:
            title = page.title()
        except Exception as e:
            print(f"  Page crashed (likely IP-banned): {e}")
            print("  eBay's block page crashes headless Chromium.")
            print("  Wait a few minutes and retry.")
            browser.close()
            return False

        print(f"  Title: {title}")

        if is_blocked(title):
            print("  BLOCKED by eBay bot detection")
            print("  This IP is likely temporarily banned from earlier tests.")
            print("  The homepage warmup strategy works when not IP-banned.")
            browser.close()
            return False

        print("  Bot detection BYPASSED!")

        # Extract listings
        try:
            listings = extract_via_js(page, 5)
        except Exception as e:
            print(f"  Extraction error: {e}")
            listings = []

        if listings:
            print(f"\n  Extracted {len(listings)} listings:")
            for i, item in enumerate(listings, 1):
                print(f"\n  {i}. {item['title'][:80]}")
                print(f"     Price: {item['price']}"
                      f"  |  Sold: {item.get('soldDate') or 'N/A'}")

        # ====== STEP 3: Sold date analysis ======
        print(f"\n{'=' * 60}")
        print("STEP 3: Sold date extraction")
        print("=" * 60)
        has_dates = any(item.get("soldDate") for item in listings)
        print(f"  Dates found: {'YES' if has_dates else 'NO'}")

        if not has_dates and listings:
            try:
                debug = page.evaluate("""() => {
                    const item = document.querySelectorAll('.s-item')[1];
                    if (!item) return [];
                    const spans = [];
                    item.querySelectorAll('span').forEach(s => {
                        const t = s.textContent.trim();
                        const c = s.className || '';
                        if (t && t.length < 80) spans.push({c, t});
                    });
                    return spans;
                }""")
                print("  DEBUG - All spans in first listing:")
                for s in debug[:25]:
                    print(f"    [{s['c']}] {s['t']}")
            except Exception:
                pass

        # ====== STEP 4: Pagination ======
        print(f"\n{'=' * 60}")
        print("STEP 4: Pagination (page 2)")
        print("=" * 60)
        try:
            page.goto(SEARCH_URL + "&_pgn=2",
                       wait_until="domcontentloaded", timeout=15000)
            time.sleep(2)
            t2 = page.title()
            print(f"  Title: {t2}")
            if is_blocked(t2):
                print("  BLOCKED on page 2")
            else:
                p2 = extract_via_js(page, 3)
                if p2:
                    print(f"  Page 2 OK ({len(p2)} listings):")
                    for i, item in enumerate(p2, 1):
                        print(f"  {i}. {item['title'][:70]} | {item['price']}")
                else:
                    print("  No listings on page 2")
        except Exception as e:
            print(f"  Error: {e}")

        # ====== STEP 5: Rate limiting ======
        print(f"\n{'=' * 60}")
        print("STEP 5: Rate limiting (5 rapid requests)")
        print("=" * 60)
        cards = ["pikachu", "blastoise", "venusaur", "mewtwo", "gyarados"]
        blocked_at = None
        for i, card in enumerate(cards, 1):
            url = (f"https://www.ebay.com/sch/i.html?"
                   f"_nkw=pokemon+{card}&LH_Complete=1&LH_Sold=1")
            t0 = time.time()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
                time.sleep(1)
                el = time.time() - t0
                t = page.title()
                bl = is_blocked(t)
                print(f"  {i}. {card}: {'BLOCKED' if bl else 'OK'} ({el:.1f}s)")
                if bl:
                    blocked_at = i
                    break
            except Exception as e:
                print(f"  {i}. {card}: ERROR ({e})")
                blocked_at = i
                break

        # ====== SUMMARY ======
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print("=" * 60)
        print(f"  Bot bypass:         YES (stealth + homepage warmup)")
        print(f"  Listings extracted: {len(listings)}")
        print(f"  Sold dates:         "
              f"{'YES' if has_dates else 'Needs selector investigation'}")
        print(f"  Pagination:         Tested")
        rate_msg = (f"Blocked after {blocked_at} requests"
                    if blocked_at else "None detected (5/5 OK)")
        print(f"  Rate limiting:      {rate_msg}")
        print(f"  Required setup:     playwright-stealth + homepage warmup")

        browser.close()
        return True


if __name__ == "__main__":
    ok = run_test()
    sys.exit(0 if ok else 1)
