"""
TCGPlayer per-condition sales history scraper using Playwright.

Navigates to product pages, intercepts XHR calls to the latestsales
endpoint, and stores results in a separate SQLite database.

Usage:
    python -m cardprice.scrapers.tcgplayer_sales          # test single card
    python -m cardprice.scrapers.tcgplayer_sales --batch   # scrape priority queue
"""
from __future__ import annotations

import logging
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.sync_api import (
    Page,
    Response,
    sync_playwright,
)

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "tcgplayer_sales.db"

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

_DDL = """\
CREATE TABLE IF NOT EXISTS tcgplayer_sales (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tcg_product_id  INTEGER NOT NULL,
    sale_date       TEXT    NOT NULL,   -- ISO-8601
    sale_price      REAL    NOT NULL,
    condition       TEXT,
    quantity        INTEGER DEFAULT 1,
    printing        TEXT,               -- e.g. "Normal", "Holofoil"
    scraped_at      TEXT    NOT NULL    -- ISO-8601
);

CREATE INDEX IF NOT EXISTS ix_sales_pid_date
    ON tcgplayer_sales (tcg_product_id, sale_date);

CREATE UNIQUE INDEX IF NOT EXISTS ix_sales_dedup
    ON tcgplayer_sales (tcg_product_id, sale_date, sale_price, condition, printing);

CREATE TABLE IF NOT EXISTS scrape_log (
    tcg_product_id  INTEGER PRIMARY KEY,
    last_scraped    TEXT NOT NULL,
    sales_count     INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'ok'
);

-- Live listings snapshots, captured from the mp-search-api /listings XHR that
-- fires on the SAME product-page visit the sales scrape already performs
-- (zero additional page loads). The default payload is the ~10 cheapest
-- listings across conditions/printings; total_results is the full listing
-- depth at capture time (a liquidity signal). Append-only snapshots.
CREATE TABLE IF NOT EXISTS tcgplayer_listings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tcg_product_id  INTEGER NOT NULL,
    listing_id      INTEGER,
    condition       TEXT,
    printing        TEXT,
    language        TEXT,
    price           REAL,
    shipping_price  REAL,
    quantity        INTEGER,
    seller_name     TEXT,
    direct_seller   INTEGER,
    verified_seller INTEGER,
    seller_rating   REAL,
    total_results   INTEGER,
    scraped_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_listings_pid_time
    ON tcgplayer_listings (tcg_product_id, scraped_at);
"""


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(_DDL)
    return conn


def _insert_sales(
    conn: sqlite3.Connection,
    product_id: int,
    sales: list[dict[str, Any]],
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for s in sales:
        condition = s.get("condition", "")
        # Skip chart data / entries without a real condition
        if not condition or "to" in s.get("sale_date", ""):
            continue
        rows.append((
            product_id,
            s.get("sale_date", ""),
            s.get("sale_price", 0.0),
            condition,
            s.get("quantity", 1),
            s.get("printing", ""),
            now,
        ))
    conn.executemany(
        "INSERT OR IGNORE INTO tcgplayer_sales "
        "(tcg_product_id, sale_date, sale_price, condition, quantity, printing, scraped_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    # Update scrape log
    conn.execute(
        "INSERT OR REPLACE INTO scrape_log (tcg_product_id, last_scraped, sales_count, status) "
        "VALUES (?, ?, ?, 'ok')",
        (product_id, now, len(rows)),
    )
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Condition code mapping (TCGPlayer internal → human-readable)
# ---------------------------------------------------------------------------

CONDITION_MAP = {
    "1": "Near Mint",
    "2": "Lightly Played",
    "3": "Moderately Played",
    "4": "Heavily Played",
    "5": "Damaged",
}

PRINTING_MAP = {
    "1": "Normal",
    "2": "Holofoil",
    "3": "Reverse Holofoil",
    "4": "1st Edition Normal",
    "5": "1st Edition Holofoil",
}

# ---------------------------------------------------------------------------
# XHR interception approach
# ---------------------------------------------------------------------------


def _parse_latestsales_response(data: dict) -> list[dict[str, Any]]:
    """Parse the JSON from the latestsales endpoint."""
    sales = []
    for item in data.get("data", []):
        sale = {
            "sale_date": item.get("orderDate", ""),
            "sale_price": item.get("purchasePrice", 0.0),
            "condition": item.get("condition", ""),
            "quantity": item.get("quantity", 1),
            "printing": item.get("variant", "") or item.get("printing", ""),
        }
        # Clean up date — strip trailing Z and microseconds for consistency
        if sale["sale_date"]:
            sale["sale_date"] = sale["sale_date"].replace("Z", "+00:00")
        sales.append(sale)
    return sales


def _parse_listings_response(data: dict) -> list[dict[str, Any]]:
    """Parse the mp-search-api /v1/product/{id}/listings payload.

    Shape: {"errors": [], "results": [{"totalResults": N, "results": [listing,
    ...]}]}. Each listing has price/shippingPrice/condition/printing/quantity/
    sellerName/directSeller/verifiedSeller/listingId/... (verified 2026-07-22).
    """
    out = []
    for block in data.get("results", []):
        if not isinstance(block, dict):
            continue
        total = block.get("totalResults")
        for l in block.get("results", []):
            if not isinstance(l, dict) or l.get("price") is None:
                continue
            out.append({
                "listing_id": int(l["listingId"]) if l.get("listingId") else None,
                "condition": l.get("condition", ""),
                "printing": l.get("printing", ""),
                "language": l.get("languageAbbreviation", ""),
                "price": float(l.get("price", 0) or 0),
                "shipping_price": float(l.get("sellerShippingPrice",
                                              l.get("shippingPrice", 0)) or 0),
                "quantity": int(l.get("quantity", 1) or 1),
                "seller_name": l.get("sellerName", ""),
                "direct_seller": int(bool(l.get("directSeller"))),
                "verified_seller": int(bool(l.get("verifiedSeller"))),
                "seller_rating": float(l.get("sellerRating", 0) or 0),
                "total_results": int(total) if total is not None else None,
            })
    return out


def _insert_listings(
    conn: sqlite3.Connection,
    product_id: int,
    listings: list[dict[str, Any]],
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    rows = [(product_id, l.get("listing_id"), l.get("condition"),
             l.get("printing"), l.get("language"), l.get("price"),
             l.get("shipping_price"), l.get("quantity"), l.get("seller_name"),
             l.get("direct_seller"), l.get("verified_seller"),
             l.get("seller_rating"), l.get("total_results"), now)
            for l in listings]
    conn.executemany(
        "INSERT INTO tcgplayer_listings "
        "(tcg_product_id, listing_id, condition, printing, language, price, "
        "shipping_price, quantity, seller_name, direct_seller, "
        "verified_seller, seller_rating, total_results, scraped_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def _parse_pricepoints_response(data: dict) -> list[dict[str, Any]]:
    """Parse the infinite-api price/history/detailed endpoint (aggregated stats)."""
    points = []
    # infinite-api format: {count: N, result: [{skuId, variant, condition, ...}]}
    results = data.get("result", data.get("data", []))
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            point = {
                "sale_date": item.get("fetchedAt", item.get("date", "")),
                "sale_price": float(item.get("marketPrice", 0) or 0),
                "condition": item.get("condition", ""),
                "quantity": int(item.get("totalQuantitySold", 1) or 1),
                "printing": item.get("variant", item.get("printing", "")),
            }
            if point["sale_price"] > 0:
                points.append(point)
    return points


def scrape_product_sales(
    page: Page,
    product_id: int,
    *,
    timeout_ms: int = 30_000,
    capture_listings: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Navigate to a TCGPlayer product page and capture sales data.

    Strategy:
    1. Set up network interception for latestsales and pricepoints
    2. Navigate to product page
    3. Click "Sales History" tab if present
    4. Wait for XHR response
    5. Fall back to DOM scraping if XHR interception fails
    """
    url = f"https://www.tcgplayer.com/product/{product_id}"
    captured_sales: list[dict[str, Any]] = []
    captured_pricepoints: list[dict[str, Any]] = []
    seen_sale_keys: set[tuple] = set()

    def _dedup_key(s: dict) -> tuple:
        return (s.get("sale_date", ""), s.get("sale_price", 0), s.get("condition", ""), s.get("printing", ""))

    def on_response(response: Response) -> None:
        nonlocal captured_sales, captured_pricepoints
        req_url = response.url
        try:
            if "latestsales" in req_url and response.status == 200:
                try:
                    body = response.json()
                    parsed = _parse_latestsales_response(body)
                    new = 0
                    for s in parsed:
                        key = _dedup_key(s)
                        if key not in seen_sale_keys:
                            seen_sale_keys.add(key)
                            captured_sales.append(s)
                            new += 1
                    log.info("Intercepted %d sales from latestsales (%d new)", len(parsed), new)
                except Exception as e:
                    log.warning("Failed to parse latestsales: %s", e)

            elif ("/listings" in req_url and response.status == 200
                    and capture_listings is not None):
                try:
                    body = response.json()
                    parsed = _parse_listings_response(body)
                    seen_ids = {l.get("listing_id") for l in capture_listings}
                    new = [l for l in parsed
                           if l.get("listing_id") not in seen_ids
                           or l.get("listing_id") is None]
                    capture_listings.extend(new)
                    log.info("Intercepted %d listings (%d new)", len(parsed), len(new))
                except Exception as e:
                    log.warning("Failed to parse listings: %s", e)

            elif ("price/history" in req_url) and response.status == 200:
                try:
                    body = response.json()
                    parsed = _parse_pricepoints_response(body)
                    captured_pricepoints.extend(parsed)
                    log.info("Intercepted %d pricepoints", len(parsed))
                except Exception as e:
                    log.warning("Failed to parse pricepoints: %s", e)
        except Exception:
            pass  # response may have been disposed

    page.on("response", on_response)

    try:
        log.info("Navigating to %s", url)
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

        # Wait for page to stabilize
        page.wait_for_timeout(2000)

        # Try to find and click "Sales History" tab/section
        _click_sales_tab(page)

        # Wait for XHR to fire
        page.wait_for_timeout(4000)

        # Try clicking "View More Data" to get more sales
        _click_view_more(page)
        page.wait_for_timeout(3000)

        # If we got latestsales data, use it
        if captured_sales:
            log.info("Got %d sales via XHR interception", len(captured_sales))
            return captured_sales

        # Try scrolling to sales section to trigger lazy load
        _scroll_to_sales(page)
        page.wait_for_timeout(3000)

        if captured_sales:
            log.info("Got %d sales after scroll", len(captured_sales))
            return captured_sales

        # Try DOM scraping as fallback
        dom_sales = _scrape_sales_from_dom(page)
        if dom_sales:
            log.info("Got %d sales from DOM scraping", len(dom_sales))
            return dom_sales

        # Last resort: use pricepoints
        if captured_pricepoints:
            log.info("Using %d pricepoints as fallback", len(captured_pricepoints))
            return captured_pricepoints

        log.warning("No sales data found for product %d", product_id)
        return []

    finally:
        page.remove_listener("response", on_response)


def _click_sales_tab(page: Page) -> None:
    """Try various selectors to click the sales history tab."""
    selectors = [
        'text="Sales History"',
        'text="Last Sold"',
        '[data-testid="sales-tab"]',
        'button:has-text("Sales")',
        'a:has-text("Sales History")',
        '.sales-history-tab',
        '[class*="sales"] button',
        '[class*="Sales"] button',
        # TCGPlayer uses tab-like navigation
        'div[role="tab"]:has-text("Sales")',
        'li:has-text("Sales")',
    ]
    for sel in selectors:
        try:
            el = page.query_selector(sel)
            if el and el.is_visible():
                el.click()
                log.info("Clicked sales tab with selector: %s", sel)
                page.wait_for_timeout(2000)
                return
        except Exception:
            continue
    log.debug("No sales tab found, page may show sales inline")


def _click_view_more(page: Page) -> None:
    """Try clicking 'View More Data' link to navigate to full sales history."""
    selectors = [
        'text="View More Data"',
        '.latest-sales__header__history',
        'a:has-text("View More")',
        'span:has-text("View More Data")',
    ]
    for sel in selectors:
        try:
            el = page.query_selector(sel)
            if el and el.is_visible():
                el.click()
                log.info("Clicked 'View More Data' with selector: %s", sel)
                page.wait_for_timeout(3000)
                return
        except Exception:
            continue
    log.debug("No 'View More Data' link found")


def _scroll_to_sales(page: Page) -> None:
    """Scroll down to trigger lazy-loaded sales section."""
    try:
        # Try scrolling to a sales-related element
        page.evaluate("""
            const els = document.querySelectorAll(
                '[class*="sales"], [class*="Sales"], [class*="history"], [class*="History"]'
            );
            if (els.length > 0) {
                els[0].scrollIntoView({ behavior: 'smooth' });
            } else {
                window.scrollTo(0, document.body.scrollHeight * 0.7);
            }
        """)
    except Exception:
        pass


def _scrape_sales_from_dom(page: Page) -> list[dict[str, Any]]:
    """
    Attempt to scrape sales data directly from the rendered DOM.
    TCGPlayer renders a table of recent sales.
    """
    sales = []
    try:
        # Try to find the sales table rows
        rows = page.evaluate("""
            () => {
                const results = [];

                // Look for the last-sold / sales-history table
                const tables = document.querySelectorAll('table');
                for (const table of tables) {
                    const trs = table.querySelectorAll('tbody tr');
                    for (const tr of trs) {
                        const cells = tr.querySelectorAll('td');
                        if (cells.length >= 3) {
                            results.push({
                                texts: Array.from(cells).map(c => c.textContent.trim())
                            });
                        }
                    }
                }

                // Also look for non-table sales listings
                const saleItems = document.querySelectorAll(
                    '[class*="sale-item"], [class*="SaleItem"], ' +
                    '[class*="last-sold"], [class*="LastSold"], ' +
                    '[class*="sales-data"], [class*="SalesData"]'
                );
                for (const item of saleItems) {
                    results.push({
                        texts: [item.textContent.trim()]
                    });
                }

                return results;
            }
        """)

        for row in rows:
            texts = row.get("texts", [])
            if len(texts) >= 3:
                sale = _parse_dom_row(texts)
                if sale:
                    sales.append(sale)

    except Exception as e:
        log.debug("DOM scraping failed: %s", e)

    return sales


def _parse_dom_row(texts: list[str]) -> dict[str, Any] | None:
    """Try to parse a table row into a sale record."""
    sale: dict[str, Any] = {
        "sale_date": "",
        "sale_price": 0.0,
        "condition": "",
        "quantity": 1,
        "printing": "",
    }
    for text in texts:
        text = text.strip()
        # Price detection
        if text.startswith("$"):
            try:
                sale["sale_price"] = float(text.replace("$", "").replace(",", ""))
            except ValueError:
                pass
        # Date detection (various formats)
        elif "/" in text and any(c.isdigit() for c in text):
            sale["sale_date"] = text
        # Condition detection
        elif text in (
            "Near Mint", "Lightly Played", "Moderately Played",
            "Heavily Played", "Damaged", "NM", "LP", "MP", "HP", "DMG",
        ):
            sale["condition"] = text
        # Quantity
        elif text.isdigit():
            sale["quantity"] = int(text)
        # Printing
        elif text in ("Normal", "Holofoil", "Reverse Holofoil", "1st Edition"):
            sale["printing"] = text

    if sale["sale_price"] > 0:
        return sale
    return None


# ---------------------------------------------------------------------------
# Browser management
# ---------------------------------------------------------------------------


def create_browser_context(playwright):
    """Create a stealth-ish browser context."""
    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )
    context = browser.new_context(
        viewport={"width": 1440, "height": 900},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        timezone_id="America/New_York",
        java_script_enabled=True,
    )
    # Remove webdriver flag
    context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        // Overwrite the `plugins` property to use a custom getter.
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });
        // Overwrite the `languages` property to use a custom getter.
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });
    """)
    return browser, context


def _worker(
    worker_id: int,
    product_ids: list[int],
    delay_range: tuple[float, float],
    max_errors: int,
    browser_restart_every: int,
) -> dict[int, int]:
    """Single worker: one browser, scrapes its chunk of product IDs."""
    conn = _get_db()
    results: dict[int, int] = {}
    consecutive_errors = 0

    with sync_playwright() as pw:
        browser, context = create_browser_context(pw)
        page = context.new_page()

        for i, pid in enumerate(product_ids):
            log.info("[W%d] Scraping product %d (%d/%d)", worker_id, pid, i + 1, len(product_ids))

            if i > 0 and i % browser_restart_every == 0:
                log.info("[W%d] Restarting browser after %d products", worker_id, i)
                try:
                    browser.close()
                except Exception:
                    pass
                browser, context = create_browser_context(pw)
                page = context.new_page()

            try:
                listings_buf: list[dict[str, Any]] = []
                sales = scrape_product_sales(page, pid, capture_listings=listings_buf)
                if listings_buf:
                    nl = _insert_listings(conn, pid, listings_buf)
                    log.info("[W%d] Stored %d listings for product %d", worker_id, nl, pid)
                if sales:
                    n = _insert_sales(conn, pid, sales)
                    results[pid] = n
                    log.info("[W%d] Stored %d sales for product %d", worker_id, n, pid)
                else:
                    results[pid] = 0
                    conn.execute(
                        "INSERT OR REPLACE INTO scrape_log "
                        "(tcg_product_id, last_scraped, sales_count, status) "
                        "VALUES (?, ?, 0, 'empty')",
                        (pid, datetime.now(timezone.utc).isoformat()),
                    )
                    conn.commit()
                consecutive_errors = 0
            except Exception as e:
                log.error("[W%d] Error scraping product %d: %s", worker_id, pid, e)
                results[pid] = -1
                consecutive_errors += 1
                conn.execute(
                    "INSERT OR REPLACE INTO scrape_log "
                    "(tcg_product_id, last_scraped, sales_count, status) "
                    "VALUES (?, ?, 0, ?)",
                    (pid, datetime.now(timezone.utc).isoformat(), f"error: {e}"),
                )
                conn.commit()
                if consecutive_errors >= max_errors:
                    log.warning("[W%d] Hit %d consecutive errors, restarting browser", worker_id, consecutive_errors)
                    try:
                        browser.close()
                    except Exception:
                        pass
                    browser, context = create_browser_context(pw)
                    page = context.new_page()
                    consecutive_errors = 0
                    max_errors = max_errors // 2 or 1

            if i < len(product_ids) - 1:
                delay = random.uniform(*delay_range)
                time.sleep(delay)

        browser.close()

    conn.close()
    return results


def scrape_batch(
    product_ids: list[int],
    *,
    delay_range: tuple[float, float] = (2.0, 4.0),
    max_errors: int = 5,
    browser_restart_every: int = 100,
    workers: int = 1,
) -> dict[int, int]:
    """
    Scrape sales for a batch of product IDs.
    Returns {product_id: num_sales_stored}.

    Uses `workers` parallel browser instances to speed up scraping.
    Restarts each browser every `browser_restart_every` products.
    """
    if workers <= 1:
        return _worker(0, product_ids, delay_range, max_errors, browser_restart_every)

    # Split product IDs into chunks, one per worker
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    chunks = [[] for _ in range(workers)]
    for i, pid in enumerate(product_ids):
        chunks[i % workers].append(pid)

    log.info("Launching %d parallel workers (%d-%d products each)",
             workers, min(len(c) for c in chunks if c), max(len(c) for c in chunks if c))

    all_results: dict[int, int] = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_worker, wid, chunk, delay_range, max_errors, browser_restart_every): wid
            for wid, chunk in enumerate(chunks) if chunk
        }
        for future in as_completed(futures):
            wid = futures[future]
            try:
                result = future.result()
                all_results.update(result)
                success = sum(1 for v in result.values() if v >= 0)
                log.info("Worker %d finished: %d/%d succeeded", wid, success, len(result))
            except Exception as e:
                log.error("Worker %d crashed: %s", wid, e)

    return all_results


# ---------------------------------------------------------------------------
# Debug / exploration helper
# ---------------------------------------------------------------------------


def explore_page(product_id: int) -> dict[str, Any]:
    """
    Navigate to a product page and capture everything interesting:
    all XHR URLs, page title, sales-related DOM elements.
    Returns a diagnostic dict.
    """
    info: dict[str, Any] = {
        "product_id": product_id,
        "xhr_urls": [],
        "title": "",
        "sales_elements": [],
        "sales_data": [],
        "pricepoints": [],
    }

    with sync_playwright() as pw:
        browser, context = create_browser_context(pw)
        page = context.new_page()

        def on_response(response: Response) -> None:
            url = response.url
            if "mpapi" in url or "api" in url:
                entry = {
                    "url": url,
                    "status": response.status,
                }
                try:
                    if response.status == 200 and "json" in response.headers.get("content-type", ""):
                        body = response.json()
                        entry["body_preview"] = str(body)[:500]
                        if "latestsales" in url:
                            info["sales_data"] = _parse_latestsales_response(body)
                        elif "pricepoints" in url:
                            info["pricepoints"] = _parse_pricepoints_response(body)
                except Exception:
                    pass
                info["xhr_urls"].append(entry)

        page.on("response", on_response)

        page.goto(
            f"https://www.tcgplayer.com/product/{product_id}",
            wait_until="domcontentloaded",
            timeout=30_000,
        )
        page.wait_for_timeout(3000)

        info["title"] = page.title()

        # Click sales tab
        _click_sales_tab(page)
        page.wait_for_timeout(3000)

        # Scroll
        _scroll_to_sales(page)
        page.wait_for_timeout(3000)

        # Capture relevant DOM elements
        info["sales_elements"] = page.evaluate("""
            () => {
                const interesting = [];
                // Find all elements with sales/history in class or text
                const all = document.querySelectorAll('*');
                for (const el of all) {
                    const cls = el.className || '';
                    const id = el.id || '';
                    if (typeof cls === 'string' &&
                        (cls.toLowerCase().includes('sale') ||
                         cls.toLowerCase().includes('history') ||
                         cls.toLowerCase().includes('last-sold') ||
                         id.toLowerCase().includes('sale'))) {
                        interesting.push({
                            tag: el.tagName,
                            class: cls.substring(0, 100),
                            id: id,
                            text: el.textContent.substring(0, 200),
                        });
                    }
                }
                return interesting.slice(0, 30);
            }
        """)

        browser.close()

    return info


# ---------------------------------------------------------------------------
# Test / CLI
# ---------------------------------------------------------------------------


def test_single(product_id: int = 42382) -> None:
    """Test scraping a single product (default: Base Set Charizard)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"\n{'='*60}")
    print(f"  TCGPlayer Sales Scraper — Testing product {product_id}")
    print(f"  URL: https://www.tcgplayer.com/product/{product_id}")
    print(f"{'='*60}\n")

    # First, explore the page to understand what we're dealing with
    print("[1/3] Exploring page structure...")
    info = explore_page(product_id)

    print(f"\n  Page title: {info['title']}")
    print(f"  XHR calls captured: {len(info['xhr_urls'])}")
    for xhr in info["xhr_urls"]:
        status_icon = "✓" if xhr["status"] == 200 else "✗"
        # Truncate URL for display
        short_url = xhr["url"][:100]
        print(f"    {status_icon} [{xhr['status']}] {short_url}")
        if "body_preview" in xhr:
            print(f"      Preview: {xhr['body_preview'][:200]}")

    print(f"\n  Sales-related DOM elements: {len(info['sales_elements'])}")
    for el in info["sales_elements"][:10]:
        print(f"    <{el['tag']} class='{el['class'][:60]}'>")
        print(f"      {el['text'][:100]}")

    print(f"\n  Sales from XHR: {len(info['sales_data'])}")
    for s in info["sales_data"][:5]:
        print(f"    {s['sale_date'][:10]}  ${s['sale_price']:>8.2f}  "
              f"{s['condition']:20s}  {s['printing']}")

    print(f"\n  Pricepoints from XHR: {len(info['pricepoints'])}")
    for p in info["pricepoints"][:5]:
        print(f"    {p.get('sale_date', '?')[:10]}  ${p['sale_price']:>8.2f}  "
              f"{p.get('condition', '?'):20s}")

    # Now do the actual scrape
    print(f"\n[2/3] Running scraper...")
    with sync_playwright() as pw:
        browser, context = create_browser_context(pw)
        page = context.new_page()
        sales = scrape_product_sales(page, product_id)
        browser.close()

    print(f"  Captured {len(sales)} sales records")
    for s in sales[:10]:
        print(f"    {s.get('sale_date', '?')[:19]:20s}  "
              f"${s.get('sale_price', 0):>8.2f}  "
              f"{s.get('condition', '?'):20s}  "
              f"qty={s.get('quantity', 1)}  "
              f"{s.get('printing', '?')}")

    # Store in DB
    if sales:
        print(f"\n[3/3] Storing in {DB_PATH}...")
        conn = _get_db()
        n = _insert_sales(conn, product_id, sales)
        conn.close()
        print(f"  Stored {n} rows")
    else:
        print("\n[3/3] No sales to store")

    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys

    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 42382
    test_single(pid)
