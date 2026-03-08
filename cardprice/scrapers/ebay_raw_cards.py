"""eBay raw (ungraded) card scraper for condition training data.

Scrapes completed/sold listings for raw Pokemon cards, downloading photos
organized by seller-stated condition (NM/LP/MP/HP/DMG) for ML training.

Completely standalone from the card identification pipeline — this is a
training data collection tool.

Usage (CLI):
    python -m cardprice.scrapers.ebay_raw_cards --condition LP --pages 5
    python -m cardprice.scrapers.ebay_raw_cards --all-conditions --pages 3
    python -m cardprice.scrapers.ebay_raw_cards --resume
    python -m cardprice.scrapers.ebay_raw_cards --stats

Usage (programmatic):
    from cardprice.scrapers.ebay_raw_cards import RawCardScraper
    scraper = RawCardScraper()
    scraper.scrape_condition("LP", max_pages=5)
"""

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEARCH_URL = "https://www.ebay.com/sch/i.html"
POKEMON_CATEGORY = "183454"

# Negative keywords to exclude graded cards
GRADED_EXCLUSIONS = "-PSA -BGS -CGC -SGC -ACE -AGS -slab -slabbed -graded"

# Rate limiting
MIN_DELAY = 2.5
MAX_DELAY = 4.0
LISTING_DELAY_MIN = 1.5
LISTING_DELAY_MAX = 3.0
IMAGE_DELAY = 0.3

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

# Condition search queries — what sellers actually write in titles
CONDITION_QUERIES = {
    "NM": ['"near mint"', '"NM"', '"pack fresh"', '"mint condition"'],
    "LP": ['"lightly played"', '"LP"', '"light play"'],
    "MP": ['"moderately played"', '"MP"', '"moderate play"'],
    "HP": ['"heavily played"', '"HP"', '"heavy play"'],
    "DMG": ['"damaged"', '"DMG"', '"poor condition"', '"creased"'],
}

# Condition detection from titles (reuse patterns from ebay_title_parser)
CONDITION_PATTERNS = {
    "NM": re.compile(r"\bNM\b|\bNear\s*Mint\b|\bPack\s*Fresh\b|\bMint\s*Condition\b", re.IGNORECASE),
    "LP": re.compile(r"\bLP\b|\bLight(?:ly)?\s*Play(?:ed)?\b", re.IGNORECASE),
    "MP": re.compile(r"\bMP\b|\bModer?at(?:e|ely)\s*Play(?:ed)?\b", re.IGNORECASE),
    "HP": re.compile(r"\bHP\b|\bHeav(?:y|ily)\s*Play(?:ed)?\b", re.IGNORECASE),
    "DMG": re.compile(r"\bDMG\b|\bDamaged\b|\bPoor\b|\bCreased\b", re.IGNORECASE),
}

# Graded card detection (to filter out false positives)
GRADED_PATTERN = re.compile(
    r"\b(PSA|BGS|CGC|SGC|ACE|AGS)\s*\d", re.IGNORECASE
)

OUTPUT_DIR = "data/condition_training/raw"
PROGRESS_FILE = "raw_scrape_progress.json"

# eBay domains to try (fallback if .com blocks us)
EBAY_DOMAINS = ["www.ebay.com", "www.ebay.co.uk"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RawListing:
    """A single raw (ungraded) card listing from eBay."""
    item_id: str
    title: str
    condition: str  # NM, LP, MP, HP, DMG
    card_name: str | None = None
    set_name: str | None = None
    card_number: str | None = None
    sold_price: float | None = None
    sold_date: str | None = None
    listing_url: str = ""
    image_urls: list[str] = field(default_factory=list)
    local_images: list[str] = field(default_factory=list)
    num_images: int = 0
    scraped_at: str = ""


@dataclass
class ScrapeProgress:
    """Resume-capable progress tracking."""
    current_condition: str = ""
    current_query_idx: int = 0
    current_page: int = 0
    total_listings: int = 0
    total_images: int = 0
    completed_conditions: list[str] = field(default_factory=list)
    seen_item_ids: set[str] = field(default_factory=set)
    started_at: str = ""
    last_updated: str = ""
    errors: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seen_item_ids"] = list(d["seen_item_ids"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ScrapeProgress":
        d["seen_item_ids"] = set(d.get("seen_item_ids", []))
        return cls(**d)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_session() -> requests.Session:
    s = requests.Session()
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _get_headers() -> dict:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.ebay.com/",
        "DNT": "1",
    }


def _polite_delay(min_s: float, max_s: float):
    time.sleep(random.uniform(min_s, max_s))


def _is_bot_blocked(html: str) -> bool:
    """Check if eBay returned a bot challenge page."""
    return "Pardon Our Interruption" in html or "captcha" in html.lower()


# ---------------------------------------------------------------------------
# Search & parse
# ---------------------------------------------------------------------------

def _build_search_url(
    condition_query: str,
    page: int = 1,
    domain: str = "www.ebay.com",
) -> str:
    """Build eBay sold-listings search URL for raw Pokemon cards."""
    query = f"pokemon card {condition_query} {GRADED_EXCLUSIONS}"
    params = {
        "_nkw": query,
        "_sacat": POKEMON_CATEGORY,
        "LH_Complete": "1",
        "LH_Sold": "1",
        "LH_PrefLoc": "1",  # US only (more consistent condition labeling)
        "_sop": "13",  # most recent first
        "_ipg": "60",
    }
    if page > 1:
        params["_pgn"] = str(page)
    return f"https://{domain}/sch/i.html?{urlencode(params)}"


def _parse_search_results(html: str) -> list[dict]:
    """Parse eBay search results into listing stubs."""
    soup = BeautifulSoup(html, "lxml")

    # Try new eBay layout first (li.s-card), then legacy (li.s-item)
    items = soup.select("li.s-card[data-listingid]")
    if items:
        return _parse_s_card_results(items)

    items = soup.select("li.s-item")
    return _parse_s_item_results(items)


def _parse_s_card_results(items) -> list[dict]:
    """Parse eBay's 2025+ s-card layout."""
    results = []
    for item_el in items:
        listing_id = item_el.get("data-listingid", "")
        if not listing_id:
            continue

        title_el = item_el.select_one("span[role='heading']") or item_el.select_one("a.bsig__title__wrapper")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)

        link_el = item_el.select_one("a[href*='/itm/']")
        listing_url = ""
        if link_el:
            listing_url = link_el["href"].split("?")[0]

        price_el = item_el.select_one("span.bsig__price") or item_el.select_one("span.s-item__price")
        sold_price = _extract_price(price_el)

        sold_date = None
        for span in item_el.select("span"):
            txt = span.get_text(strip=True)
            if txt.lower().startswith("sold"):
                sold_date = re.sub(r"^Sold\s+", "", txt, flags=re.IGNORECASE)
                break

        img_el = item_el.select_one("img")
        thumbnail = img_el.get("src") or img_el.get("data-src") if img_el else None

        results.append({
            "item_id": listing_id,
            "title": title,
            "listing_url": listing_url,
            "sold_price": sold_price,
            "sold_date": sold_date,
            "thumbnail_url": thumbnail,
        })
    return results


def _parse_s_item_results(items) -> list[dict]:
    """Parse eBay's legacy s-item layout."""
    results = []
    for item_el in items:
        title_el = (
            item_el.select_one("div.s-item__title span[role='heading']")
            or item_el.select_one("div.s-item__title span")
            or item_el.select_one("div.s-item__title")
        )
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        if title.lower().startswith("shop on ebay"):
            continue

        link_el = item_el.select_one("a.s-item__link")
        if not link_el or not link_el.get("href"):
            continue
        url = link_el["href"].split("?")[0]
        item_id_match = re.search(r"/itm/(\d+)", url)
        if not item_id_match:
            continue

        price_el = item_el.select_one("span.s-item__price")
        sold_price = _extract_price(price_el)

        sold_date = None
        for span in item_el.select("span.POSITIVE") + item_el.find_all("span"):
            txt = span.get_text(strip=True)
            if txt.lower().startswith("sold"):
                sold_date = re.sub(r"^Sold\s+", "", txt, flags=re.IGNORECASE)
                break

        img_el = item_el.select_one("img.s-item__image-img") or item_el.select_one("img")
        thumbnail = img_el.get("src") or img_el.get("data-src") if img_el else None

        results.append({
            "item_id": item_id_match.group(1),
            "title": title,
            "listing_url": url,
            "sold_price": sold_price,
            "sold_date": sold_date,
            "thumbnail_url": thumbnail,
        })
    return results


def _extract_price(el) -> float | None:
    if not el:
        return None
    price_match = re.search(
        r"[\$£]?([\d,]+\.?\d*)",
        el.get_text(strip=True).replace(",", ""),
    )
    if price_match:
        try:
            return float(price_match.group(1))
        except ValueError:
            pass
    return None


def _detect_condition(title: str) -> str | None:
    """Detect condition from listing title. Returns None if ambiguous."""
    # Skip graded cards that slipped through
    if GRADED_PATTERN.search(title):
        return None

    for condition, pattern in CONDITION_PATTERNS.items():
        if pattern.search(title):
            return condition
    return None


# ---------------------------------------------------------------------------
# Listing page image extraction
# ---------------------------------------------------------------------------

def _scrape_listing_images(session: requests.Session, listing_url: str) -> list[str]:
    """Fetch listing page and extract all high-res image URLs."""
    try:
        resp = session.get(listing_url, headers=_get_headers(), timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch listing %s: %s", listing_url, e)
        return []

    html = resp.text
    if _is_bot_blocked(html):
        logger.warning("Bot blocked on listing page: %s", listing_url)
        return []

    image_urls = []

    # Strategy 1: JSON-LD structured data
    soup = BeautifulSoup(html, "lxml")
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, dict) and "image" in data:
                imgs = data["image"]
                if isinstance(imgs, str):
                    image_urls.append(imgs)
                elif isinstance(imgs, list):
                    image_urls.extend(imgs)
        except (json.JSONDecodeError, TypeError):
            continue

    # Strategy 2: JS image array ("maxImageUrl")
    for match in re.finditer(r'"maxImageUrl"\s*:\s*"(https?://[^"]+)"', html):
        url = match.group(1)
        if url not in image_urls:
            image_urls.append(url)

    # Strategy 3: eBay image gallery URLs
    for match in re.finditer(
        r'"(https://i\.ebayimg\.com/images/g/[^"]+/s-l\d+\.(?:jpg|png|webp))"',
        html,
    ):
        url = match.group(1)
        if url not in image_urls:
            image_urls.append(url)

    # Strategy 4: og:image fallback
    if not image_urls:
        og_img = soup.find("meta", property="og:image")
        if og_img and og_img.get("content"):
            image_urls.append(og_img["content"])

    # Deduplicate, upgrade to max resolution
    seen = set()
    deduped = []
    for url in image_urls:
        hi_res = re.sub(r"/s-l\d+\.", "/s-l1600.", url)
        key = re.sub(r"/s-l\d+\.\w+$", "", hi_res)
        if key not in seen:
            seen.add(key)
            deduped.append(hi_res)

    return deduped


def _download_image(
    session: requests.Session,
    url: str,
    dest: Path,
) -> bool:
    """Download an image to dest. Skip if exists."""
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = session.get(url, headers=_get_headers(), timeout=30)
        resp.raise_for_status()
        if len(resp.content) < 1024:
            return False
        dest.write_bytes(resp.content)
        return True
    except Exception as e:
        logger.warning("Image download failed %s: %s", url, e)
        return False


# ---------------------------------------------------------------------------
# Title parsing (card name / set extraction)
# ---------------------------------------------------------------------------

# Card number: 4/102, 044/185, SV049/SV122
CARD_NUMBER_PATTERN = re.compile(r"\b([A-Z]*\d+)\s*/\s*([A-Z]*\d+)\b", re.IGNORECASE)

NOISE_PATTERN = re.compile(
    r"\b(pokemon|card|tcg|trading|game|holo|rare|ultra|secret|full\s*art|"
    r"alt\s*art|illustration|promo|japanese|english|mint|near\s*mint|"
    r"pack\s*fresh|lightly\s*played|moderately\s*played|heavily\s*played|"
    r"damaged|nm|lp|mp|hp|dmg|lot|bundle|singles?|official|authentic|"
    r"new|sealed|opened|free\s*ship\w*|fast\s*ship\w*)\b",
    re.IGNORECASE,
)

TITLE_PREFIX = re.compile(
    r"^(?:pokemon\s+(?:card|tcg|trading\s+card\s+game)\s*[-:]?\s*)+",
    re.IGNORECASE,
)


def _parse_raw_title(title: str) -> dict:
    """Extract card name, set name, card number from a raw card listing title."""
    result = {"card_name": None, "set_name": None, "card_number": None}
    if not title:
        return result

    working = title.strip()

    # Card number
    num_match = CARD_NUMBER_PATTERN.search(working)
    if num_match:
        result["card_number"] = f"{num_match.group(1)}/{num_match.group(2)}"
        working = working[:num_match.start()] + working[num_match.end():]

    # Set name (reuse from ebay_title_parser if available)
    try:
        from cardprice.scrapers.ebay_title_parser import _SET_PATTERNS
        for set_name, pattern in _SET_PATTERNS:
            if pattern.search(working):
                result["set_name"] = set_name
                working = pattern.sub("", working)
                break
    except ImportError:
        pass

    # Clean to card name
    working = TITLE_PREFIX.sub("", working)
    working = NOISE_PATTERN.sub("", working)
    working = re.sub(r"[#\-\u2013\u2014|,]+", " ", working)
    working = re.sub(r"\s{2,}", " ", working).strip()
    working = re.sub(r"^[\s\-\u2013\u2014:,!.]+|[\s\-\u2013\u2014:,!.]+$", "", working)

    if working and len(working) >= 2:
        result["card_name"] = working

    return result


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

class RawCardScraper:
    """Scrapes eBay sold listings for raw Pokemon cards by condition."""

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = _get_session()
        self.progress = ScrapeProgress()
        self._load_progress()
        self._domain_idx = 0  # current eBay domain to use

    @property
    def _domain(self) -> str:
        return EBAY_DOMAINS[self._domain_idx % len(EBAY_DOMAINS)]

    def _switch_domain(self):
        self._domain_idx += 1
        logger.info("Switching to eBay domain: %s", self._domain)

    # -- Progress persistence --

    def _progress_path(self) -> Path:
        return self.output_dir / PROGRESS_FILE

    def _load_progress(self):
        p = self._progress_path()
        if p.exists():
            try:
                self.progress = ScrapeProgress.from_dict(
                    json.loads(p.read_text())
                )
                logger.info(
                    "Resumed: %d listings, %d images, completed %s",
                    self.progress.total_listings,
                    self.progress.total_images,
                    self.progress.completed_conditions,
                )
            except Exception as e:
                logger.warning("Failed to load progress: %s", e)

    def _save_progress(self):
        self.progress.last_updated = datetime.now(timezone.utc).isoformat()
        self._progress_path().write_text(
            json.dumps(self.progress.to_dict(), indent=2)
        )

    # -- File paths --

    def _condition_dir(self, condition: str) -> Path:
        return self.output_dir / condition.upper()

    def _image_dir(self, condition: str) -> Path:
        d = self._condition_dir(condition) / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _listings_path(self, condition: str) -> Path:
        d = self._condition_dir(condition)
        d.mkdir(parents=True, exist_ok=True)
        return d / "listings.jsonl"

    def _append_listing(self, listing: RawListing):
        path = self._listings_path(listing.condition)
        with open(path, "a") as f:
            f.write(json.dumps(asdict(listing)) + "\n")

    # -- Scraping --

    def _fetch_search_page(self, url: str) -> str | None:
        """Fetch a search results page, with domain fallback on bot block."""
        for attempt in range(len(EBAY_DOMAINS)):
            # Replace domain in URL for retries
            actual_url = url
            if attempt > 0:
                self._switch_domain()
                actual_url = re.sub(
                    r"https?://[^/]+/",
                    f"https://{self._domain}/",
                    url,
                )

            try:
                resp = self.session.get(
                    actual_url, headers=_get_headers(), timeout=30
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.warning("Search request failed: %s", e)
                _polite_delay(5.0, 10.0)
                continue

            if _is_bot_blocked(resp.text):
                logger.warning("Bot blocked on %s, trying next domain", self._domain)
                _polite_delay(5.0, 10.0)
                continue

            return resp.text

        logger.error("All eBay domains blocked. Try again later or use Playwright.")
        return None

    def _process_listing(
        self, stub: dict, condition: str
    ) -> RawListing | None:
        """Process a single listing: verify condition, download images."""
        item_id = stub["item_id"]
        title = stub["title"]

        # Skip if already seen
        if item_id in self.progress.seen_item_ids:
            return None

        # Verify condition from title
        detected = _detect_condition(title)
        if detected is None:
            # Title has no clear condition label — skip
            return None
        if detected != condition:
            # Condition mismatch — file under detected condition instead
            condition = detected

        # Parse card metadata from title
        meta = _parse_raw_title(title)

        listing = RawListing(
            item_id=item_id,
            title=title,
            condition=condition,
            card_name=meta["card_name"],
            set_name=meta["set_name"],
            card_number=meta["card_number"],
            sold_price=stub.get("sold_price"),
            sold_date=stub.get("sold_date"),
            listing_url=stub.get("listing_url", ""),
            scraped_at=datetime.now(timezone.utc).isoformat(),
        )

        # Fetch full-size images from listing page
        if listing.listing_url:
            _polite_delay(LISTING_DELAY_MIN, LISTING_DELAY_MAX)
            image_urls = _scrape_listing_images(self.session, listing.listing_url)
            listing.image_urls = image_urls
            listing.num_images = len(image_urls)

            # Download images
            img_dir = self._image_dir(condition)
            for idx, img_url in enumerate(image_urls):
                ext = "jpg"
                if ".png" in img_url:
                    ext = "png"
                elif ".webp" in img_url:
                    ext = "webp"
                dest = img_dir / f"{item_id}_{idx:02d}.{ext}"
                if _download_image(self.session, img_url, dest):
                    listing.local_images.append(str(dest.relative_to(self.output_dir)))
                    self.progress.total_images += 1
                _polite_delay(IMAGE_DELAY, IMAGE_DELAY + 0.2)

        self.progress.seen_item_ids.add(item_id)
        self.progress.total_listings += 1
        self._append_listing(listing)
        return listing

    def scrape_condition(
        self,
        condition: str,
        max_pages: int = 5,
    ) -> int:
        """Scrape sold listings for one condition tier.

        Args:
            condition: One of NM, LP, MP, HP, DMG.
            max_pages: Max search result pages per query variant.

        Returns:
            Number of new listings scraped.
        """
        condition = condition.upper()
        if condition in self.progress.completed_conditions:
            logger.info("Condition %s already completed, skipping", condition)
            return 0

        queries = CONDITION_QUERIES.get(condition, [f'"{condition}"'])
        new_count = 0

        if not self.progress.started_at:
            self.progress.started_at = datetime.now(timezone.utc).isoformat()

        self.progress.current_condition = condition
        logger.info("=== Scraping condition: %s (%d query variants) ===", condition, len(queries))

        for qi, query in enumerate(queries):
            if qi < self.progress.current_query_idx and condition == self.progress.current_condition:
                continue  # resume past completed queries

            self.progress.current_query_idx = qi
            logger.info("Query %d/%d: %s", qi + 1, len(queries), query)

            start_page = 1
            if qi == self.progress.current_query_idx and self.progress.current_page > 0:
                start_page = self.progress.current_page

            for page in range(start_page, max_pages + 1):
                self.progress.current_page = page
                self._save_progress()

                url = _build_search_url(query, page=page, domain=self._domain)
                logger.info("  Page %d/%d", page, max_pages)

                html = self._fetch_search_page(url)
                if not html:
                    self.progress.errors += 1
                    break

                stubs = _parse_search_results(html)
                if not stubs:
                    logger.info("  No results on page %d, moving on", page)
                    break

                page_new = 0
                for stub in stubs:
                    try:
                        listing = self._process_listing(stub, condition)
                        if listing:
                            page_new += 1
                            new_count += 1
                    except Exception as e:
                        logger.warning("Error processing listing: %s", e)
                        self.progress.errors += 1

                logger.info(
                    "  Page %d: %d new listings (%d total, %d images)",
                    page, page_new, self.progress.total_listings,
                    self.progress.total_images,
                )

                _polite_delay(MIN_DELAY, MAX_DELAY)

            # Reset page counter for next query
            self.progress.current_page = 0

        self.progress.current_query_idx = 0
        self.progress.completed_conditions.append(condition)
        self._save_progress()
        logger.info(
            "=== Condition %s complete: %d new listings ===",
            condition, new_count,
        )
        return new_count

    def scrape_all_conditions(self, max_pages: int = 5) -> dict:
        """Scrape all condition tiers."""
        results = {}
        for condition in CONDITION_QUERIES:
            count = self.scrape_condition(condition, max_pages=max_pages)
            results[condition] = count
        return results

    def stats(self) -> dict:
        """Return current scraping stats."""
        stats = {
            "total_listings": self.progress.total_listings,
            "total_images": self.progress.total_images,
            "completed_conditions": self.progress.completed_conditions,
            "errors": self.progress.errors,
            "conditions": {},
        }
        for cond in CONDITION_QUERIES:
            cond_dir = self._condition_dir(cond)
            listings_file = cond_dir / "listings.jsonl"
            img_dir = cond_dir / "images"

            n_listings = 0
            if listings_file.exists():
                n_listings = sum(1 for _ in open(listings_file))

            n_images = 0
            if img_dir.exists():
                n_images = sum(1 for _ in img_dir.iterdir())

            stats["conditions"][cond] = {
                "listings": n_listings,
                "images": n_images,
            }
        return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape eBay sold listings for raw Pokemon cards by condition"
    )
    parser.add_argument(
        "--condition", "-c",
        choices=["NM", "LP", "MP", "HP", "DMG"],
        help="Specific condition to scrape",
    )
    parser.add_argument(
        "--all-conditions", "-a",
        action="store_true",
        help="Scrape all condition tiers",
    )
    parser.add_argument(
        "--pages", "-p",
        type=int, default=5,
        help="Max search result pages per query (default: 5)",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from last progress checkpoint",
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Print stats and exit",
    )
    parser.add_argument(
        "--output", "-o",
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    scraper = RawCardScraper(output_dir=args.output)

    if args.stats:
        s = scraper.stats()
        print(f"\nTotal: {s['total_listings']} listings, {s['total_images']} images")
        print(f"Completed: {s['completed_conditions']}")
        print(f"Errors: {s['errors']}")
        print("\nPer condition:")
        for cond, info in s["conditions"].items():
            print(f"  {cond}: {info['listings']} listings, {info['images']} images")
        return

    if args.all_conditions:
        results = scraper.scrape_all_conditions(max_pages=args.pages)
        print("\nResults:", json.dumps(results, indent=2))
    elif args.condition:
        count = scraper.scrape_condition(args.condition, max_pages=args.pages)
        print(f"\nScraped {count} new {args.condition} listings")
    elif args.resume:
        # Resume: scrape remaining conditions
        results = scraper.scrape_all_conditions(max_pages=args.pages)
        print("\nResults:", json.dumps(results, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
