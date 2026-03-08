"""eBay graded Pokemon card scraper for corner classifier training data.

Scrapes completed/sold listings for PSA, BGS, and CGC graded Pokemon cards.
Downloads listing photos and organizes them by grade for ML training.

For BGS listings specifically, the slab label shows per-subcategory grades
(Centering, Corners, Edges, Surface) which are extracted from listing titles
and used to create labeled corner ROI training pairs.

Output directory structure:
    data/condition_training/
        corners/{grade}/img_{cert}_{corner}.png   -- corner ROIs labeled by grade
        listings/{authority}_{grade}.jsonl         -- listing metadata
        images/{authority}_{grade}/{item_id}_{nn}.jpg  -- raw listing photos

The corner ROIs are organized into the 5-class structure expected by
cardprice.ml.corner_classifier.CornerWearDataset:
    Gem/      -- BGS sub-corners 10, PSA 10
    Mint/     -- BGS sub-corners 9.5, PSA 9
    Light/    -- BGS sub-corners 9, PSA 8
    Moderate/ -- BGS sub-corners 8-8.5, PSA 6-7
    Heavy/    -- BGS sub-corners <=7.5, PSA <=5

Usage (programmatic):
    from scrapers.ebay_graded import GradedCardScraper
    scraper = GradedCardScraper(authority="BGS")
    scraper.scrape_grade("9.5", max_pages=3)

Usage (CLI):
    python scrapers/ebay_graded.py --authority BGS --grade 9.5 --pages 3
    python scrapers/ebay_graded.py --authority PSA --all-grades --pages 2
"""

import hashlib
import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

# Prefer curl_cffi for browser TLS fingerprint impersonation (avoids eBay bot detection)
try:
    from curl_cffi import requests as cffi_requests
    _HAS_CFFI = True
except ImportError:
    _HAS_CFFI = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Use .co.uk as fallback when .com triggers bot challenge
SEARCH_URL = "https://www.ebay.com/sch/i.html"
SEARCH_URL_FALLBACK = "https://www.ebay.co.uk/sch/i.html"
POKEMON_CATEGORY = "183454"

# Rate limiting -- max 1 request per 2 seconds
MIN_DELAY = 2.0
MAX_DELAY = 3.5
LISTING_DELAY_MIN = 2.0
LISTING_DELAY_MAX = 3.0
IMAGE_DELAY = 0.5

# Rotating user agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) "
    "Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

# Grade tiers for each authority
PSA_GRADES = ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]
BGS_GRADES = ["10", "9.5", "9", "8.5", "8", "7.5", "7", "6.5", "6", "5", "4", "3"]
CGC_GRADES = ["10", "9.5", "9", "8.5", "8", "7.5", "7", "6.5", "6", "5", "4", "3"]

# BGS sub-grade labels (order on the slab label)
BGS_SUBGRADE_LABELS = ["centering", "corners", "edges", "surface"]

# Mapping from numeric sub-grade to corner classifier wear class.
# BGS sub-corner grades map to the 5-class system used by CornerWearDataset.
BGS_CORNER_GRADE_MAP = {
    "10":  "Gem",
    "9.5": "Mint",
    "9":   "Light",
    "8.5": "Moderate",
    "8":   "Moderate",
    "7.5": "Heavy",
    "7":   "Heavy",
    "6.5": "Heavy",
    "6":   "Heavy",
}

# PSA overall grade to approximate corner wear class.
# PSA doesn't provide sub-grades, so this is a rough proxy.
PSA_GRADE_MAP = {
    "10": "Gem",
    "9":  "Mint",
    "8":  "Light",
    "7":  "Moderate",
    "6":  "Moderate",
    "5":  "Heavy",
    "4":  "Heavy",
    "3":  "Heavy",
    "2":  "Heavy",
    "1":  "Heavy",
}

# Grade pattern: "PSA 10", "BGS 9.5", "CGC 9", etc.
GRADE_PATTERN = re.compile(
    r"\b(PSA|BGS|CGC|SGC)\s+([\d]+(?:\.[\d]+)?)\b", re.IGNORECASE
)

# Cert number patterns
CERT_IN_TITLE = re.compile(
    r"(?:cert\.?\s*#?\s*|certification\s*#?\s*)(\d{7,10})", re.IGNORECASE
)
PSA_CERT_PATTERN = re.compile(r"\b(\d{7,8})\b")
BGS_CERT_PATTERN = re.compile(r"\b(\d{8,10})\b")

# Default output
DEFAULT_OUTPUT_DIR = "data/condition_training"
PROGRESS_FILE = "scrape_progress.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GradedListing:
    """A single graded card eBay sold listing."""
    item_id: str
    title: str
    authority: str          # PSA, BGS, CGC
    grade: str              # "10", "9.5", etc.
    cert_number: str | None = None
    card_name: str | None = None
    sold_price: float | None = None
    sold_date: str | None = None
    listing_url: str = ""
    image_urls: list[str] = field(default_factory=list)
    local_images: list[str] = field(default_factory=list)
    # BGS sub-grades
    sub_centering: str | None = None
    sub_corners: str | None = None
    sub_edges: str | None = None
    sub_surface: str | None = None
    # Derived corner wear class (for training label)
    corner_wear_class: str | None = None
    scraped_at: str = ""


@dataclass
class ScrapeProgress:
    """Tracks scraping progress for resume capability."""
    authority: str = "PSA"
    current_grade: str = ""
    current_page: int = 0
    total_listings: int = 0
    total_images: int = 0
    total_corners: int = 0
    completed_grades: list[str] = field(default_factory=list)
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

class _CffiSessionWrapper:
    """Wraps curl_cffi.Session to look like requests.Session for the scraper."""

    def __init__(self):
        self._session = cffi_requests.Session(impersonate="chrome")

    def get(self, url, headers=None, timeout=30, **kwargs):
        return self._session.get(url, headers=headers, timeout=timeout, **kwargs)

    @property
    def cookies(self):
        return self._session.cookies


def _get_session():
    """Create an HTTP session.  Uses curl_cffi (Chrome TLS fingerprint) when
    available, otherwise falls back to plain requests with retry logic."""
    if _HAS_CFFI:
        logger.info("Using curl_cffi session (Chrome TLS impersonation)")
        return _CffiSessionWrapper()

    s = requests.Session()
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _get_headers() -> dict:
    """Request headers with random user agent."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.ebay.com/",
        "DNT": "1",
    }


def _polite_delay(min_s: float, max_s: float):
    """Sleep for a random interval between min_s and max_s."""
    time.sleep(random.uniform(min_s, max_s))


# ---------------------------------------------------------------------------
# Title parsing
# ---------------------------------------------------------------------------

def parse_graded_title(title: str, authority: str = "PSA") -> dict:
    """Parse a graded card listing title for grade, cert, and card info.

    Returns dict with: authority, grade, cert_number, card_name,
    sub_centering, sub_corners, sub_edges, sub_surface.
    """
    result = {
        "authority": authority,
        "grade": None,
        "cert_number": None,
        "card_name": None,
        "sub_centering": None,
        "sub_corners": None,
        "sub_edges": None,
        "sub_surface": None,
    }

    if not title:
        return result

    # Extract authority and grade
    grade_match = GRADE_PATTERN.search(title)
    if grade_match:
        result["authority"] = grade_match.group(1).upper()
        result["grade"] = grade_match.group(2)

    # Extract cert number
    cert_match = CERT_IN_TITLE.search(title)
    if cert_match:
        result["cert_number"] = cert_match.group(1)
    else:
        # Look for bare 7-8 digit numbers near the authority keyword
        auth_pos = title.upper().find(authority.upper())
        if auth_pos >= 0:
            context = title[max(0, auth_pos - 30):auth_pos + 40]
            for m in PSA_CERT_PATTERN.finditer(context):
                num = m.group(1)
                if len(num) >= 7:
                    result["cert_number"] = num
                    break

    # Extract BGS sub-grades
    if result["authority"] == "BGS":
        subs = _extract_bgs_subgrades(title)
        result["sub_centering"] = subs.get("centering")
        result["sub_corners"] = subs.get("corners")
        result["sub_edges"] = subs.get("edges")
        result["sub_surface"] = subs.get("surface")

    # Extract card name (strip grading noise)
    working = title
    # Remove grade mention
    working = GRADE_PATTERN.sub("", working)
    # Remove cert numbers
    working = CERT_IN_TITLE.sub("", working)
    # Remove BGS sub-grade text
    for label in BGS_SUBGRADE_LABELS:
        working = re.sub(
            rf"\b{label}\s*[:=]?\s*[\d]+(?:\.[\d]+)?\b", "",
            working, flags=re.IGNORECASE,
        )
    # Remove slash-separated sub-grades
    working = re.sub(
        r"[\|(]\s*[\d.]+\s*/\s*[\d.]+\s*/\s*[\d.]+\s*/\s*[\d.]+\s*[)\s]?",
        " ", working,
    )
    # Remove common noise words
    noise = re.compile(
        r"\b(pokemon|card|tcg|trading|game|holo|rare|ultra|secret|full\s*art|"
        r"alt\s*art|promo|japanese|english|mint|gem|pristine|graded|slab|"
        r"slabbed|lot|bundle|freshly|pop\s*\d+|low\s*pop|invest|investment|"
        r"PSA|BGS|CGC|SGC)\b",
        re.IGNORECASE,
    )
    working = noise.sub("", working)
    working = re.sub(r"[#\-\u2013\u2014|,]+", " ", working)
    working = re.sub(r"\s{2,}", " ", working).strip()
    working = re.sub(r"^[\s\-:,!.]+|[\s\-:,!.]+$", "", working)

    if working and len(working) >= 2:
        result["card_name"] = working

    return result


def _extract_bgs_subgrades(title: str) -> dict:
    """Extract BGS sub-grades from listing title.

    BGS slabs show 4 sub-grades: Centering, Corners, Edges, Surface.
    Sellers often include these in titles:
        "BGS 9.5 Centering 9.5 Corners 9.5 Edges 9.5 Surface 10"
        "BGS 9.5 | 9.5/9.5/9/10"
        "BGS 9.5 (9.5, 9.5, 9, 10)"
    """
    result = {"centering": None, "corners": None, "edges": None, "surface": None}

    # Pattern 1: Named sub-grades
    for label in BGS_SUBGRADE_LABELS:
        pat = re.compile(
            rf"\b{label}\s*[:=]?\s*([\d]+(?:\.[\d]+)?)\b", re.IGNORECASE
        )
        match = pat.search(title)
        if match:
            result[label] = match.group(1)

    if any(v is not None for v in result.values()):
        return result

    # Pattern 2: Slash-separated "BGS 9.5 | 9.5/9.5/9/10"
    slash_pat = re.compile(
        r"\bBGS\s+[\d.]+\s*[\|(]?\s*"
        r"([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)",
        re.IGNORECASE,
    )
    match = slash_pat.search(title)
    if match:
        # Standard BGS order: Centering / Corners / Edges / Surface
        result["centering"] = match.group(1)
        result["corners"] = match.group(2)
        result["edges"] = match.group(3)
        result["surface"] = match.group(4)
        return result

    # Pattern 3: Comma-separated in parens "BGS 9.5 (9.5, 9.5, 9, 10)"
    comma_pat = re.compile(
        r"\bBGS\s+[\d.]+\s*\(\s*"
        r"([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)",
        re.IGNORECASE,
    )
    match = comma_pat.search(title)
    if match:
        result["centering"] = match.group(1)
        result["corners"] = match.group(2)
        result["edges"] = match.group(3)
        result["surface"] = match.group(4)

    return result


def subgrade_to_wear_class(subgrade: str | None) -> str | None:
    """Map a BGS numeric sub-grade to a corner wear class name."""
    if subgrade is None:
        return None
    return BGS_CORNER_GRADE_MAP.get(subgrade)


def overall_grade_to_wear_class(grade: str, authority: str = "PSA") -> str | None:
    """Map an overall PSA/CGC grade to an approximate corner wear class."""
    if authority.upper() == "BGS":
        return BGS_CORNER_GRADE_MAP.get(grade)
    return PSA_GRADE_MAP.get(grade)


# ---------------------------------------------------------------------------
# Search result scraping
# ---------------------------------------------------------------------------

def _build_search_url(
    authority: str, grade: str, page: int = 1, *, use_fallback: bool = False,
) -> str:
    """Build eBay sold-listings search URL for graded Pokemon cards."""
    query = f"{authority} {grade} pokemon card"
    params = {
        "_nkw": query,
        "_sacat": POKEMON_CATEGORY,
        "LH_Complete": "1",
        "LH_Sold": "1",
        "_sop": "13",   # sort: most recent
        "_ipg": "60",   # 60 items per page
    }
    if page > 1:
        params["_pgn"] = str(page)
    base = SEARCH_URL_FALLBACK if use_fallback else SEARCH_URL
    return f"{base}?{urlencode(params)}"


def _parse_search_results(html: str) -> list[dict]:
    """Parse eBay search results page into listing stubs.

    Returns list of dicts with: item_id, title, listing_url, sold_price,
    sold_date, thumbnail_url.

    Supports both the legacy li.s-item layout and the 2025+ li.s-card layout.
    """
    soup = BeautifulSoup(html, "lxml")

    # Try new layout first (2025+), then legacy
    items = soup.select("li.s-card[id^=item]")
    if items:
        return _parse_search_results_v2(items)

    items = soup.select("li.s-item")
    return _parse_search_results_v1(items)


def _parse_search_results_v2(items) -> list[dict]:
    """Parse the 2025+ eBay s-card search result layout."""
    results = []

    for item_el in items:
        # Item ID from data-listingid attribute or from href
        item_id = item_el.get("data-listingid")
        if not item_id:
            link = item_el.find("a", href=re.compile(r"/itm/\d+"))
            if link:
                m = re.search(r"/itm/(\d+)", link["href"])
                if m:
                    item_id = m.group(1)
        if not item_id:
            continue

        # Title from the card title element or image alt text
        title_el = item_el.select_one(".s-card__title")
        if not title_el:
            img_el = item_el.select_one("img.s-card__image")
            if img_el and img_el.get("alt"):
                title = img_el["alt"]
            else:
                continue
        else:
            # The .s-card__title may contain "Sold  Mar 6, 2026" as child spans
            # Get text from the heading link instead
            heading_link = item_el.select_one(
                "div[class*=header] a"
            )
            if heading_link:
                # Get only the first text, not the "Sold" span
                title = heading_link.get_text(strip=True)
                # Strip trailing "Sold  Date" if concatenated
                title = re.sub(r"Sold\s+\w{3}\s+\d{1,2},\s*\d{4}$", "", title).strip()
            else:
                title = title_el.get_text(strip=True)

        if not title or title.lower().startswith("shop on ebay"):
            continue

        # URL
        link_el = item_el.find("a", href=re.compile(r"/itm/\d+"))
        listing_url = ""
        if link_el:
            listing_url = link_el["href"].split("?")[0]

        # Price
        sold_price = None
        price_el = item_el.select_one("span[class*=price]")
        if price_el:
            price_text = price_el.get_text(strip=True).replace(",", "")
            price_match = re.search(r"\$?([\d]+\.?\d*)", price_text)
            if price_match:
                try:
                    sold_price = float(price_match.group(1))
                except ValueError:
                    pass

        # Sold date
        sold_date = None
        for span in item_el.find_all("span"):
            txt = span.get_text(strip=True)
            if txt.lower().startswith("sold"):
                cleaned = re.sub(r"^Sold\s+", "", txt, flags=re.IGNORECASE)
                sold_date = cleaned
                break

        # Thumbnail
        img_el = item_el.select_one("img.s-card__image")
        if not img_el:
            img_el = item_el.select_one("img")
        thumbnail = None
        if img_el:
            thumbnail = (
                img_el.get("src")
                or img_el.get("data-src")
                or img_el.get("data-defer-load")
            )

        results.append({
            "item_id": item_id,
            "title": title,
            "listing_url": listing_url,
            "sold_price": sold_price,
            "sold_date": sold_date,
            "thumbnail_url": thumbnail,
        })

    return results


def _parse_search_results_v1(items) -> list[dict]:
    """Parse the legacy eBay s-item search result layout."""
    results = []

    for item_el in items:
        # Title
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

        # URL and item ID
        link_el = item_el.select_one("a.s-item__link")
        if not link_el or not link_el.get("href"):
            continue
        url = link_el["href"].split("?")[0]
        item_id_match = re.search(r"/itm/(\d+)", url)
        if not item_id_match:
            continue
        item_id = item_id_match.group(1)

        # Price
        price_el = (
            item_el.select_one("span.s-item__price")
            or item_el.select_one("span.POSITIVE")
        )
        sold_price = None
        if price_el:
            price_match = re.search(
                r"\$?([\d,]+\.?\d*)",
                price_el.get_text(strip=True).replace(",", ""),
            )
            if price_match:
                try:
                    sold_price = float(price_match.group(1))
                except ValueError:
                    pass

        # Sold date
        sold_date = None
        for span in item_el.select("span.POSITIVE") + item_el.find_all("span"):
            txt = span.get_text(strip=True)
            if txt.lower().startswith("sold"):
                cleaned = re.sub(r"^Sold\s+", "", txt, flags=re.IGNORECASE)
                sold_date = cleaned
                break

        # Thumbnail
        img_el = (
            item_el.select_one("img.s-item__image-img")
            or item_el.select_one("img")
        )
        thumbnail = None
        if img_el:
            thumbnail = img_el.get("src") or img_el.get("data-src")

        results.append({
            "item_id": item_id,
            "title": title,
            "listing_url": url,
            "sold_price": sold_price,
            "sold_date": sold_date,
            "thumbnail_url": thumbnail,
        })

    return results


# ---------------------------------------------------------------------------
# Listing page image extraction
# ---------------------------------------------------------------------------

def _scrape_listing_images(session: requests.Session, listing_url: str) -> list[str]:
    """Fetch an individual eBay listing page and extract high-res image URLs.

    Uses multiple extraction strategies (JSON-LD, inline JS, gallery regex,
    og:image) and deduplicates by URL fingerprint.

    Returns list of image URLs (typically 2-6 per listing).
    """
    try:
        resp = session.get(listing_url, headers=_get_headers(), timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch listing %s: %s", listing_url, e)
        return []

    html = resp.text
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

    # Strategy 2: maxImageUrl in inline JS
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

    # Deduplicate and upgrade to max resolution
    seen = set()
    deduped = []
    for url in image_urls:
        hi_res = _upgrade_image_url(url)
        key = _url_fingerprint(hi_res)
        if key not in seen:
            seen.add(key)
            deduped.append(hi_res)

    return deduped


def _upgrade_image_url(url: str) -> str:
    """Upgrade eBay image URL to highest resolution (s-l1600)."""
    return re.sub(r"/s-l\d+\.", "/s-l1600.", url)


def _url_fingerprint(url: str) -> str:
    """Stable fingerprint for deduplication (strips size suffix)."""
    return re.sub(r"/s-l\d+\.\w+$", "", url)


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

def _download_image(session: requests.Session, url: str, dest: Path) -> bool:
    """Download a single image to disk. Returns True on success."""
    if dest.exists():
        return True

    try:
        resp = session.get(url, headers=_get_headers(), timeout=20)
        if resp.status_code != 200:
            logger.debug("HTTP %d downloading %s", resp.status_code, url)
            return False
        if len(resp.content) < 1024:
            logger.debug("Suspiciously small image (%d bytes): %s", len(resp.content), url)
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return True
    except requests.RequestException as e:
        logger.debug("Download error for %s: %s", url, e)
        return False


# ---------------------------------------------------------------------------
# Corner ROI extraction (for training data)
# ---------------------------------------------------------------------------

def extract_and_save_corner_rois(
    image_path: Path,
    output_dir: Path,
    wear_class: str,
    cert_or_id: str,
) -> int:
    """Extract 4 corner ROIs from a card image and save to the training dir.

    Saves each corner as:
        output_dir/corners/{wear_class}/img_{cert_or_id}_{corner}.png

    This matches the CornerWearDataset expected layout:
        data_dir/{Gem,Mint,Light,Moderate,Heavy}/*.png

    Parameters
    ----------
    image_path : Path to downloaded listing image.
    output_dir : Base output dir (e.g. data/condition_training).
    wear_class : One of "Gem", "Mint", "Light", "Moderate", "Heavy".
    cert_or_id : Cert number or item ID for filename uniqueness.

    Returns number of corners saved (0-4).
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available, skipping corner extraction")
        return 0

    img = cv2.imread(str(image_path))
    if img is None:
        logger.debug("Cannot read image for corner extraction: %s", image_path)
        return 0

    h, w = img.shape[:2]

    # Skip images that are too small or not card-shaped
    # Graded slab photos are typically portrait orientation
    if h < 200 or w < 150:
        logger.debug("Image too small for corner extraction: %dx%d", w, h)
        return 0

    # Corner ROI size: ~16% of width, ~12% of height (matches ~10mm on a card)
    roi_w = max(int(w * 0.16), 32)
    roi_h = max(int(h * 0.12), 32)

    corner_names = ["tl", "tr", "bl", "br"]
    corners = {
        "tl": img[0:roi_h, 0:roi_w],
        "tr": img[0:roi_h, w - roi_w:w],
        "bl": img[h - roi_h:h, 0:roi_w],
        "br": img[h - roi_h:h, w - roi_w:w],
    }

    corners_dir = output_dir / "corners" / wear_class
    corners_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for name, roi in corners.items():
        dest = corners_dir / f"img_{cert_or_id}_{name}.png"
        if not dest.exists():
            cv2.imwrite(str(dest), roi)
            saved += 1

    return saved


# ---------------------------------------------------------------------------
# Main scraper class
# ---------------------------------------------------------------------------

class GradedCardScraper:
    """Scrapes eBay sold listings for graded Pokemon cards.

    Collects listing photos organized by grade for corner classifier
    training. Supports PSA, BGS, and CGC with resume capability.
    """

    def __init__(
        self,
        authority: Literal["PSA", "BGS", "CGC"] = "PSA",
        output_dir: str = DEFAULT_OUTPUT_DIR,
        extract_corners: bool = True,
        fetch_listing_images: bool = True,
    ):
        """
        Args:
            authority: Grading authority to search for.
            output_dir: Base directory for downloaded data.
            extract_corners: If True, extract corner ROIs from downloaded
                images and organize by wear class for training.
            fetch_listing_images: If True, visit each listing page to get
                full-resolution photos. If False, use thumbnails only.
        """
        self.authority = authority.upper()
        self.output_dir = Path(output_dir)
        self.extract_corners = extract_corners
        self.fetch_listing_images = fetch_listing_images
        self.session = _get_session()
        self.progress = ScrapeProgress(
            authority=self.authority,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._load_progress()

    # -- Progress persistence --

    def _progress_path(self) -> Path:
        return self.output_dir / f"{self.authority.lower()}_{PROGRESS_FILE}"

    def _load_progress(self):
        path = self._progress_path()
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self.progress = ScrapeProgress.from_dict(data)
                logger.info(
                    "Resumed %s scrape: %d listings, %d images, %d corners, "
                    "grade=%s page=%d",
                    self.authority,
                    self.progress.total_listings,
                    self.progress.total_images,
                    self.progress.total_corners,
                    self.progress.current_grade,
                    self.progress.current_page,
                )
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning("Could not load progress: %s", e)

    def _save_progress(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress.last_updated = datetime.now(timezone.utc).isoformat()
        path = self._progress_path()
        with open(path, "w") as f:
            json.dump(self.progress.to_dict(), f, indent=2)

    # -- Listing metadata --

    def _listings_path(self, grade: str) -> Path:
        safe_grade = grade.replace(".", "_")
        return (
            self.output_dir
            / "listings"
            / f"{self.authority.lower()}_{safe_grade}.jsonl"
        )

    def _append_listing(self, listing: GradedListing):
        path = self._listings_path(listing.grade)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(asdict(listing), default=str) + "\n")

    # -- Image storage --

    def _image_dir(self, grade: str) -> Path:
        safe_grade = grade.replace(".", "_")
        return (
            self.output_dir
            / "images"
            / f"{self.authority.lower()}_{safe_grade}"
        )

    def _image_path(self, grade: str, item_id: str, idx: int, ext: str = "jpg") -> Path:
        return self._image_dir(grade) / f"{item_id}_{idx:02d}.{ext}"

    # -- Core scraping --

    def scrape_grade(
        self,
        grade: str,
        max_pages: int = 5,
        max_listings: int = 0,
    ) -> int:
        """Scrape sold listings for a single grade level.

        Args:
            grade: Grade value (e.g. "10", "9.5").
            max_pages: Maximum search result pages (60 items each).
            max_listings: Stop after this many listings (0 = no limit).

        Returns:
            Number of new listings scraped.
        """
        if grade in self.progress.completed_grades:
            logger.info("Grade %s %s already completed, skipping.", self.authority, grade)
            return 0

        # Resume support
        start_page = 1
        if self.progress.current_grade == grade and self.progress.current_page > 0:
            start_page = self.progress.current_page
            logger.info("Resuming grade %s from page %d", grade, start_page)

        self.progress.current_grade = grade
        new_listings = 0

        for page in range(start_page, max_pages + 1):
            self.progress.current_page = page
            self._save_progress()

            # Try primary URL, then fallback domain if blocked
            resp = None
            for use_fallback in (False, True):
                url = _build_search_url(
                    self.authority, grade, page, use_fallback=use_fallback,
                )
                logger.info(
                    "[%s %s] Page %d/%d: %s",
                    self.authority, grade, page, max_pages, url,
                )

                for attempt in range(2):
                    try:
                        resp = self.session.get(
                            url, headers=_get_headers(), timeout=30,
                        )
                        resp.raise_for_status()
                    except (requests.RequestException, Exception) as e:
                        logger.error(
                            "Failed to fetch page %d (attempt %d): %s",
                            page, attempt + 1, e,
                        )
                        self.progress.errors += 1
                        _polite_delay(10.0, 20.0)
                        continue

                    # Check for eBay bot challenge page
                    if (
                        "Pardon Our Interruption" in resp.text
                        or len(resp.text) < 20000
                    ):
                        wait = 15 * (attempt + 1)
                        logger.warning(
                            "eBay bot challenge on page %d (attempt %d), "
                            "waiting %ds...", page, attempt + 1, wait,
                        )
                        _polite_delay(wait, wait + 5)
                        self.session = _get_session()
                        resp = None
                        continue
                    break

                if resp and "Pardon Our Interruption" not in resp.text:
                    break
                if not use_fallback:
                    logger.info(
                        "Primary domain blocked, trying fallback domain..."
                    )
                    self.session = _get_session()
                    _polite_delay(3.0, 5.0)

            if resp is None or "Pardon Our Interruption" in resp.text:
                logger.error(
                    "Could not bypass eBay bot challenge, "
                    "stopping grade %s.", grade,
                )
                break

            stubs = _parse_search_results(resp.text)
            if not stubs:
                logger.info("No results on page %d, stopping.", page)
                break

            logger.info("Found %d listings on page %d", len(stubs), page)

            for stub in stubs:
                item_id = stub["item_id"]
                if item_id in self.progress.seen_item_ids:
                    continue

                listing = self._process_listing(stub, grade)
                if listing:
                    new_listings += 1
                    self.progress.total_listings += 1
                    self.progress.seen_item_ids.add(item_id)
                    self._append_listing(listing)

                    if new_listings % 10 == 0:
                        logger.info(
                            "[%s %s] %d listings, %d images, %d corners",
                            self.authority, grade,
                            new_listings,
                            self.progress.total_images,
                            self.progress.total_corners,
                        )

                    if max_listings > 0 and new_listings >= max_listings:
                        logger.info(
                            "Reached max_listings=%d, stopping.", max_listings
                        )
                        break

            if max_listings > 0 and new_listings >= max_listings:
                break

            # Check for next page
            soup = BeautifulSoup(resp.text, "lxml")
            if not soup.select_one("a.pagination__next") and page < max_pages:
                logger.info("No next page after page %d.", page)
                break

            _polite_delay(MIN_DELAY, MAX_DELAY)

        self.progress.completed_grades.append(grade)
        self.progress.current_page = 0
        self._save_progress()

        logger.info(
            "[%s %s] Done: %d new listings, %d images, %d corners.",
            self.authority, grade,
            new_listings,
            self.progress.total_images,
            self.progress.total_corners,
        )
        return new_listings

    def _process_listing(self, stub: dict, grade: str) -> GradedListing | None:
        """Process a single listing: parse title, fetch images, extract corners."""
        title = stub["title"]
        item_id = stub["item_id"]
        listing_url = stub["listing_url"]

        parsed = parse_graded_title(title, self.authority)

        listing = GradedListing(
            item_id=item_id,
            title=title,
            authority=parsed.get("authority", self.authority),
            grade=parsed.get("grade") or grade,
            cert_number=parsed.get("cert_number"),
            card_name=parsed.get("card_name"),
            sold_price=stub.get("sold_price"),
            sold_date=stub.get("sold_date"),
            listing_url=listing_url,
            sub_centering=parsed.get("sub_centering"),
            sub_corners=parsed.get("sub_corners"),
            sub_edges=parsed.get("sub_edges"),
            sub_surface=parsed.get("sub_surface"),
            scraped_at=datetime.now(timezone.utc).isoformat(),
        )

        # Determine corner wear class
        if self.authority == "BGS" and listing.sub_corners:
            listing.corner_wear_class = subgrade_to_wear_class(listing.sub_corners)
        else:
            listing.corner_wear_class = overall_grade_to_wear_class(
                listing.grade, self.authority
            )

        # Fetch full listing page for high-res images
        image_urls = []
        if self.fetch_listing_images and listing_url:
            _polite_delay(LISTING_DELAY_MIN, LISTING_DELAY_MAX)
            image_urls = _scrape_listing_images(self.session, listing_url)
        elif stub.get("thumbnail_url"):
            image_urls = [_upgrade_image_url(stub["thumbnail_url"])]

        listing.image_urls = image_urls

        # Download images
        downloaded_paths = []
        for idx, img_url in enumerate(image_urls):
            ext = "jpg"
            if ".png" in img_url.lower():
                ext = "png"
            elif ".webp" in img_url.lower():
                ext = "webp"

            dest = self._image_path(listing.grade, item_id, idx, ext)
            _polite_delay(IMAGE_DELAY, IMAGE_DELAY + 0.3)

            if _download_image(self.session, img_url, dest):
                downloaded_paths.append(dest)
                listing.local_images.append(
                    str(dest.relative_to(self.output_dir))
                )
                self.progress.total_images += 1

        # Extract corner ROIs for training
        if self.extract_corners and listing.corner_wear_class:
            cert_or_id = listing.cert_number or item_id
            for img_path in downloaded_paths:
                n = extract_and_save_corner_rois(
                    img_path,
                    self.output_dir,
                    listing.corner_wear_class,
                    f"{cert_or_id}_{img_path.stem}",
                )
                self.progress.total_corners += n

        return listing

    def scrape_all_grades(self, max_pages_per_grade: int = 5) -> dict:
        """Scrape all grades for the configured authority."""
        if self.authority == "BGS":
            grades = BGS_GRADES
        elif self.authority == "CGC":
            grades = CGC_GRADES
        else:
            grades = PSA_GRADES

        summary = {}
        for grade in grades:
            count = self.scrape_grade(grade, max_pages=max_pages_per_grade)
            summary[grade] = count

        total = sum(summary.values())
        logger.info(
            "%s scrape complete: %d listings, %d images, %d corners",
            self.authority, total,
            self.progress.total_images,
            self.progress.total_corners,
        )
        return summary

    def get_stats(self) -> dict:
        """Return current scraping statistics."""
        # Count corner ROIs per wear class
        corners_dir = self.output_dir / "corners"
        class_counts = {}
        if corners_dir.exists():
            for cls_dir in corners_dir.iterdir():
                if cls_dir.is_dir():
                    count = sum(
                        1 for f in cls_dir.iterdir()
                        if f.suffix in (".png", ".jpg")
                    )
                    class_counts[cls_dir.name] = count

        return {
            "authority": self.authority,
            "total_listings": self.progress.total_listings,
            "total_images": self.progress.total_images,
            "total_corners": self.progress.total_corners,
            "corner_class_counts": class_counts,
            "completed_grades": self.progress.completed_grades,
            "current_grade": self.progress.current_grade,
            "current_page": self.progress.current_page,
            "unique_items": len(self.progress.seen_item_ids),
            "errors": self.progress.errors,
            "started_at": self.progress.started_at,
            "last_updated": self.progress.last_updated,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Scrape eBay graded Pokemon card sold listings for "
            "corner classifier training data."
        ),
    )
    parser.add_argument(
        "--authority", "-a",
        choices=["PSA", "BGS", "CGC"],
        default="PSA",
        help="Grading authority (default: PSA)",
    )
    parser.add_argument(
        "--grade", "-g",
        type=str, default=None,
        help="Specific grade to scrape (e.g. '10', '9.5'). Omit for --all-grades.",
    )
    parser.add_argument(
        "--all-grades",
        action="store_true",
        help="Scrape all grades for the authority.",
    )
    parser.add_argument(
        "--pages", "-p",
        type=int, default=5,
        help="Max search result pages per grade (default: 5, 60 items each).",
    )
    parser.add_argument(
        "--count", "-c",
        type=int, default=0,
        help="Max listings per grade (0 = no limit, default: 0).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-corners",
        action="store_true",
        help="Skip corner ROI extraction (just download images).",
    )
    parser.add_argument(
        "--no-listing-images",
        action="store_true",
        help="Skip listing page visits (thumbnails only, much faster).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print scraping statistics and exit.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    scraper = GradedCardScraper(
        authority=args.authority,
        output_dir=args.output,
        extract_corners=not args.no_corners,
        fetch_listing_images=not args.no_listing_images,
    )

    if args.stats:
        stats = scraper.get_stats()
        print(json.dumps(stats, indent=2))
        return

    if args.grade:
        count = scraper.scrape_grade(
            args.grade, max_pages=args.pages, max_listings=args.count
        )
        print(f"\nScraped {count} new {args.authority} {args.grade} listings.")
    elif args.all_grades:
        summary = scraper.scrape_all_grades(max_pages_per_grade=args.pages)
        print(f"\nScrape summary for {args.authority}:")
        for grade, count in summary.items():
            print(f"  Grade {grade}: {count} new listings")
        total = sum(summary.values())
        print(
            f"  Total: {total} listings, "
            f"{scraper.progress.total_images} images, "
            f"{scraper.progress.total_corners} corner ROIs"
        )
    else:
        parser.print_help()
        print("\nSpecify --grade or --all-grades to start scraping.")
        return

    stats = scraper.get_stats()
    print(f"\nFinal stats: {stats['total_listings']} listings, "
          f"{stats['total_images']} images, "
          f"{stats['total_corners']} corner ROIs, "
          f"{stats['errors']} errors")
    if stats["corner_class_counts"]:
        print("Corner ROIs by class:")
        for cls, cnt in sorted(stats["corner_class_counts"].items()):
            print(f"  {cls}: {cnt}")


if __name__ == "__main__":
    main()
