"""eBay graded-card scraper for condition training data collection.

Scrapes completed/sold listings for PSA and BGS graded Pokemon cards,
downloading listing photos organized by grade for ML training.

Target: ~3,000 listings / ~12,000 photos per 5-hour session.

Usage (CLI):
    python -m cardprice.scrapers.ebay_condition --grade 10 --pages 5
    python -m cardprice.scrapers.ebay_condition --authority BGS --grade 9.5
    python -m cardprice.scrapers.ebay_condition --resume
    python -m cardprice.scrapers.ebay_condition --all-grades --pages 3

Usage (programmatic):
    from cardprice.scrapers.ebay_condition import ConditionScraper
    scraper = ConditionScraper(authority="PSA")
    scraper.scrape_grade(grade="10", max_pages=5)
"""

import hashlib
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEARCH_URL = "https://www.ebay.com/sch/i.html"
POKEMON_CATEGORY = "183454"

# Rate limiting
MIN_DELAY = 2.0  # seconds between requests
MAX_DELAY = 3.5
LISTING_DELAY_MIN = 1.5  # for individual listing pages
LISTING_DELAY_MAX = 2.5
IMAGE_DELAY = 0.3  # image CDN is more tolerant

# Rotating user agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

# PSA grades to scrape (integer + half grades)
PSA_GRADES = ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]

# BGS grades (including half grades)
BGS_GRADES = ["10", "9.5", "9", "8.5", "8", "7.5", "7", "6.5", "6", "5", "4", "3"]

# BGS sub-grade labels found on slab photos
BGS_SUBGRADE_LABELS = ["centering", "corners", "edges", "surface"]

# PSA cert number pattern (7-8 digit number)
PSA_CERT_PATTERN = re.compile(r"\b(\d{7,8})\b")

# BGS cert number pattern (typically 9-10 digits or formatted)
BGS_CERT_PATTERN = re.compile(r"\b(\d{8,10})\b")

# Output directory
DEFAULT_OUTPUT_DIR = "data/condition_training"

# Progress file
PROGRESS_FILE = "scrape_progress.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GradedListing:
    """A single graded card listing from eBay."""
    item_id: str
    title: str
    authority: str  # PSA, BGS, CGC
    grade: str  # "10", "9.5", etc.
    cert_number: str | None = None
    card_name: str | None = None
    set_name: str | None = None
    card_number: str | None = None
    sold_price: float | None = None
    sold_date: str | None = None
    listing_url: str = ""
    image_urls: list[str] = field(default_factory=list)
    local_images: list[str] = field(default_factory=list)
    # BGS sub-grades (only populated for BGS listings with slab photos)
    sub_centering: str | None = None
    sub_corners: str | None = None
    sub_edges: str | None = None
    sub_surface: str | None = None
    scraped_at: str = ""


@dataclass
class ScrapeProgress:
    """Tracks scraping progress for resume capability."""
    authority: str = "PSA"
    current_grade: str = ""
    current_page: int = 0
    total_listings: int = 0
    total_images: int = 0
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
# Title parsing for graded cards
# ---------------------------------------------------------------------------

# Grade pattern: "PSA 10", "BGS 9.5", etc.
GRADE_PATTERN = re.compile(
    r"\b(PSA|BGS|CGC|SGC)\s+([\d]+(?:\.[\d]+)?)\b", re.IGNORECASE
)

# Card number pattern: 4/102, 044/185, SV049/SV122
CARD_NUMBER_PATTERN = re.compile(
    r"\b([A-Z]*\d+)\s*/\s*([A-Z]*\d+)\b", re.IGNORECASE
)

# PSA cert in title: "Cert #12345678" or just a 7-8 digit number near "PSA"
CERT_IN_TITLE = re.compile(
    r"(?:cert\.?\s*#?\s*|certification\s*#?\s*)(\d{7,10})", re.IGNORECASE
)

# Noise words for card name extraction
NOISE_PATTERN = re.compile(
    r"\b(pokemon|card|tcg|trading|game|holo|rare|ultra|secret|full\s*art|"
    r"alt\s*art|illustration|promo|japanese|english|mint|gem|pristine|"
    r"graded|slab|slabbed|lot|bundle|freshly|pop\s*\d+|low\s*pop|"
    r"invest|investment|comp|comparison)\b",
    re.IGNORECASE,
)


def parse_graded_title(title: str, authority: str = "PSA") -> dict:
    """Parse a graded card listing title.

    Returns dict with: card_name, set_name, card_number, grade,
    cert_number, authority.
    """
    result = {
        "card_name": None,
        "set_name": None,
        "card_number": None,
        "grade": None,
        "cert_number": None,
        "authority": authority,
    }

    if not title:
        return result

    working = title.strip()

    # Extract grade
    grade_match = GRADE_PATTERN.search(working)
    if grade_match:
        result["authority"] = grade_match.group(1).upper()
        result["grade"] = grade_match.group(2)
        working = working[:grade_match.start()] + working[grade_match.end():]

    # Strip BGS sub-grade text BEFORE card number extraction
    # (sub-grade slashes like "9.5/9.5/9/10" would collide with card numbers)
    # Pattern: named sub-grades "Centering 9.5 Corners 9.5 ..."
    for label in BGS_SUBGRADE_LABELS:
        working = re.sub(
            rf"\b{label}\s*[:=]?\s*[\d]+(?:\.[\d]+)?\b", "", working, flags=re.IGNORECASE
        )
    # Pattern: slash-separated "| 9.5/9.5/9/10" or "(9.5/9.5/9/10)"
    working = re.sub(
        r"[\|(]\s*[\d.]+\s*/\s*[\d.]+\s*/\s*[\d.]+\s*/\s*[\d.]+\s*[)\s]?",
        " ", working,
    )
    # Pattern: comma-separated "(9.5, 9.5, 9, 10)"
    working = re.sub(
        r"\(\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*\)",
        "", working,
    )

    # Extract cert number from title
    cert_match = CERT_IN_TITLE.search(working)
    if cert_match:
        result["cert_number"] = cert_match.group(1)
        working = working[:cert_match.start()] + working[cert_match.end():]

    # Extract card number (now safe from sub-grade slash patterns)
    num_match = CARD_NUMBER_PATTERN.search(working)
    if num_match:
        result["card_number"] = f"{num_match.group(1)}/{num_match.group(2)}"
        working = working[:num_match.start()] + working[num_match.end():]

    # Known set names (re-use from ebay_title_parser)
    try:
        from cardprice.scrapers.ebay_title_parser import _SET_PATTERNS
        for set_name, pattern in _SET_PATTERNS:
            if pattern.search(working):
                result["set_name"] = set_name
                working = pattern.sub("", working)
                break
    except ImportError:
        pass

    # Clean up card name
    working = re.sub(r"^(?:pokemon\s+(?:card|tcg)\s*[-:]?\s*)+", "", working, flags=re.IGNORECASE)
    working = NOISE_PATTERN.sub("", working)
    working = re.sub(r"\b(PSA|BGS|CGC|SGC)\b", "", working, flags=re.IGNORECASE)
    working = re.sub(r"[#\-\u2013\u2014|,]+", " ", working)
    working = re.sub(r"\s{2,}", " ", working).strip()
    working = re.sub(r"^[\s\-\u2013\u2014:,!.]+|[\s\-\u2013\u2014:,!.]+$", "", working)

    if working and len(working) >= 2:
        result["card_name"] = working

    return result


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_session() -> requests.Session:
    """Create a requests session with retry logic."""
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
# Search result scraping
# ---------------------------------------------------------------------------

def _build_search_url(authority: str, grade: str, page: int = 1) -> str:
    """Build eBay search URL for graded Pokemon card sold listings.

    Searches for e.g. 'PSA 10 pokemon card' with sold-listing filters.
    """
    query = f"{authority} {grade} pokemon card"
    params = {
        "_nkw": query,
        "_sacat": POKEMON_CATEGORY,
        "LH_Complete": "1",
        "LH_Sold": "1",
        "_sop": "13",  # sort: most recent
        "_ipg": "60",  # 60 items per page
    }
    if page > 1:
        params["_pgn"] = str(page)
    return f"{SEARCH_URL}?{urlencode(params)}"


def _parse_search_results(html: str) -> list[dict]:
    """Parse eBay search results page into listing stubs.

    Returns list of dicts with: item_id, title, listing_url, sold_price,
    sold_date, thumbnail_url.
    """
    soup = BeautifulSoup(html, "lxml")
    items = soup.select("li.s-item")
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
        price_el = item_el.select_one("span.s-item__price") or item_el.select_one("span.POSITIVE")
        sold_price = None
        if price_el:
            price_match = re.search(r"\$?([\d,]+\.?\d*)", price_el.get_text(strip=True).replace(",", ""))
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
        img_el = item_el.select_one("img.s-item__image-img") or item_el.select_one("img")
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
# Individual listing page scraping (for full-size images)
# ---------------------------------------------------------------------------

def _scrape_listing_images(session: requests.Session, listing_url: str) -> list[str]:
    """Fetch an individual eBay listing page and extract all image URLs.

    eBay listing pages contain a JSON blob with image URLs in the
    PicturePanel or image gallery section. We extract the highest
    resolution versions available.

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

    # Strategy 1: Extract from JSON-LD structured data
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

    # Strategy 2: Extract from inline JS image array
    # eBay embeds image URLs in a JS variable like:
    #   "maxImageUrl":"https://i.ebayimg.com/images/g/.../s-l1600.jpg"
    for match in re.finditer(r'"maxImageUrl"\s*:\s*"(https?://[^"]+)"', html):
        url = match.group(1)
        if url not in image_urls:
            image_urls.append(url)

    # Strategy 3: Look for image gallery URLs
    for match in re.finditer(r'"(https://i\.ebayimg\.com/images/g/[^"]+/s-l\d+\.(?:jpg|png|webp))"', html):
        url = match.group(1)
        if url not in image_urls:
            image_urls.append(url)

    # Strategy 4: og:image meta tag (usually just the first image)
    if not image_urls:
        og_img = soup.find("meta", property="og:image")
        if og_img and og_img.get("content"):
            image_urls.append(og_img["content"])

    # Deduplicate while preserving order, prefer highest resolution
    seen = set()
    deduped = []
    for url in image_urls:
        # Normalize: upgrade to highest resolution variant
        hi_res = _upgrade_image_url(url)
        key = _url_fingerprint(hi_res)
        if key not in seen:
            seen.add(key)
            deduped.append(hi_res)

    return deduped


def _upgrade_image_url(url: str) -> str:
    """Upgrade eBay image URL to highest resolution.

    eBay image URLs follow the pattern:
        https://i.ebayimg.com/images/g/{hash}/s-l{size}.jpg
    We replace the size suffix with l1600 for max resolution.
    """
    # Replace s-l140, s-l225, s-l300, s-l500 etc. with s-l1600
    upgraded = re.sub(r"/s-l\d+\.", "/s-l1600.", url)
    return upgraded


def _url_fingerprint(url: str) -> str:
    """Extract a stable fingerprint from an eBay image URL.

    Strips the size suffix so different resolutions of the same image
    match as duplicates.
    """
    # Remove size suffix and extension for comparison
    stripped = re.sub(r"/s-l\d+\.\w+$", "", url)
    return stripped


# ---------------------------------------------------------------------------
# PSA cert number extraction
# ---------------------------------------------------------------------------

def _extract_psa_cert(title: str, image_urls: list[str]) -> str | None:
    """Try to extract PSA cert number from title text.

    PSA cert numbers are 7-8 digits. We look for them in the title
    near the "PSA" keyword. Future: OCR on slab images.
    """
    # Check title first
    cert_match = CERT_IN_TITLE.search(title)
    if cert_match:
        return cert_match.group(1)

    # Look for 7-8 digit numbers in the title near "PSA"
    psa_pos = title.upper().find("PSA")
    if psa_pos >= 0:
        # Search in the 60 chars around the PSA mention
        context = title[max(0, psa_pos - 30):psa_pos + 30]
        for m in PSA_CERT_PATTERN.finditer(context):
            num = m.group(1)
            # Filter out numbers that look like card numbers (< 7 digits)
            if len(num) >= 7:
                return num

    return None


# ---------------------------------------------------------------------------
# BGS sub-grade parsing
# ---------------------------------------------------------------------------

def _extract_bgs_subgrades(title: str) -> dict:
    """Extract BGS sub-grades from listing title or description.

    BGS slabs show 4 sub-grades: Centering, Corners, Edges, Surface.
    Sellers often include these in the title, e.g.:
        "BGS 9.5 Centering 9.5 Corners 9.5 Edges 9.5 Surface 10"
        "BGS 9.5 | 9.5/9.5/9/10"
        "BGS 9.5 (9.5, 9.5, 9, 10)"

    Returns dict with keys: centering, corners, edges, surface (all str|None).
    """
    result = {
        "centering": None,
        "corners": None,
        "edges": None,
        "surface": None,
    }

    # Pattern 1: Named sub-grades
    # "Centering 9.5" or "Centering: 9.5"
    for label in BGS_SUBGRADE_LABELS:
        pat = re.compile(
            rf"\b{label}\s*[:=]?\s*([\d]+(?:\.[\d]+)?)\b",
            re.IGNORECASE,
        )
        match = pat.search(title)
        if match:
            result[label] = match.group(1)

    # If we found at least one named sub-grade, return
    if any(v is not None for v in result.values()):
        return result

    # Pattern 2: Slash-separated after BGS grade
    # "BGS 9.5 | 9.5/9.5/9/10" or "BGS 9.5 (9.5/9.5/9/10)"
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

    # Pattern 3: Comma-separated in parentheses
    # "BGS 9.5 (9.5, 9.5, 9, 10)"
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


# ---------------------------------------------------------------------------
# Image downloader
# ---------------------------------------------------------------------------

def _download_image(
    session: requests.Session,
    url: str,
    dest_path: Path,
) -> bool:
    """Download a single image to disk. Returns True on success."""
    if dest_path.exists():
        return True  # already downloaded

    try:
        resp = session.get(url, headers=_get_headers(), timeout=20)
        if resp.status_code != 200:
            logger.debug("HTTP %d downloading %s", resp.status_code, url)
            return False
        # Verify it looks like an image (at least 1KB)
        if len(resp.content) < 1024:
            logger.debug("Suspiciously small image (%d bytes): %s", len(resp.content), url)
            return False
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(resp.content)
        return True
    except requests.RequestException as e:
        logger.debug("Download error for %s: %s", url, e)
        return False


# ---------------------------------------------------------------------------
# Main scraper class
# ---------------------------------------------------------------------------

class ConditionScraper:
    """Scrapes eBay sold listings for graded Pokemon cards.

    Collects listing photos organized by grade for condition ML training.
    Supports PSA and BGS grading authorities with resume capability.
    """

    def __init__(
        self,
        authority: Literal["PSA", "BGS"] = "PSA",
        output_dir: str = DEFAULT_OUTPUT_DIR,
        fetch_listing_images: bool = True,
    ):
        """
        Args:
            authority: Grading authority to search for.
            output_dir: Base directory for downloaded images.
            fetch_listing_images: If True, visit each listing page to get
                full-resolution photos. If False, only save search thumbnails.
        """
        self.authority = authority.upper()
        self.output_dir = Path(output_dir)
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
        """Load progress from disk if it exists."""
        path = self._progress_path()
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self.progress = ScrapeProgress.from_dict(data)
                logger.info(
                    "Resumed %s scrape: %d listings, %d images, grade=%s page=%d",
                    self.authority,
                    self.progress.total_listings,
                    self.progress.total_images,
                    self.progress.current_grade,
                    self.progress.current_page,
                )
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning("Could not load progress file: %s", e)

    def _save_progress(self):
        """Persist progress to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress.last_updated = datetime.now(timezone.utc).isoformat()
        path = self._progress_path()
        with open(path, "w") as f:
            json.dump(self.progress.to_dict(), f, indent=2)

    # -- Listing metadata persistence --

    def _listings_path(self, grade: str) -> Path:
        """Path to the JSONL file storing listing metadata for a grade."""
        safe_grade = grade.replace(".", "_")
        return self.output_dir / self.authority.lower() / f"grade_{safe_grade}" / "listings.jsonl"

    def _append_listing(self, listing: GradedListing):
        """Append a listing record to the grade's JSONL file."""
        path = self._listings_path(listing.grade)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(asdict(listing), default=str) + "\n")

    # -- Image storage --

    def _image_dir(self, grade: str) -> Path:
        """Directory for images of a specific grade."""
        safe_grade = grade.replace(".", "_")
        return self.output_dir / self.authority.lower() / f"grade_{safe_grade}" / "images"

    def _image_path(self, grade: str, item_id: str, idx: int, ext: str = "jpg") -> Path:
        """Path for a specific listing image."""
        return self._image_dir(grade) / f"{item_id}_{idx:02d}.{ext}"

    # -- Core scraping --

    def scrape_grade(self, grade: str, max_pages: int = 5) -> int:
        """Scrape sold listings for a single grade level.

        Args:
            grade: Grade value (e.g. "10", "9.5").
            max_pages: Maximum search result pages to scrape (60 items each).

        Returns:
            Number of new listings scraped.
        """
        # Check if this grade was already completed in a previous run
        if grade in self.progress.completed_grades:
            logger.info("Grade %s %s already completed, skipping.", self.authority, grade)
            return 0

        # Resume from where we left off if this is the current grade
        start_page = 1
        if self.progress.current_grade == grade and self.progress.current_page > 0:
            start_page = self.progress.current_page
            logger.info("Resuming grade %s from page %d", grade, start_page)

        self.progress.current_grade = grade
        new_listings = 0

        for page in range(start_page, max_pages + 1):
            self.progress.current_page = page
            self._save_progress()

            url = _build_search_url(self.authority, grade, page)
            logger.info(
                "[%s %s] Scraping page %d/%d: %s",
                self.authority, grade, page, max_pages, url,
            )

            try:
                resp = self.session.get(url, headers=_get_headers(), timeout=30)
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.error("Failed to fetch search page %d: %s", page, e)
                self.progress.errors += 1
                _polite_delay(5.0, 10.0)  # longer backoff on error
                continue

            stubs = _parse_search_results(resp.text)
            if not stubs:
                logger.info("No results on page %d, stopping grade %s.", page, grade)
                break

            logger.info("Found %d listings on page %d", len(stubs), page)

            for stub in stubs:
                item_id = stub["item_id"]

                # Skip already-scraped listings
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
                            "[%s %s] %d new listings, %d total images",
                            self.authority, grade,
                            new_listings, self.progress.total_images,
                        )

            # Check for next page
            soup = BeautifulSoup(resp.text, "lxml")
            if not soup.select_one("a.pagination__next") and page < max_pages:
                logger.info("No next page after page %d.", page)
                break

            _polite_delay(MIN_DELAY, MAX_DELAY)

        # Mark grade as completed
        self.progress.completed_grades.append(grade)
        self.progress.current_page = 0
        self._save_progress()

        logger.info(
            "[%s %s] Complete: %d new listings scraped.",
            self.authority, grade, new_listings,
        )
        return new_listings

    def _process_listing(self, stub: dict, grade: str) -> GradedListing | None:
        """Process a single listing: parse title, fetch images, download.

        Args:
            stub: Dict from search results with item_id, title, listing_url, etc.
            grade: The grade being scraped (for directory organization).

        Returns:
            GradedListing or None on failure.
        """
        title = stub["title"]
        item_id = stub["item_id"]
        listing_url = stub["listing_url"]

        # Parse title for metadata
        parsed = parse_graded_title(title, self.authority)

        # Build listing object
        listing = GradedListing(
            item_id=item_id,
            title=title,
            authority=parsed.get("authority", self.authority),
            grade=parsed.get("grade") or grade,
            cert_number=parsed.get("cert_number"),
            card_name=parsed.get("card_name"),
            set_name=parsed.get("set_name"),
            card_number=parsed.get("card_number"),
            sold_price=stub.get("sold_price"),
            sold_date=stub.get("sold_date"),
            listing_url=listing_url,
            scraped_at=datetime.now(timezone.utc).isoformat(),
        )

        # Extract PSA cert number
        if self.authority == "PSA":
            listing.cert_number = listing.cert_number or _extract_psa_cert(title, [])

        # Extract BGS sub-grades from title
        if self.authority == "BGS":
            subs = _extract_bgs_subgrades(title)
            listing.sub_centering = subs.get("centering")
            listing.sub_corners = subs.get("corners")
            listing.sub_edges = subs.get("edges")
            listing.sub_surface = subs.get("surface")

        # Fetch full listing page for high-res images
        image_urls = []
        if self.fetch_listing_images and listing_url:
            _polite_delay(LISTING_DELAY_MIN, LISTING_DELAY_MAX)
            image_urls = _scrape_listing_images(self.session, listing_url)
        elif stub.get("thumbnail_url"):
            # Fallback: upgrade thumbnail to larger size
            image_urls = [_upgrade_image_url(stub["thumbnail_url"])]

        listing.image_urls = image_urls

        # Download images
        downloaded = []
        for idx, img_url in enumerate(image_urls):
            ext = "jpg"
            if ".png" in img_url.lower():
                ext = "png"
            elif ".webp" in img_url.lower():
                ext = "webp"

            dest = self._image_path(listing.grade, item_id, idx, ext)
            _polite_delay(IMAGE_DELAY, IMAGE_DELAY + 0.2)

            if _download_image(self.session, img_url, dest):
                downloaded.append(str(dest.relative_to(self.output_dir)))
                self.progress.total_images += 1

        listing.local_images = downloaded
        return listing

    def scrape_all_grades(self, max_pages_per_grade: int = 5) -> dict:
        """Scrape all grades for the configured authority.

        Returns summary dict with per-grade counts.
        """
        grades = PSA_GRADES if self.authority == "PSA" else BGS_GRADES
        summary = {}

        for grade in grades:
            count = self.scrape_grade(grade, max_pages=max_pages_per_grade)
            summary[grade] = count

        total = sum(summary.values())
        logger.info(
            "%s scrape complete: %d total new listings, %d total images",
            self.authority,
            total,
            self.progress.total_images,
        )
        return summary

    def get_stats(self) -> dict:
        """Return current scraping statistics."""
        return {
            "authority": self.authority,
            "total_listings": self.progress.total_listings,
            "total_images": self.progress.total_images,
            "completed_grades": self.progress.completed_grades,
            "current_grade": self.progress.current_grade,
            "current_page": self.progress.current_page,
            "unique_items": len(self.progress.seen_item_ids),
            "errors": self.progress.errors,
            "started_at": self.progress.started_at,
            "last_updated": self.progress.last_updated,
        }


# ---------------------------------------------------------------------------
# PSA cert verification (Phase 2 enhancement)
# ---------------------------------------------------------------------------

def verify_psa_cert(cert_number: str, session: requests.Session | None = None) -> dict | None:
    """Look up a PSA cert number at psacard.com/cert.

    Returns dict with card details if found, None otherwise.
    Rate limit: call sparingly (1 req/5s suggested).

    Note: This hits psacard.com which may have anti-scraping measures.
    Use responsibly and cache results.
    """
    if not cert_number or not cert_number.isdigit():
        return None

    if session is None:
        session = _get_session()

    url = f"https://www.psacard.com/cert/{cert_number}"
    try:
        resp = session.get(url, headers=_get_headers(), timeout=15)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        # PSA cert page has a table with card details
        result = {"cert_number": cert_number, "url": url}

        # Look for spec table rows
        for row in soup.select("table tr, .detail-row, .spec-row"):
            cells = row.find_all(["td", "th", "span", "div"])
            if len(cells) >= 2:
                label = cells[0].get_text(strip=True).lower()
                value = cells[1].get_text(strip=True)
                if "grade" in label:
                    result["grade"] = value
                elif "year" in label:
                    result["year"] = value
                elif "brand" in label:
                    result["brand"] = value
                elif "card" in label or "description" in label:
                    result["description"] = value

        return result if len(result) > 2 else None  # return only if we got data

    except requests.RequestException as e:
        logger.debug("PSA cert lookup failed for %s: %s", cert_number, e)
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for the condition scraper."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape eBay graded Pokemon card listings for condition training data."
    )
    parser.add_argument(
        "--authority", "-a",
        choices=["PSA", "BGS"],
        default="PSA",
        help="Grading authority (default: PSA)",
    )
    parser.add_argument(
        "--grade", "-g",
        type=str,
        default=None,
        help="Specific grade to scrape (e.g. '10', '9.5'). Omit for all grades.",
    )
    parser.add_argument(
        "--all-grades",
        action="store_true",
        help="Scrape all grades for the authority.",
    )
    parser.add_argument(
        "--pages", "-p",
        type=int,
        default=5,
        help="Max search result pages per grade (default: 5, 60 items each).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-listing-images",
        action="store_true",
        help="Skip fetching individual listing pages (faster, thumbnails only).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last saved progress (automatic if progress file exists).",
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

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    scraper = ConditionScraper(
        authority=args.authority,
        output_dir=args.output,
        fetch_listing_images=not args.no_listing_images,
    )

    if args.stats:
        stats = scraper.get_stats()
        print(json.dumps(stats, indent=2))
        return

    if args.grade:
        count = scraper.scrape_grade(args.grade, max_pages=args.pages)
        print(f"\nScraped {count} new {args.authority} {args.grade} listings.")
    elif args.all_grades:
        summary = scraper.scrape_all_grades(max_pages_per_grade=args.pages)
        print(f"\nScrape summary for {args.authority}:")
        for grade, count in summary.items():
            print(f"  Grade {grade}: {count} new listings")
        print(f"  Total: {sum(summary.values())} listings, "
              f"{scraper.progress.total_images} images")
    else:
        parser.print_help()
        print("\nSpecify --grade or --all-grades to start scraping.")
        return

    stats = scraper.get_stats()
    print(f"\nFinal stats: {stats['total_listings']} listings, "
          f"{stats['total_images']} images, {stats['errors']} errors")


if __name__ == "__main__":
    main()
