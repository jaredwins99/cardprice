#!/usr/bin/env python3
"""Scrape Reddit for Pokemon card condition-labeled images.

Searches condition-related posts in Pokemon card subreddits and extracts
image URLs with condition labels from top comments.

Examples:
    # Dry run: see what would be scraped
    python scripts/scrape_reddit_conditions.py --dry-run --limit 20

    # Scrape 50 posts from all default subreddits
    python scripts/scrape_reddit_conditions.py --limit 50

    # Scrape a specific subreddit
    python scripts/scrape_reddit_conditions.py --subreddit pokemoncardvalue --limit 30

    # Verbose output
    python scripts/scrape_reddit_conditions.py --limit 20 -v

Output:
    data/condition_training/reddit/          -- downloaded card images
    data/condition_training/reddit/labels.jsonl  -- one JSON object per image
"""

import argparse
import json
import logging
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SUBREDDITS = ["PokemonTCG", "pokemoncardvalue", "pkmntcg"]

SEARCH_QUERIES = [
    "grade my card",
    "what condition",
    "is this NM",
    "is this near mint",
    "would this get a PSA",
    "card condition",
    "rate my card condition",
    "is this lightly played",
    "centering grade",
]

USER_AGENT = (
    "CardpriceConditionScraper/1.0 "
    "(Pokemon card condition training data collector; "
    "educational/research use)"
)

REQUEST_DELAY = 2.0  # seconds between requests

# Condition keyword patterns
CONDITION_PATTERNS = {
    "NM": [
        r"\bNM\b", r"\bnear\s*mint\b", r"\bnear-mint\b",
    ],
    "LP": [
        r"\bLP\b", r"\blightly\s*played\b", r"\blight\s*play\b",
    ],
    "MP": [
        r"\bMP\b", r"\bmoderately\s*played\b", r"\bmoderate\s*play\b",
    ],
    "HP": [
        r"\bHP\b", r"\bheavily\s*played\b", r"\bheavy\s*play\b",
    ],
    "DMG": [
        r"\bdamaged\b", r"\bDMG\b",
    ],
    "MINT": [
        r"\bmint\b", r"\bgem\s*mint\b",
    ],
}

# PSA grade to condition mapping
PSA_TO_CONDITION = {
    10: "NM", 9: "NM",
    8: "LP", 7: "LP",
    6: "MP", 5: "MP",
    4: "HP", 3: "HP",
    2: "DMG", 1: "DMG",
}

# Regex for PSA/BGS/CGC grade mentions
GRADE_PATTERN = re.compile(
    r"\b(?:PSA|BGS|CGC|SGC)\s*(\d{1,2}(?:\.\d)?)\b", re.IGNORECASE
)

# Image URL patterns
IMAGE_PATTERNS = [
    re.compile(r"https?://i\.redd\.it/\S+\.(?:jpg|jpeg|png|webp)", re.IGNORECASE),
    re.compile(r"https?://(?:i\.)?imgur\.com/\S+\.(?:jpg|jpeg|png|webp)", re.IGNORECASE),
    re.compile(r"https?://preview\.redd\.it/\S+\.(?:jpg|jpeg|png|webp)", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Reddit JSON API helpers
# ---------------------------------------------------------------------------

def reddit_get(url: str, after: str | None = None) -> dict:
    """Fetch a Reddit JSON endpoint with rate limiting."""
    if not url.endswith(".json"):
        url = url.rstrip("/") + ".json"

    params = []
    if after:
        params.append(f"after={after}")
    params.append("limit=25")
    params.append("raw_json=1")

    full_url = url + "?" + "&".join(params)
    logger.debug("GET %s", full_url)

    req = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            logger.warning("Rate limited, sleeping 10s...")
            time.sleep(10)
            return reddit_get(url, after)
        raise

    time.sleep(REQUEST_DELAY)
    return data


def search_subreddit(subreddit: str, query: str, limit: int = 25) -> list[dict]:
    """Search a subreddit and return post data dicts."""
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = [
        f"q={urllib.request.quote(query)}",
        "restrict_sr=on",
        "sort=relevance",
        "t=all",
        f"limit={min(limit, 100)}",
        "raw_json=1",
    ]
    full_url = url + "?" + "&".join(params)
    logger.debug("SEARCH %s", full_url)

    req = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            logger.warning("Rate limited, sleeping 10s...")
            time.sleep(10)
            return search_subreddit(subreddit, query, limit)
        logger.error("HTTP %d searching r/%s for %r", e.code, subreddit, query)
        return []

    time.sleep(REQUEST_DELAY)

    posts = []
    if "data" in data and "children" in data["data"]:
        for child in data["data"]["children"]:
            posts.append(child["data"])
    return posts


def get_post_comments(permalink: str, limit: int = 10) -> list[dict]:
    """Fetch top comments for a post."""
    url = f"https://www.reddit.com{permalink}.json"
    params = [
        f"limit={limit}",
        "sort=top",
        "raw_json=1",
    ]
    full_url = url + "?" + "&".join(params)
    logger.debug("COMMENTS %s", full_url)

    req = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        logger.error("HTTP %d fetching comments for %s", e.code, permalink)
        return []

    time.sleep(REQUEST_DELAY)

    comments = []
    if isinstance(data, list) and len(data) > 1:
        comment_listing = data[1]
        if "data" in comment_listing and "children" in comment_listing["data"]:
            for child in comment_listing["data"]["children"]:
                if child["kind"] == "t1":
                    comments.append(child["data"])
    return comments


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

def extract_image_urls(post: dict) -> list[str]:
    """Extract image URLs from a Reddit post."""
    urls = set()

    # Direct image post
    post_url = post.get("url", "")
    if post_url:
        for pat in IMAGE_PATTERNS:
            if pat.match(post_url):
                urls.add(post_url)
                break

    # Reddit gallery
    if post.get("is_gallery") and "media_metadata" in post:
        for media_id, meta in post["media_metadata"].items():
            if meta.get("status") == "valid":
                # Use the source (full-res) image
                source = meta.get("s", {})
                img_url = source.get("u") or source.get("gif")
                if img_url:
                    # Reddit HTML-encodes ampersands in gallery URLs
                    img_url = img_url.replace("&amp;", "&")
                    urls.add(img_url)

    # Preview images (fallback)
    if not urls and "preview" in post:
        images = post["preview"].get("images", [])
        for img in images:
            source = img.get("source", {})
            img_url = source.get("url", "")
            if img_url:
                img_url = img_url.replace("&amp;", "&")
                urls.add(img_url)

    # Check selftext for image links
    selftext = post.get("selftext", "")
    if selftext:
        for pat in IMAGE_PATTERNS:
            for match in pat.finditer(selftext):
                urls.add(match.group(0))

    return list(urls)


# ---------------------------------------------------------------------------
# Condition label extraction
# ---------------------------------------------------------------------------

def extract_conditions_from_text(text: str) -> list[str]:
    """Extract condition labels from comment text."""
    conditions = []
    text_lower = text.lower()

    for condition, patterns in CONDITION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                conditions.append(condition)
                break

    # PSA/BGS grade mentions
    for match in GRADE_PATTERN.finditer(text):
        grade_val = float(match.group(1))
        grade_int = int(grade_val)
        mapped = PSA_TO_CONDITION.get(grade_int)
        if mapped and mapped not in conditions:
            conditions.append(mapped)

    return conditions


def extract_labels_from_comments(
    comments: list[dict], max_comments: int = 3
) -> tuple[list[str], str, int]:
    """Extract condition labels from top comments.

    Returns:
        (labels, confidence, best_score)
        - labels: deduplicated condition labels
        - confidence: "high", "medium", or "low"
        - best_score: highest comment score among label-bearing comments
    """
    all_labels = []
    label_sets = []  # per-comment label sets for agreement checking
    best_score = 0

    # Sort by score descending, take top N
    scored_comments = sorted(
        comments, key=lambda c: c.get("score", 0), reverse=True
    )

    for comment in scored_comments[:max_comments]:
        body = comment.get("body", "")
        score = comment.get("score", 0)

        labels = extract_conditions_from_text(body)
        if labels:
            all_labels.extend(labels)
            label_sets.append(set(labels))
            best_score = max(best_score, score)

    if not all_labels:
        return [], "low", 0

    # Deduplicate preserving order
    seen = set()
    unique_labels = []
    for lbl in all_labels:
        if lbl not in seen:
            seen.add(lbl)
            unique_labels.append(lbl)

    # Determine confidence
    if len(label_sets) >= 2:
        # Check agreement between commenters
        if label_sets[0] == label_sets[1]:
            confidence = "high"
        elif label_sets[0] & label_sets[1]:
            # Partial overlap
            confidence = "medium"
        else:
            confidence = "low"
    elif best_score >= 10:
        confidence = "high"
    elif best_score >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    return unique_labels, confidence, best_score


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

def download_image(url: str, dest: Path) -> bool:
    """Download an image URL to dest path. Returns True on success."""
    if dest.exists():
        logger.debug("Already exists: %s", dest.name)
        return True

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        dest.write_bytes(data)
        logger.debug("Downloaded %s (%d bytes)", dest.name, len(data))
        return True
    except Exception as e:
        logger.warning("Failed to download %s: %s", url, e)
        return False


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

def scrape_reddit_conditions(
    subreddits: list[str],
    limit: int = 50,
    dry_run: bool = False,
    output_dir: Path = Path("data/condition_training/reddit"),
) -> list[dict]:
    """Main scraper function.

    Returns list of label records (one per image).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_file = output_dir / "labels.jsonl"

    # Load existing records to avoid re-processing
    existing_urls = set()
    if labels_file.exists():
        for line in labels_file.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    existing_urls.add(rec.get("source_url", ""))
                except json.JSONDecodeError:
                    pass
    logger.info("Existing records: %d", len(existing_urls))

    # Collect candidate posts
    seen_post_ids = set()
    candidate_posts = []

    for subreddit in subreddits:
        for query in SEARCH_QUERIES:
            if len(candidate_posts) >= limit * 3:
                # Collect more candidates than needed since many will lack labels
                break

            logger.info("Searching r/%s for %r", subreddit, query)
            posts = search_subreddit(subreddit, query, limit=25)
            for post in posts:
                post_id = post.get("id", "")
                if post_id not in seen_post_ids:
                    seen_post_ids.add(post_id)
                    candidate_posts.append(post)

        if len(candidate_posts) >= limit * 3:
            break

    logger.info("Found %d unique candidate posts", len(candidate_posts))

    # Process posts
    records = []
    processed = 0

    for post in candidate_posts:
        if len(records) >= limit:
            break

        post_id = post.get("id", "")
        title = post.get("title", "")
        permalink = post.get("permalink", "")
        source_url = f"https://www.reddit.com{permalink}"

        # Extract images from the post
        image_urls = extract_image_urls(post)
        if not image_urls:
            logger.debug("No images in post: %s", title[:60])
            continue

        # Also check title for condition mentions
        title_labels = extract_conditions_from_text(title)

        # Fetch comments for condition labels
        logger.info(
            "[%d/%d] Processing: %s (%d images)",
            processed + 1, limit, title[:60], len(image_urls),
        )

        if not dry_run:
            comments = get_post_comments(permalink, limit=10)
        else:
            comments = []

        comment_labels, confidence, best_score = extract_labels_from_comments(comments)

        # Merge title and comment labels
        all_labels = list(dict.fromkeys(title_labels + comment_labels))

        if not all_labels and not dry_run:
            logger.debug("No condition labels found in comments for: %s", title[:60])
            processed += 1
            continue

        if dry_run:
            print(f"\n{'='*70}")
            print(f"Post: {title}")
            print(f"URL:  {source_url}")
            print(f"Images: {len(image_urls)}")
            if title_labels:
                print(f"Title labels: {title_labels}")
            if all_labels:
                print(f"Labels: {all_labels} (confidence: {confidence})")
            for url in image_urls[:3]:
                print(f"  Image: {url[:80]}...")
            processed += 1
            continue

        # Download images and create records
        for i, img_url in enumerate(image_urls):
            if img_url in existing_urls:
                logger.debug("Skipping already-downloaded: %s", img_url[:60])
                continue

            # Generate filename
            ext = _get_extension(img_url)
            filename = f"reddit_{post_id}_{i}{ext}"
            dest = output_dir / filename

            if download_image(img_url, dest):
                record = {
                    "image": filename,
                    "source_url": img_url,
                    "post_url": source_url,
                    "post_title": title,
                    "condition_labels": all_labels,
                    "confidence": confidence,
                    "reddit_score": best_score,
                    "subreddit": post.get("subreddit", ""),
                }
                records.append(record)
                existing_urls.add(img_url)

                # Append to JSONL file
                with open(labels_file, "a") as f:
                    f.write(json.dumps(record) + "\n")

                logger.info(
                    "  Saved %s -> %s (confidence: %s)",
                    filename, all_labels, confidence,
                )

        processed += 1

    return records


def _get_extension(url: str) -> str:
    """Extract file extension from URL."""
    # Strip query params
    path = url.split("?")[0]
    if path.lower().endswith(".png"):
        return ".png"
    if path.lower().endswith(".webp"):
        return ".webp"
    if path.lower().endswith(".jpeg"):
        return ".jpeg"
    return ".jpg"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape Reddit for Pokemon card condition-labeled images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run --limit 20
  %(prog)s --limit 50
  %(prog)s --subreddit pokemoncardvalue --limit 30
  %(prog)s --limit 100 -v
        """,
    )
    parser.add_argument(
        "--limit", "-l",
        type=int, default=50,
        help="Max number of labeled images to collect (default: 50).",
    )
    parser.add_argument(
        "--subreddit", "-s",
        type=str, default=None,
        help="Single subreddit to scrape (default: all three).",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be scraped without downloading.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default="data/condition_training/reddit",
        help="Output directory (default: data/condition_training/reddit).",
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

    subreddits = [args.subreddit] if args.subreddit else DEFAULT_SUBREDDITS
    output_dir = Path(args.output)

    print(f"Reddit condition scraper")
    print(f"  Subreddits: {', '.join(f'r/{s}' for s in subreddits)}")
    print(f"  Limit: {args.limit} images")
    print(f"  Output: {output_dir}")
    print(f"  Dry run: {args.dry_run}")
    print()

    records = scrape_reddit_conditions(
        subreddits=subreddits,
        limit=args.limit,
        dry_run=args.dry_run,
        output_dir=output_dir,
    )

    if args.dry_run:
        print(f"\nDry run complete. Found posts with images above.")
    else:
        print(f"\nDone. Collected {len(records)} labeled images.")
        labels_file = output_dir / "labels.jsonl"
        if labels_file.exists():
            total = sum(1 for _ in labels_file.read_text().splitlines() if _.strip())
            print(f"Total records in {labels_file}: {total}")

        # Print label distribution
        if records:
            from collections import Counter
            label_counts = Counter()
            conf_counts = Counter()
            for r in records:
                for lbl in r["condition_labels"]:
                    label_counts[lbl] += 1
                conf_counts[r["confidence"]] += 1

            print("\nLabel distribution:")
            for lbl, cnt in label_counts.most_common():
                print(f"  {lbl}: {cnt}")
            print("\nConfidence distribution:")
            for conf, cnt in conf_counts.most_common():
                print(f"  {conf}: {cnt}")


if __name__ == "__main__":
    main()
