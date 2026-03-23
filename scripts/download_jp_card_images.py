#!/usr/bin/env python3
"""Download Japanese Pokemon card reference images.

Pipeline: JustTCG API (pokemon-japan game) -> TCGPlayer CDN images.

Rate limits (free tier):
  - JustTCG: 10 req/min, 100 req/day, 1000 req/month
  - TCGPlayer CDN: no known limit, use 1s delay

Usage:
  python scripts/download_jp_card_images.py --list          # List available sets
  python scripts/download_jp_card_images.py --set "Rocket Gang"  # Download one set
  python scripts/download_jp_card_images.py --priority      # Download priority sets
  python scripts/download_jp_card_images.py --resume        # Resume interrupted download
  python scripts/download_jp_card_images.py --all           # Download everything (multi-day)
  python scripts/download_jp_card_images.py --set "Fossil" --dry-run  # Preview only
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JUSTTCG_BASE = "https://api.justtcg.com/v1"
JUSTTCG_KEY = "tcg_d507de6c16dc43bdaaa29f7f4cece6cd"
JUSTTCG_GAME = "pokemon-japan"

CDN_URL_TEMPLATE = "https://product-images.tcgplayer.com/fit-in/437x437/{tcgplayerId}.jpg"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "card_images_jp")
MANIFEST_PATH = os.path.join(DATA_DIR, "download_manifest.json")

# Rate limiting
JUSTTCG_DELAY = 7.0   # seconds between JustTCG API calls
CDN_DELAY = 1.0        # seconds between CDN image downloads

# JustTCG pagination max
PAGE_SIZE = 20

# Priority sets: WotC-era Japanese sets most likely to be scanned
PRIORITY_SETS = [
    "Rocket Gang",
    "Leaders' Stadium",
    "Challenge from the Darkness",
    "Gym Booster 1",
    "Gym Booster 2",
    "Gold, Silver, to a New World...",
    "Crossing the Ruins...",
    "Awakening Legends",
    "Darkness, and to Light...",
    "Pokemon Jungle",
    "Mystery of the Fossils",
    "Base Set",
    "Base Set 2",
    "Intro Pack",
    "Expansion Pack",
    # Broader matches for gym sets
    "Gym",
    "Neo",
]

HEADERS = {
    "x-api-key": JUSTTCG_KEY,
    "Accept": "application/json",
    "User-Agent": "cardprice-downloader/1.0",
}


# ---------------------------------------------------------------------------
# Manifest (resume support)
# ---------------------------------------------------------------------------

def load_manifest():
    """Load download manifest tracking completed sets and cards."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"completed_sets": [], "downloaded_cards": {}}


def save_manifest(manifest):
    """Save manifest to disk."""
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# JustTCG API helpers
# ---------------------------------------------------------------------------

def _api_get(path, params=None):
    """Make a GET request to JustTCG API. Returns parsed JSON."""
    url = f"{JUSTTCG_BASE}{path}"
    if params:
        query = "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in params.items())
        url = f"{url}?{query}"

    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        if e.code == 429:
            print(f"\n  RATE LIMITED (HTTP 429): {body[:200]}", file=sys.stderr)
            print("  Daily/monthly JustTCG API limit exceeded.", file=sys.stderr)
            print("  Try again tomorrow or use --resume to continue later.", file=sys.stderr)
        else:
            print(f"  HTTP {e.code} from {url}: {body[:200]}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"  Error fetching {url}: {e}", file=sys.stderr)
        raise


def fetch_sets():
    """Fetch all pokemon-japan sets from JustTCG."""
    data = _api_get(f"/games/{JUSTTCG_GAME}/sets")
    # API returns a list of set objects
    if isinstance(data, list):
        return data
    # Or it may be wrapped in a key
    if isinstance(data, dict):
        return data.get("sets", data.get("data", []))
    return []


def fetch_set_cards(set_slug, delay=JUSTTCG_DELAY):
    """Fetch all cards in a set, handling pagination. Returns list of card dicts."""
    all_cards = []
    offset = 0

    while True:
        params = {"limit": PAGE_SIZE, "offset": offset}
        data = _api_get(f"/sets/{set_slug}/cards", params)

        if isinstance(data, list):
            cards = data
        elif isinstance(data, dict):
            cards = data.get("cards", data.get("data", []))
        else:
            break

        if not cards:
            break

        all_cards.extend(cards)
        print(f"    Fetched {len(all_cards)} cards so far (offset={offset})...")

        if len(cards) < PAGE_SIZE:
            break

        offset += PAGE_SIZE
        time.sleep(delay)

    return all_cards


# ---------------------------------------------------------------------------
# Image downloading
# ---------------------------------------------------------------------------

def sanitize_dirname(name):
    """Convert set name to a safe directory name."""
    s = name.lower().strip()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s]+', '_', s)
    return s


def sanitize_filename(name):
    """Convert card name to a safe filename component."""
    s = name.replace(" ", "_")
    s = re.sub(r'[^\w._-]', '', s)
    return s


def image_url(tcgplayer_id):
    """Build TCGPlayer CDN image URL."""
    return CDN_URL_TEMPLATE.format(tcgplayerId=tcgplayer_id)


def download_image(url, filepath):
    """Download an image file. Returns True on success."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "cardprice-downloader/1.0",
        "Referer": "https://www.tcgplayer.com/",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if len(data) < 500:
                print(f"    SKIP tiny image ({len(data)} bytes): {os.path.basename(filepath)}")
                return False
            with open(filepath, "wb") as f:
                f.write(data)
            return True
    except urllib.error.HTTPError as e:
        print(f"    HTTP {e.code} downloading {os.path.basename(filepath)}")
        return False
    except Exception as e:
        print(f"    Error downloading {os.path.basename(filepath)}: {e}")
        return False


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------

def download_set(set_info, manifest, dry_run=False):
    """Download all card images for a single set.

    Args:
        set_info: dict with at least 'slug' and 'name' keys.
                  Optional 'dir_override' to use existing directory.
        manifest: download manifest dict (mutated in place)
        dry_run: if True, print what would be done without downloading

    Returns:
        (downloaded_count, skipped_count, error_count)
    """
    set_slug = set_info["slug"]
    set_name = set_info.get("name", set_slug)

    if set_slug in manifest["completed_sets"]:
        print(f"  SET ALREADY COMPLETE: {set_name} -- skipping")
        return 0, 0, 0

    if "dir_override" in set_info:
        set_dir = set_info["dir_override"]
    else:
        dir_name = sanitize_dirname(set_name)
        set_dir = os.path.join(DATA_DIR, dir_name)

    print(f"\n{'='*60}")
    print(f"Set: {set_name} ({set_slug})")
    print(f"Dir: {set_dir}")
    print(f"{'='*60}")

    # Try local cards.json first, then API
    local_cards_json = os.path.join(set_dir, "cards.json")
    cards = None

    if os.path.exists(local_cards_json):
        print(f"  Loading cached card list from {local_cards_json}")
        with open(local_cards_json) as f:
            cards = json.load(f)
        print(f"  Found {len(cards)} cards (cached)")

    if not cards:
        print(f"  Fetching cards for {set_slug} from API...")
        try:
            cards = fetch_set_cards(set_slug)
            print(f"  Found {len(cards)} cards")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print("  Cannot fetch card list -- API rate limited. Try again later.")
                return 0, 0, 0
            raise

    if not cards:
        print("  No cards found -- skipping")
        return 0, 0, 0

    if dry_run:
        print(f"\n  [DRY RUN] Would download {len(cards)} card images to {set_dir}/")
        for c in cards[:10]:
            name = c.get("name", "Unknown")
            number = str(c.get("number", "N_A")).replace("/", "_")
            tid = c.get("tcgplayerId", "?")
            fname = f"{sanitize_filename(name)}_{number}_{tid}.jpg"
            print(f"    {fname}")
        if len(cards) > 10:
            print(f"    ... and {len(cards) - 10} more")
        return 0, 0, 0

    os.makedirs(set_dir, exist_ok=True)

    # Save cards.json metadata (without priceHistory to save space)
    cards_meta = []
    for c in cards:
        meta = {k: v for k, v in c.items() if k != "variants"}
        # Keep variants but strip priceHistory
        if "variants" in c and c["variants"]:
            meta["variants"] = []
            for v in c["variants"]:
                vm = {k2: v2 for k2, v2 in v.items() if k2 != "priceHistory"}
                meta["variants"].append(vm)
        cards_meta.append(meta)

    cards_json_path = os.path.join(set_dir, "cards.json")
    with open(cards_json_path, "w") as f:
        json.dump(cards_meta, f, indent=2)
    print(f"  Saved {cards_json_path}")

    downloaded = 0
    skipped = 0
    errors = 0

    for i, card in enumerate(cards):
        name = card.get("name", "Unknown")
        number = str(card.get("number", "N_A")).replace("/", "_")
        tid = card.get("tcgplayerId")

        if not tid:
            print(f"  [{i+1}/{len(cards)}] {name} -- no tcgplayerId, skipping")
            errors += 1
            continue

        filename = f"{sanitize_filename(name)}_{number}_{tid}.jpg"
        filepath = os.path.join(set_dir, filename)

        # Check manifest and filesystem
        card_key = f"{set_slug}/{tid}"
        if card_key in manifest["downloaded_cards"] or os.path.exists(filepath):
            skipped += 1
            continue

        url = image_url(tid)
        print(f"  [{i+1}/{len(cards)}] {name} -> {filename}")

        if download_image(url, filepath):
            downloaded += 1
            manifest["downloaded_cards"][card_key] = {
                "name": name,
                "file": filename,
                "timestamp": int(time.time()),
            }
            # Save manifest periodically (every 10 downloads)
            if downloaded % 10 == 0:
                save_manifest(manifest)
        else:
            errors += 1

        time.sleep(CDN_DELAY)

    # Mark set complete if no errors
    if errors == 0:
        manifest["completed_sets"].append(set_slug)
    save_manifest(manifest)

    print(f"\n  Done: {downloaded} downloaded, {skipped} skipped, {errors} errors")
    return downloaded, skipped, errors


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _fetch_sets_or_exit():
    """Fetch sets from JustTCG, exiting gracefully on rate limit."""
    try:
        return fetch_sets()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("API rate limit exceeded. Try again tomorrow.", file=sys.stderr)
            sys.exit(1)
        raise


def cmd_list(args):
    """List all available pokemon-japan sets."""
    print("Fetching sets from JustTCG...")
    sets = _fetch_sets_or_exit()
    print(f"\nFound {len(sets)} sets:\n")
    print(f"{'Set Name':<50} {'Slug':<50} {'Cards':>5}")
    print("-" * 107)
    for s in sorted(sets, key=lambda x: x.get("name", "")):
        name = s.get("name", "?")
        slug = s.get("slug", "?")
        # Card count may not be in set listing -- show if available
        count = s.get("totalCards", s.get("cardCount", "?"))
        print(f"{name:<50} {slug:<50} {count:>5}")


def cmd_set(args):
    """Download a single set by name."""
    target_name = args.set
    print(f"Looking up set: {target_name}")

    sets = _fetch_sets_or_exit()
    time.sleep(JUSTTCG_DELAY)

    # Find matching set (case-insensitive substring match)
    matches = [s for s in sets if target_name.lower() in s.get("name", "").lower()]
    if not matches:
        # Try slug match
        matches = [s for s in sets if target_name.lower() in s.get("slug", "").lower()]

    if not matches:
        print(f"No set found matching '{target_name}'")
        print("Use --list to see available sets")
        sys.exit(1)

    if len(matches) > 1:
        print(f"Multiple sets match '{target_name}':")
        for m in matches:
            print(f"  - {m.get('name')} ({m.get('slug')})")
        print("Please be more specific.")
        sys.exit(1)

    manifest = load_manifest()
    download_set(matches[0], manifest, dry_run=args.dry_run)


def cmd_priority(args):
    """Download priority WotC-era Japanese sets."""
    print("Fetching set list...")
    sets = _fetch_sets_or_exit()
    time.sleep(JUSTTCG_DELAY)

    # Build ordered download queue from priority list
    queue = []
    used_slugs = set()
    for pname in PRIORITY_SETS:
        for s in sets:
            sname = s.get("name", "")
            slug = s.get("slug", "")
            if slug in used_slugs:
                continue
            if pname.lower() in sname.lower():
                queue.append(s)
                used_slugs.add(slug)

    print(f"\nPriority download queue ({len(queue)} sets):")
    for i, s in enumerate(queue, 1):
        print(f"  {i}. {s.get('name')} ({s.get('slug')})")

    if args.dry_run:
        print("\n[DRY RUN] Would download the above sets")
        manifest = load_manifest()
        for s in queue:
            time.sleep(JUSTTCG_DELAY)
            download_set(s, manifest, dry_run=True)
        return

    manifest = load_manifest()
    total_dl = 0
    total_skip = 0
    total_err = 0

    for s in queue:
        dl, skip, err = download_set(s, manifest, dry_run=False)
        total_dl += dl
        total_skip += skip
        total_err += err

    print(f"\n{'='*60}")
    print(f"PRIORITY COMPLETE: {total_dl} downloaded, {total_skip} skipped, {total_err} errors")


def cmd_resume(args):
    """Resume an interrupted download by re-running incomplete sets."""
    manifest = load_manifest()
    completed = set(manifest.get("completed_sets", []))

    # Find sets that have a directory but aren't marked complete
    incomplete = []
    if os.path.exists(DATA_DIR):
        for dirname in sorted(os.listdir(DATA_DIR)):
            dirpath = os.path.join(DATA_DIR, dirname)
            cards_json = os.path.join(dirpath, "cards.json")
            if os.path.isdir(dirpath) and os.path.exists(cards_json):
                with open(cards_json) as f:
                    cards = json.load(f)
                if cards:
                    set_slug = cards[0].get("set", "")
                    set_name = cards[0].get("set_name", dirname)
                    if set_slug and set_slug not in completed:
                        incomplete.append({
                            "slug": set_slug,
                            "name": set_name,
                            "dir_override": dirpath,
                        })

    if not incomplete:
        print("Nothing to resume -- all started sets are complete.")
        return

    print(f"Found {len(incomplete)} incomplete sets:")
    for s in incomplete:
        print(f"  - {s['name']} ({s['slug']})")

    manifest = load_manifest()
    for s in incomplete:
        download_set(s, manifest, dry_run=args.dry_run)


def cmd_all(args):
    """Download ALL pokemon-japan sets."""
    print("Fetching complete set list...")
    sets = fetch_sets()
    time.sleep(JUSTTCG_DELAY)

    print(f"Found {len(sets)} sets total")

    # Sort: priority sets first, then alphabetical
    priority_lower = {p.lower() for p in PRIORITY_SETS}

    def sort_key(s):
        name = s.get("name", "").lower()
        is_priority = any(p in name for p in priority_lower)
        return (0 if is_priority else 1, name)

    sets_sorted = sorted(sets, key=sort_key)

    manifest = load_manifest()
    total_dl = 0
    total_skip = 0
    total_err = 0

    for i, s in enumerate(sets_sorted, 1):
        slug = s.get("slug", "")
        name = s.get("name", slug)
        if slug in manifest["completed_sets"]:
            print(f"[{i}/{len(sets_sorted)}] {name} -- already complete, skipping")
            continue

        print(f"\n[{i}/{len(sets_sorted)}] Processing: {name}")
        dl, skip, err = download_set(s, manifest, dry_run=args.dry_run)
        total_dl += dl
        total_skip += skip
        total_err += err

    print(f"\n{'='*60}")
    print(f"ALL SETS COMPLETE: {total_dl} downloaded, {total_skip} skipped, {total_err} errors")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download Japanese Pokemon card reference images from JustTCG/TCGPlayer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List available sets")
    group.add_argument("--set", type=str, help="Download a specific set by name")
    group.add_argument("--priority", action="store_true", help="Download priority WotC-era sets")
    group.add_argument("--resume", action="store_true", help="Resume interrupted download")
    group.add_argument("--all", action="store_true", help="Download all sets (multi-day)")

    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without actually downloading")

    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.list:
        cmd_list(args)
    elif args.set:
        cmd_set(args)
    elif args.priority:
        cmd_priority(args)
    elif args.resume:
        cmd_resume(args)
    elif args.all:
        cmd_all(args)


if __name__ == "__main__":
    main()
