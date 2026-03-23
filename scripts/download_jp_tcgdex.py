#!/usr/bin/env python3
"""Download Japanese Pokemon card images from TCGdex.

TCGdex is open-source with no API key required and no strict rate limits.
Japanese card images are available for Scarlet & Violet era sets (~3,200 cards)
and a few late Sword & Shield sets.

Older sets (WotC/Neo/e-Card/ADV/PCG/DP/BW/XY/SM) have card metadata but NO images.

Usage:
  python scripts/download_jp_tcgdex.py --list           # List sets with image availability
  python scripts/download_jp_tcgdex.py --download       # Download all available JP images
  python scripts/download_jp_tcgdex.py --set SV1S       # Download a specific set
  python scripts/download_jp_tcgdex.py --download --dry-run  # Preview only
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TCGDEX_API = "https://api.tcgdex.net/v2/ja"
TCGDEX_ASSETS = "https://assets.tcgdex.net"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "card_images_jp")
MANIFEST_PATH = os.path.join(DATA_DIR, "tcgdex_manifest.json")

# Be polite but TCGdex has no strict rate limit
REQUEST_DELAY = 0.2  # seconds between API calls
IMAGE_WORKERS = 4     # parallel image downloads
IMAGE_QUALITY = "high"  # "high" (600x825) or "low" (245x342)
IMAGE_FORMAT = "png"   # "png", "jpg", or "webp"

HEADERS = {
    "User-Agent": "cardprice-downloader/1.0",
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(path):
    """GET from TCGdex API. Returns parsed JSON."""
    url = f"{TCGDEX_API}{path}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} from {url}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"  Error fetching {url}: {e}", file=sys.stderr)
        raise


def fetch_all_sets():
    """Fetch all Japanese sets with card counts."""
    return api_get("/sets")


def fetch_set_detail(set_id):
    """Fetch set detail including card list."""
    return api_get(f"/sets/{set_id}")


def fetch_card_detail(card_id):
    """Fetch full card detail."""
    return api_get(f"/cards/{card_id}")


# ---------------------------------------------------------------------------
# Manifest for resume support
# ---------------------------------------------------------------------------

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"completed_sets": [], "downloaded_cards": []}


def save_manifest(manifest):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Image downloading
# ---------------------------------------------------------------------------

def sanitize_name(name):
    """Convert name to filesystem-safe string."""
    s = re.sub(r'[^\w\s.-]', '', name)
    s = re.sub(r'\s+', '_', s.strip())
    return s


def download_card_image(card_id, image_base_url, dest_dir):
    """Download a single card image. Returns (card_id, success, filepath)."""
    filename = f"{card_id.replace('/', '_')}.{IMAGE_FORMAT}"
    filepath = os.path.join(dest_dir, filename)

    if os.path.exists(filepath):
        return card_id, True, filepath  # already exists

    url = f"{image_base_url}/{IMAGE_QUALITY}.{IMAGE_FORMAT}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if len(data) < 500:
                return card_id, False, filepath
            with open(filepath, "wb") as f:
                f.write(data)
            return card_id, True, filepath
    except Exception as e:
        return card_id, False, str(e)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(args):
    """List all Japanese sets with image availability."""
    print("Fetching Japanese sets from TCGdex...")
    sets = fetch_all_sets()
    print(f"\nFound {len(sets)} Japanese sets\n")

    # Check each set for images by fetching detail
    has_images = []
    no_images = []
    empty_sets = []

    for i, s in enumerate(sets):
        set_id = s["id"]
        set_name = s.get("name", set_id)
        total = s.get("cardCount", {}).get("total", 0)

        if i > 0 and i % 10 == 0:
            print(f"  Checked {i}/{len(sets)} sets...", file=sys.stderr)

        try:
            detail = fetch_set_detail(set_id)
            cards = detail.get("cards", [])
            time.sleep(REQUEST_DELAY)
        except Exception:
            empty_sets.append((set_id, set_name, total, "ERROR"))
            continue

        if not cards:
            empty_sets.append((set_id, set_name, total, "no cards in API"))
            continue

        first_has_image = cards[0].get("image") is not None
        if first_has_image:
            has_images.append((set_id, set_name, len(cards)))
        else:
            no_images.append((set_id, set_name, len(cards)))

    # Print results
    total_downloadable = sum(c for _, _, c in has_images)

    print(f"\n{'='*70}")
    print(f"SETS WITH IMAGES ({len(has_images)} sets, {total_downloadable} cards)")
    print(f"{'='*70}")
    print(f"{'ID':<12} {'Name':<40} {'Cards':>6}")
    print("-" * 60)
    for sid, name, count in has_images:
        print(f"{sid:<12} {name:<40} {count:>6}")

    print(f"\n{'='*70}")
    print(f"SETS WITHOUT IMAGES ({len(no_images)} sets)")
    print(f"{'='*70}")
    for sid, name, count in no_images:
        print(f"  {sid:<12} {name:<40} ({count} cards, metadata only)")

    print(f"\n{'='*70}")
    print(f"EMPTY SETS ({len(empty_sets)} sets — cards not yet in API)")
    print(f"{'='*70}")
    for sid, name, total, reason in empty_sets[:20]:
        print(f"  {sid:<12} {name:<40} (listed: {total}, {reason})")
    if len(empty_sets) > 20:
        print(f"  ... and {len(empty_sets) - 20} more")


def cmd_download(args):
    """Download all available Japanese card images."""
    print("Fetching Japanese sets from TCGdex...")
    sets = fetch_all_sets()
    manifest = load_manifest()
    completed = set(manifest.get("completed_sets", []))
    downloaded_cards = set(manifest.get("downloaded_cards", []))

    total_dl = 0
    total_skip = 0
    total_err = 0

    # Filter to specific set if requested
    if args.set:
        sets = [s for s in sets if s["id"] == args.set]
        if not sets:
            print(f"Set '{args.set}' not found")
            sys.exit(1)

    for s in sets:
        set_id = s["id"]
        set_name = s.get("name", set_id)

        if set_id in completed:
            continue

        try:
            detail = fetch_set_detail(set_id)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"  Error fetching {set_id}: {e}")
            continue

        cards = detail.get("cards", [])
        if not cards:
            continue

        # Check if first card has image
        if not cards[0].get("image"):
            continue

        # Filter to cards not yet downloaded
        to_download = []
        for c in cards:
            cid = c["id"]
            img = c.get("image")
            if not img:
                continue
            if cid in downloaded_cards:
                total_skip += 1
                continue
            to_download.append((cid, img))

        if not to_download:
            if set_id not in completed:
                manifest["completed_sets"].append(set_id)
                save_manifest(manifest)
            continue

        # Create set directory
        dir_name = sanitize_name(f"{set_id}_{set_name}")
        set_dir = os.path.join(DATA_DIR, dir_name)

        print(f"\n{'='*60}")
        print(f"Set: {set_name} ({set_id}) — {len(to_download)} to download")
        print(f"Dir: {set_dir}")

        if args.dry_run:
            print(f"  [DRY RUN] Would download {len(to_download)} images")
            for cid, _ in to_download[:5]:
                print(f"    {cid}")
            if len(to_download) > 5:
                print(f"    ... and {len(to_download) - 5} more")
            continue

        os.makedirs(set_dir, exist_ok=True)

        # Save card metadata
        meta_path = os.path.join(set_dir, "cards.json")
        with open(meta_path, "w") as f:
            json.dump(cards, f, indent=2, ensure_ascii=False)

        # Download images in parallel
        dl_count = 0
        err_count = 0

        with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool:
            futures = {
                pool.submit(download_card_image, cid, img_url, set_dir): cid
                for cid, img_url in to_download
            }
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    card_id, success, path = future.result()
                    if success:
                        dl_count += 1
                        downloaded_cards.add(card_id)
                        manifest["downloaded_cards"].append(card_id)
                        if dl_count % 50 == 0:
                            save_manifest(manifest)
                            print(f"  Progress: {dl_count}/{len(to_download)}")
                    else:
                        err_count += 1
                        print(f"  FAIL: {card_id} — {path}")
                except Exception as e:
                    err_count += 1
                    print(f"  ERROR: {cid} — {e}")

        if err_count == 0:
            manifest["completed_sets"].append(set_id)
        save_manifest(manifest)

        print(f"  Done: {dl_count} downloaded, {err_count} errors")
        total_dl += dl_count
        total_err += err_count

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_dl} downloaded, {total_skip} skipped, {total_err} errors")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download Japanese Pokemon card images from TCGdex (no API key needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true",
                       help="List all sets with image availability")
    group.add_argument("--download", action="store_true",
                       help="Download all available images")

    parser.add_argument("--set", type=str,
                        help="Limit to a specific set ID (e.g. SV1S)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without downloading")

    args = parser.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)

    if args.list:
        cmd_list(args)
    elif args.download:
        cmd_download(args)


if __name__ == "__main__":
    main()
