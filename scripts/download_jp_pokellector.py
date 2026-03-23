"""Download Japanese Pokemon card images from Pokellector.

Scrapes jp.pokellector.com for card image URLs, then downloads from
den-cards.pokellector.com CDN.

Usage:
    python scripts/download_jp_pokellector.py --list              # List JP sets
    python scripts/download_jp_pokellector.py --set "Expansion Pack"  # Download one set
    python scripts/download_jp_pokellector.py --priority          # WotC-era first
    python scripts/download_jp_pokellector.py --all               # Everything
    python scripts/download_jp_pokellector.py --thumb             # Download thumbnails instead of full-size
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://jp.pokellector.com"
CDN_BASE = "https://den-cards.pokellector.com"
OUTPUT_DIR = Path("data/card_images_jp_pokellector")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# WotC-era JP sets in chronological order
PRIORITY_SETS = [
    "Expansion Pack",
    "Pokemon Jungle",
    "Mystery of the Fossils",
    "Rocket Gang",
    "Leader's Stadium",
    "Challenge from the Darkness",
    "Gold, Silver, to a New World...",
    "Crossing the Ruins",
    "Awakening Legends",
    "Darkness and to Light",
]

PAGE_SCRAPE_DELAY = 1.0  # seconds between page scrapes
CDN_DOWNLOAD_DELAY = 0.3  # seconds between CDN downloads


def get_sets() -> list[dict]:
    """Scrape the sets listing page and return all available JP sets.

    Returns list of dicts with keys: code, name, url, set_id (from logo URL).
    """
    resp = requests.get(f"{BASE_URL}/sets", headers=HEADERS, timeout=30)
    resp.raise_for_status()

    sets = []
    # Pattern: <a class="button" name="CODE" href="/Set-Name-Expansion/" title="...">
    #   ...logo with set_id...<span>Set Name</span></a>
    pattern = re.compile(
        r'<a\s+class="button"\s+name="([^"]*)"\s+href="(/[^"]+-Expansion/)"\s+'
        r'title="[^"]*">'
        r".*?"
        r"<span>([^<]*)</span>",
        re.DOTALL,
    )

    for match in pattern.finditer(resp.text):
        code, url, name = match.groups()
        # Extract set_id from logo URL if present
        logo_match = re.search(
            r"logo\.(\d+)\.png", match.group(0)
        )
        set_id = logo_match.group(1) if logo_match else None
        sets.append({
            "code": code,
            "name": name.strip(),
            "url": url,
            "set_id": set_id,
        })

    return sets


def get_cards_for_set(set_url: str) -> list[dict]:
    """Scrape a set page and extract all card image URLs.

    Returns list of dicts with: name, number, card_id, set_code, set_id,
    thumb_url, full_url.
    """
    full_url = f"{BASE_URL}{set_url}"
    resp = requests.get(full_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    cards = []
    # Pattern: data-src="https://den-cards.pokellector.com/{set_id}/{Name}.{Code}.{Num}.{CardId}.thumb.png"
    pattern = re.compile(
        r'data-src="(https://den-cards\.pokellector\.com/'
        r"(\d+)/([^.]+)\.([^.]+)\.(\d+)\.(\d+)\.thumb\.png)"
        r'"'
    )

    for match in pattern.finditer(resp.text):
        thumb_url = match.group(1)
        set_id = match.group(2)
        card_name = match.group(3)
        set_code = match.group(4)
        card_number = match.group(5)
        card_id = match.group(6)

        # Full-size URL is the same without .thumb
        full_img_url = thumb_url.replace(".thumb.png", ".png")

        cards.append({
            "name": card_name.replace("-", " "),
            "number": int(card_number),
            "card_id": int(card_id),
            "set_code": set_code,
            "set_id": int(set_id),
            "thumb_url": thumb_url,
            "full_url": full_img_url,
            "filename": f"{card_name}.{set_code}.{card_number}.{card_id}.png",
        })

    return cards


def download_set(set_info: dict, use_thumb: bool = False) -> dict:
    """Download all card images for a set.

    Returns stats dict with downloaded/skipped/failed counts.
    """
    set_name = set_info["name"]
    set_url = set_info["url"]
    safe_name = re.sub(r'[^\w\s-]', '', set_name).strip().replace(' ', '-')
    set_dir = OUTPUT_DIR / safe_name

    print(f"\n{'='*60}")
    print(f"Set: {set_name} ({set_info['code']})")
    print(f"URL: {BASE_URL}{set_url}")
    print(f"Output: {set_dir}")
    print(f"{'='*60}")

    # Get card list
    cards = get_cards_for_set(set_url)
    if not cards:
        print(f"  WARNING: No cards found for {set_name}")
        return {"downloaded": 0, "skipped": 0, "failed": 0, "total": 0}

    print(f"  Found {len(cards)} cards")

    # Create output directory
    set_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta_path = set_dir / "cards.json"
    with open(meta_path, "w") as f:
        json.dump({
            "set_name": set_name,
            "set_code": set_info["code"],
            "set_id": set_info.get("set_id"),
            "cards": cards,
        }, f, indent=2)

    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "total": len(cards)}

    for i, card in enumerate(cards):
        img_url = card["thumb_url"] if use_thumb else card["full_url"]
        filename = card["filename"]
        if use_thumb:
            filename = filename.replace(".png", ".thumb.png")
        filepath = set_dir / filename

        # Resume support
        if filepath.exists() and filepath.stat().st_size > 0:
            stats["skipped"] += 1
            continue

        try:
            resp = requests.get(img_url, headers=HEADERS, timeout=30)
            resp.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(resp.content)

            stats["downloaded"] += 1
            size_kb = len(resp.content) / 1024
            print(
                f"  [{i+1}/{len(cards)}] {card['name']} #{card['number']} "
                f"({size_kb:.0f} KB)"
            )

            time.sleep(CDN_DOWNLOAD_DELAY)

        except Exception as e:
            stats["failed"] += 1
            print(f"  [{i+1}/{len(cards)}] FAILED {card['name']}: {e}")

    print(
        f"\n  Done: {stats['downloaded']} downloaded, "
        f"{stats['skipped']} skipped, {stats['failed']} failed"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download Japanese Pokemon card images from Pokellector"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List all JP sets")
    group.add_argument("--set", type=str, help="Download a specific set by name")
    group.add_argument(
        "--priority", action="store_true", help="Download WotC-era priority sets"
    )
    group.add_argument("--all", action="store_true", help="Download all sets")
    parser.add_argument(
        "--thumb", action="store_true", help="Download thumbnails instead of full-size"
    )
    args = parser.parse_args()

    print("Fetching set list from jp.pokellector.com...")
    all_sets = get_sets()
    print(f"Found {len(all_sets)} sets")

    if args.list:
        for s in all_sets:
            print(f"  [{s['code']:>6}] {s['name']}")
        return

    if args.set:
        # Find matching set - prefer exact match, fall back to partial
        target = args.set.lower()
        exact = [s for s in all_sets if s["name"].lower() == target]
        if exact:
            matches = exact
        else:
            matches = [s for s in all_sets if target in s["name"].lower()]
        if not matches:
            print(f"No set matching '{args.set}'. Use --list to see available sets.")
            sys.exit(1)
        if len(matches) > 1:
            print(f"Multiple matches for '{args.set}':")
            for s in matches:
                print(f"  [{s['code']}] {s['name']}")
            print("Be more specific.")
            sys.exit(1)
        sets_to_download = matches

    elif args.priority:
        sets_to_download = []
        for pname in PRIORITY_SETS:
            matches = [s for s in all_sets if pname.lower() in s["name"].lower()]
            if matches:
                sets_to_download.append(matches[0])
            else:
                print(f"  WARNING: Priority set '{pname}' not found")

    elif args.all:
        sets_to_download = all_sets

    total_stats = {"downloaded": 0, "skipped": 0, "failed": 0, "total": 0}
    for i, set_info in enumerate(sets_to_download):
        stats = download_set(set_info, use_thumb=args.thumb)
        for k in total_stats:
            total_stats[k] += stats[k]

        # Delay between set page scrapes
        if i < len(sets_to_download) - 1:
            time.sleep(PAGE_SCRAPE_DELAY)

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_stats['downloaded']} downloaded, "
          f"{total_stats['skipped']} skipped, "
          f"{total_stats['failed']} failed "
          f"(of {total_stats['total']} cards)")


if __name__ == "__main__":
    main()
