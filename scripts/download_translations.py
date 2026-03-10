#!/usr/bin/env python3
"""Download multilingual Pokemon card names from TCGdex API."""

import json
import os
import sys
import time
import urllib.request
import urllib.error

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "card_translations.json")
LANGUAGES = ["ja", "fr", "es", "de", "zh-tw"]
API_BASE = "https://api.tcgdex.net/v2"
DELAY = 1.0  # seconds between requests


def fetch_cards(lang: str) -> dict[str, str]:
    """Fetch all card names for a language. Returns {card_id: name}."""
    url = f"{API_BASE}/{lang}/cards"
    print(f"  Fetching {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "cardprice/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"  ERROR fetching {lang}: {e}")
        return {}

    mapping = {}
    for card in data:
        cid = card.get("id")
        name = card.get("name")
        if cid and name:
            mapping[cid] = name
    return mapping


def main():
    # Load existing data if present
    existing: dict[str, dict[str, str]] = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Loaded existing data from {OUTPUT_PATH}")

    missing = [lang for lang in LANGUAGES if lang not in existing]
    if not missing:
        print("All languages already downloaded.")
    else:
        print(f"Languages to fetch: {missing}")
        for i, lang in enumerate(missing):
            if i > 0:
                time.sleep(DELAY)
            print(f"[{lang}]")
            cards = fetch_cards(lang)
            if cards:
                existing[lang] = cards
                print(f"  Got {len(cards)} cards")
            else:
                print(f"  No cards returned for {lang}")

        # Save
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=1)
        print(f"\nSaved to {OUTPUT_PATH}")

    # Stats
    print("\n--- Stats ---")
    all_ids: set[str] = set()
    for lang in LANGUAGES:
        cards = existing.get(lang, {})
        print(f"  {lang}: {len(cards)} cards")
        all_ids.update(cards.keys())
    print(f"  Total unique card IDs: {len(all_ids)}")


if __name__ == "__main__":
    main()
