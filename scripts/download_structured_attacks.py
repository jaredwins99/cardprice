#!/usr/bin/env python3
"""Download structured attack/ability data from the PokemonTCG GitHub mirror.

Fetches all set JSON files, extracts attacks and abilities for each card,
and saves a structured database to data/structured_attacks.json.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "https://raw.githubusercontent.com/PokemonTCG/pokemon-tcg-data/master/cards/en/"
INDEX_URL = "https://api.github.com/repos/PokemonTCG/pokemon-tcg-data/contents/cards/en"
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "structured_attacks.json")


def get_set_files():
    """Get list of all set JSON filenames from GitHub API."""
    req = urllib.request.Request(INDEX_URL)
    req.add_header("User-Agent", "cardprice-bot")
    with urllib.request.urlopen(req, timeout=30) as resp:
        files = json.loads(resp.read())
    return sorted(f["name"] for f in files if f["name"].endswith(".json"))


def fetch_set(filename):
    """Download and parse a single set JSON file."""
    url = BASE_URL + filename
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "cardprice-bot")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def extract_card_data(card):
    """Extract attacks and abilities from a card dict."""
    result = {}

    attacks = card.get("attacks", [])
    if attacks:
        result["attacks"] = [
            {
                "name": a["name"],
                "text": a.get("text", ""),
                "damage": a.get("damage", ""),
                "cost": a.get("cost", []),
            }
            for a in attacks
        ]

    abilities = card.get("abilities", [])
    if abilities:
        result["abilities"] = [
            {
                "name": ab["name"],
                "text": ab.get("text", ""),
                "type": ab.get("type", ""),
            }
            for ab in abilities
        ]

    rules = card.get("rules", [])
    if rules:
        result["rules"] = rules

    return result


def main():
    print("Fetching set file list...")
    set_files = get_set_files()
    print(f"Found {len(set_files)} set files")

    attack_db = {}
    total_cards = 0
    cards_with_attacks = 0
    cards_with_abilities = 0
    total_attacks = 0
    total_abilities = 0
    errors = 0

    for i, filename in enumerate(set_files):
        set_name = filename.replace(".json", "")
        try:
            cards = fetch_set(filename)
            for card in cards:
                card_id = card["id"]
                data = extract_card_data(card)
                if data:  # only store cards that have attacks/abilities/rules
                    attack_db[card_id] = data
                    if "attacks" in data:
                        cards_with_attacks += 1
                        total_attacks += len(data["attacks"])
                    if "abilities" in data:
                        cards_with_abilities += 1
                        total_abilities += len(data["abilities"])
                total_cards += 1

            if (i + 1) % 20 == 0 or i == len(set_files) - 1:
                print(f"  [{i+1}/{len(set_files)}] {set_name}: {len(cards)} cards")

        except Exception as e:
            print(f"  ERROR on {filename}: {e}", file=sys.stderr)
            errors += 1
            time.sleep(1)

    # Save
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(attack_db, f, indent=1)

    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"\nDone! Saved to {OUTPUT}")
    print(f"  Total cards scanned: {total_cards}")
    print(f"  Cards with attack/ability data: {len(attack_db)}")
    print(f"  Cards with attacks: {cards_with_attacks} ({total_attacks} total attacks)")
    print(f"  Cards with abilities: {cards_with_abilities} ({total_abilities} total abilities)")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()
