#!/usr/bin/env python3
"""Download Pokemon TCG card data from GitHub mirror and build an attack/move name index."""

import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cardprice.db.session import SessionLocal
from sqlalchemy import text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MIRROR_URL = "https://raw.githubusercontent.com/PokemonTCG/pokemon-tcg-data/master/cards/en/{set_id}.json"


def get_set_ids():
    s = SessionLocal()
    rows = s.execute(text("SELECT set_id FROM dim_sets ORDER BY set_id")).fetchall()
    set_ids = [r[0] for r in rows]
    s.close()
    return set_ids


def download_set(set_id: str) -> list | None:
    url = MIRROR_URL.format(set_id=set_id)
    req = Request(url, headers={"User-Agent": "cardprice/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        if e.code == 404:
            print(f"  SKIP {set_id}: 404 not found")
        else:
            print(f"  ERROR {set_id}: HTTP {e.code}")
        return None
    except URLError as e:
        print(f"  ERROR {set_id}: {e}")
        return None


def main():
    set_ids = get_set_ids()
    print(f"Found {len(set_ids)} sets in database")

    # Download all sets
    all_cards_raw = {}  # api_id -> card data
    failed_sets = []
    for i, set_id in enumerate(set_ids):
        print(f"[{i+1}/{len(set_ids)}] Downloading {set_id}...", end="", flush=True)
        cards = download_set(set_id)
        if cards is None:
            failed_sets.append(set_id)
            continue
        for card in cards:
            all_cards_raw[card["id"]] = card
        print(f" {len(cards)} cards")
        # Small delay to be polite
        if i % 20 == 19:
            time.sleep(0.5)

    print(f"\nDownloaded {len(all_cards_raw)} total cards from {len(set_ids) - len(failed_sets)} sets")
    if failed_sets:
        print(f"Failed sets ({len(failed_sets)}): {failed_sets}")

    # Build card_attacks.json and indexes
    # card_id in DB is like "base1-4/normal", API id is "base1-4"
    # We need to map API ids to all DB variants
    # Get all card_ids from DB to know variants
    s = SessionLocal()
    rows = s.execute(text("SELECT card_id FROM dim_cards")).fetchall()
    db_card_ids = [r[0] for r in rows]
    s.close()

    # Build api_id -> list of db_card_ids
    api_to_db = defaultdict(list)
    for cid in db_card_ids:
        # card_id format: "base1-4/normal" -> api_id "base1-4"
        api_id = cid.split("/")[0]
        api_to_db[api_id].append(cid)

    # Build outputs
    card_attacks_json = {}  # card_id (api format) -> {name, attacks, hp, types}
    attack_to_cards = defaultdict(list)  # attack_name (lower) -> [db_card_ids]
    card_to_attacks = {}  # db_card_id -> [attack_names (lower)]

    cards_with_attacks = 0
    cards_without_attacks = 0

    for api_id, card in all_cards_raw.items():
        attacks = card.get("attacks", [])
        attack_names = [a["name"].lower() for a in attacks]

        # Save raw data
        card_attacks_json[api_id] = {
            "name": card.get("name", ""),
            "attacks": [
                {
                    "name": a["name"],
                    "cost": a.get("cost", []),
                    "damage": a.get("damage", ""),
                    "text": a.get("text", ""),
                }
                for a in attacks
            ],
            "hp": card.get("hp", ""),
            "types": card.get("types", []),
        }

        if attack_names:
            cards_with_attacks += 1
        else:
            cards_without_attacks += 1

        # Map to DB card IDs
        db_ids = api_to_db.get(api_id, [api_id])
        for db_id in db_ids:
            if attack_names:
                card_to_attacks[db_id] = attack_names
                for atk in attack_names:
                    attack_to_cards[atk].append(db_id)

    # Convert defaultdict to regular dict for pickle
    attack_to_cards = dict(attack_to_cards)

    # Save outputs
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save card_attacks.json
    json_path = DATA_DIR / "card_attacks.json"
    with open(json_path, "w") as f:
        json.dump(card_attacks_json, f, indent=1)
    print(f"\nSaved {json_path} ({len(card_attacks_json)} cards)")

    # Save attack_index.pkl
    pkl_path = DATA_DIR / "attack_index.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"attack_to_cards": attack_to_cards, "card_to_attacks": card_to_attacks}, f)
    print(f"Saved {pkl_path}")

    # Stats
    unique_attacks = len(attack_to_cards)
    total_mappings = sum(len(v) for v in attack_to_cards.values())
    print(f"\n--- Stats ---")
    print(f"Total cards downloaded:     {len(all_cards_raw)}")
    print(f"Cards with attacks:         {cards_with_attacks}")
    print(f"Cards without attacks:      {cards_without_attacks}")
    print(f"Unique attack names:        {unique_attacks}")
    print(f"DB card_ids with attacks:   {len(card_to_attacks)}")
    print(f"Total attack->card mappings:{total_mappings}")

    # Top 10 most common attacks
    top = sorted(attack_to_cards.items(), key=lambda x: len(x[1]), reverse=True)[:15]
    print(f"\nTop 15 most common attacks:")
    for name, cards in top:
        print(f"  {name:30s} -> {len(cards)} cards")


if __name__ == "__main__":
    main()
