#!/usr/bin/env python3
"""Build Japanese card image -> English card_id mapping.

Scans data/card_images_jp/ subdirectories, reads cards.json metadata from each,
and matches to English card_ids in dim_cards by name + rarity.

Output: data/jp_en_card_mapping.json
    {jp_image_path: english_card_id, ...}
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from sqlalchemy import create_engine, text


# Rarity mapping: JustTCG JP rarity -> EN rarity in dim_cards
JP_TO_EN_RARITY = {
    "Common": "Common",
    "Uncommon": "Uncommon",
    "Rare": "Rare",
    "Double Rare": "Double Rare",
    "Holo Rare": "Rare Holo",          # WOTC-era
    "Super Rare Holo": "Rare Holo",     # WOTC-era (e.g., Here Comes Team Rocket!)
    "Super Rare": "Ultra Rare",         # Modern (Full Art)
    "Art Rare": "Illustration Rare",
    "Special Art Rare": "Special Illustration Rare",
    "Ultra Rare": "Hyper Rare",         # Gold cards
}

# Known set mapping: JP set directory -> EN set_id
JP_SET_MAP = {
    "team_rocket_jp": "base5",
    "glory_of_team_rocket_jp": "sv10",
}

# JP card names that differ from EN equivalents
# JP name -> EN name (as it appears in dim_cards)
JP_NAME_ALIASES = {
    "Team Rocket's Surprise Bomb": "Team Rocket's Venture Bomb",
    "Team Rocket Energy": "Team Rocket's Energy",
    "Team Rocket's Hindering Robo": "Team Rocket's Bother-Bot",
    "Team Rocket's Receiver": "Team Rocket's Transceiver",
    "Team Rocket's Nidoran F": "Team Rocket's Nidoran\u2640",
    "Team Rocket's Nidoran M": "Team Rocket's Nidoran\u2642",
}


def _clean_jp_name(name: str) -> str:
    """Strip ' - NUMBER/SETSIZE' suffix from JP card names."""
    return re.sub(r"\s*-\s*\d+/\d+$", "", name).strip()


def _filename_from_card(card: dict) -> str | None:
    """Reconstruct the expected image filename from cards.json entry.

    Format: {Name}_{number}_{setsize}_{tcgplayerId}.jpg
    where Name has spaces replaced with underscores and special chars preserved.
    """
    name = card["name"].replace(" ", "_")
    # Handle apostrophes: "Imposter Oak's Revenge" -> "Imposter_Oaks_Revenge"
    name = name.replace("'", "")
    number = card.get("number", "N/A")
    tcg_id = card.get("tcgplayerId", "")
    if not tcg_id:
        return None

    if "/" in number:
        # Modern format: "125/098" -> parts
        parts = number.split("/")
        return f"{name}_{parts[0]}_{parts[1]}_{tcg_id}.jpg"
    else:
        # WOTC format: "N/A"
        return f"{name}_{number}_{tcg_id}.jpg"


def _find_image_file(jp_dir: Path, card: dict) -> str | None:
    """Find the actual image file for a card by matching tcgplayerId in filename."""
    tcg_id = card.get("tcgplayerId", "")
    if not tcg_id:
        return None

    for f in jp_dir.iterdir():
        if f.suffix.lower() in (".jpg", ".png", ".jpeg") and tcg_id in f.stem:
            return str(f)
    return None


def _build_en_name_index(engine, set_id: str) -> dict:
    """Build {(name, rarity): [card_id, ...]} index for an English set."""
    index = defaultdict(list)
    name_index = defaultdict(list)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT card_id, name, card_number, rarity "
                "FROM dim_cards WHERE set_id = :set_id"
            ),
            {"set_id": set_id},
        )
        for card_id, name, card_number, rarity in rows:
            index[(name, rarity)].append(card_id)
            name_index[name].append((card_id, rarity))
    return index, name_index


def build_mapping(jp_cards_dir: str = "data/card_images_jp") -> dict:
    """Build Japanese card image -> English card_id mapping.

    Returns: dict mapping jp_image_path (relative) -> english_card_id
    """
    base = Path(jp_cards_dir)
    if not base.exists():
        print(f"ERROR: {base} does not exist")
        return {}

    engine = create_engine("postgresql+psycopg2://godli@/cardprice")
    mapping = {}
    stats = {"matched": 0, "unmatched": 0, "skipped": 0}
    unmatched_cards = []

    for jp_dir in sorted(base.iterdir()):
        if not jp_dir.is_dir():
            continue

        cards_json = jp_dir / "cards.json"
        if not cards_json.exists():
            print(f"SKIP: {jp_dir.name} — no cards.json")
            continue

        # Determine EN set_id
        en_set_id = JP_SET_MAP.get(jp_dir.name)
        if not en_set_id:
            print(f"SKIP: {jp_dir.name} — no known EN set mapping")
            continue

        print(f"\nProcessing {jp_dir.name} -> EN set {en_set_id}")

        # Load JP card data
        with open(cards_json) as f:
            jp_cards = json.load(f)

        # Build EN index
        rarity_index, name_index = _build_en_name_index(engine, en_set_id)

        # Filter out non-card items
        jp_cards = [
            c for c in jp_cards
            if not any(
                skip in c["name"]
                for skip in ("Booster Box", "Booster Pack", "Build & Battle")
            )
        ]

        for card in jp_cards:
            # Find the image file
            img_path = _find_image_file(jp_dir, card)
            if not img_path:
                stats["skipped"] += 1
                continue

            rel_path = os.path.relpath(img_path, start=".")

            # Clean the JP card name and apply aliases
            jp_name = _clean_jp_name(card["name"])
            jp_name = JP_NAME_ALIASES.get(jp_name, jp_name)
            jp_rarity = card.get("rarity", "")
            en_rarity = JP_TO_EN_RARITY.get(jp_rarity)

            # Strategy 1: Exact name + mapped rarity
            if en_rarity:
                candidates = rarity_index.get((jp_name, en_rarity), [])
                if len(candidates) == 1:
                    mapping[rel_path] = candidates[0]
                    stats["matched"] += 1
                    continue
                elif len(candidates) > 1:
                    # Multiple cards with same name+rarity: pick lowest card number
                    # (usually the "main" version)
                    mapping[rel_path] = sorted(candidates)[0]
                    stats["matched"] += 1
                    continue

            # Strategy 2: Name-only match (if rarity mapping fails or no match)
            name_matches = name_index.get(jp_name, [])
            if len(name_matches) == 1:
                mapping[rel_path] = name_matches[0][0]
                stats["matched"] += 1
                continue
            elif len(name_matches) > 1:
                # Try to pick by rarity preference:
                # Prefer non-promo, lower card numbers (base version)
                if en_rarity:
                    rarity_filtered = [
                        (cid, r) for cid, r in name_matches if r == en_rarity
                    ]
                    if rarity_filtered:
                        mapping[rel_path] = sorted(rarity_filtered)[0][0]
                        stats["matched"] += 1
                        continue

                # Fall back to lowest card_id (base version)
                mapping[rel_path] = sorted(name_matches, key=lambda x: x[0])[0][0]
                stats["matched"] += 1
                continue

            # No match found
            stats["unmatched"] += 1
            unmatched_cards.append(
                f"  {jp_name} ({jp_rarity}) [{jp_dir.name}]"
            )

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {stats['matched']} matched, {stats['unmatched']} unmatched, "
          f"{stats['skipped']} skipped (no image)")
    if unmatched_cards:
        print(f"\nUnmatched cards:")
        for c in unmatched_cards:
            print(c)

    return mapping


def main():
    mapping = build_mapping()

    out_path = "data/jp_en_card_mapping.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

    print(f"\nWrote {len(mapping)} mappings to {out_path}")

    # Show sample
    print("\nSample mappings:")
    for path, card_id in list(sorted(mapping.items()))[:10]:
        print(f"  {os.path.basename(path)} -> {card_id}")


if __name__ == "__main__":
    main()
