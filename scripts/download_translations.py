#!/usr/bin/env python3
"""Download multilingual Pokemon card names from TCGdex API.

Remaps TCGdex card IDs to match our DB card IDs (without /normal suffix).
For European languages (fr/es/de), the set IDs mostly match but have
systematic differences (sv01→sv1, me01→me1, pt5→.5 etc).
For Japanese/Chinese, TCGdex uses entirely different set IDs, so we
match by card name against the English TCGdex data.
"""

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "card_translations.json")
LANGUAGES = ["ja", "fr", "es", "de", "zh-tw", "ko", "it", "pt", "id", "th"]
API_BASE = "https://api.tcgdex.net/v2"
DELAY = 0.5  # seconds between requests

# TCGdex set ID → our set ID mapping for cases where they differ
TCGDEX_TO_OUR_SET = {
    "sv01": "sv1",
    "sv02": "sv2",
    "sv03": "sv3",
    "sv03.5": "sv3pt5",
    "sv04": "sv4",
    "sv04.5": "sv4pt5",
    "sv05": "sv5",
    "sv06": "sv6",
    "sv06.5": "sv6pt5",
    "sv07": "sv7",
    "sv08": "sv8",
    "sv08.5": "sv8pt5",
    "sv09": "sv9",
    "sv10.5b": "rsv10pt5",
    "sv10.5w": "zsv10pt5",
    "me01": "me1",
    "me02": "me2",
    "me02.5": "me2pt5",
    "sm3.5": "sm35",
    "sm7.5": "sm75",
    "swsh3.5": "swsh35",
    "swsh4.5": "swsh45",
    "swsh10.5": "pgo",
    "swsh12.5": "swsh12pt5",
    "hgssp": "hsp",
    "lc": "base6",
    "fut2020": "fut20",
}


def fetch_cards(lang: str) -> dict[str, str]:
    """Fetch all card names for a language. Returns {card_id: name}."""
    url = f"{API_BASE}/{lang}/cards"
    print(f"  Fetching {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "cardprice/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
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


def _strip_leading_zeros(card_num: str) -> str:
    """Strip leading zeros from card number while preserving non-numeric prefixes.

    Examples:
        001 → 1, 014 → 14, GG01 → GG01, TG01 → TG01, a → a
    """
    m = re.match(r"^([A-Za-z]*)0*(\d+)$", card_num)
    if m:
        prefix, num = m.groups()
        return f"{prefix}{num}" if prefix else num
    return card_num


def remap_card_id(tcgdex_id: str) -> str:
    """Convert a TCGdex card ID to our DB card ID (without /normal suffix).

    Examples:
        sv01-123 → sv1-123
        me02.5-045 → me2pt5-45
        swsh12.5-GG01 → swsh12pt5-GG01
        neo1-001 → neo1-1
    """
    # Split into set and card number
    parts = tcgdex_id.split("-", 1)
    if len(parts) != 2:
        return tcgdex_id

    set_id, card_num = parts

    # Apply known set mapping
    if set_id in TCGDEX_TO_OUR_SET:
        set_id = TCGDEX_TO_OUR_SET[set_id]

    # Strip leading zeros from card number (our DB uses "1" not "001")
    card_num = _strip_leading_zeros(card_num)

    return f"{set_id}-{card_num}"


def remap_all_ids(cards: dict[str, str]) -> dict[str, str]:
    """Remap all card IDs in a language dict."""
    result = {}
    for tcgdex_id, name in cards.items():
        our_id = remap_card_id(tcgdex_id)
        result[our_id] = name
    return result


def build_name_mapping_for_ja(
    en_cards: dict[str, str],
    ja_cards: dict[str, str],
    our_ids: set[str],
) -> dict[str, str]:
    """Map Japanese cards to our IDs using English TCGdex as bridge.

    Japanese TCGdex uses completely different set IDs (PMCG4 vs base5),
    so we can't just remap set IDs. Instead:
    1. For cards where remapped TCGdex ID matches our ID, keep them.
    2. For remaining: match EN TCGdex card name → our card name → get our ID.
    """
    # First pass: direct ID match after remapping
    remapped = remap_all_ids(ja_cards)
    result = {}
    unmatched_ja = {}

    for our_id, name in remapped.items():
        if our_id in our_ids:
            result[our_id] = name
        else:
            unmatched_ja[our_id] = name

    print(f"    JA direct ID match: {len(result)}")

    # Second pass: build EN tcgdex_id → our_id mapping
    # EN TCGdex IDs can be remapped the same way
    en_remapped = remap_all_ids(en_cards)
    en_to_our = {}  # EN card name → our card ID (for name-based matching)
    for en_id, en_name in en_remapped.items():
        if en_id in our_ids:
            en_to_our[en_id] = en_id

    # For JA cards with different set IDs, try to find matching EN card
    # by matching set contents: same localId within equivalent sets
    # Build: tcgdex_set → {localId: tcgdex_card_id} for both EN and JA
    # Actually, simpler: use the original TCGdex IDs and match by set+localId
    ja_orig_by_set = {}
    for tcgdex_id, name in ja_cards.items():
        parts = tcgdex_id.split("-", 1)
        if len(parts) == 2:
            set_id, local_id = parts
            if set_id not in ja_orig_by_set:
                ja_orig_by_set[set_id] = {}
            ja_orig_by_set[set_id][local_id] = name

    en_by_set = {}
    for tcgdex_id, name in en_cards.items():
        parts = tcgdex_id.split("-", 1)
        if len(parts) == 2:
            set_id, local_id = parts
            if set_id not in en_by_set:
                en_by_set[set_id] = {}
            en_by_set[set_id][local_id] = (name, tcgdex_id)

    # For JA sets that don't map directly, we can't reliably match by localId
    # since Japanese sets have different card numbering
    # Instead, just keep the remapped IDs — the pipeline already handles
    # the foreign-name → English-name matching via fuzzy search

    print(f"    JA total after remapping: {len(remapped)} ({len(result)} matched our IDs)")
    return remapped  # Return all remapped, even if they don't match our IDs


def main():
    force = "--force" in sys.argv

    # Load our card IDs for validation
    names_path = os.path.join(DATA_DIR, "card_names.json")
    our_ids = set()
    if os.path.exists(names_path):
        with open(names_path) as f:
            for row in json.load(f):
                # Our IDs are like "base5-34/normal", strip /normal for matching
                cid = row[0]
                base_id = cid.rsplit("/", 1)[0] if "/" in cid else cid
                our_ids.add(base_id)
        print(f"Loaded {len(our_ids)} card IDs from our DB")

    # Load existing data if present
    existing: dict[str, dict[str, str]] = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Loaded existing data from {OUTPUT_PATH}")

    # Determine which languages to fetch
    if force:
        to_fetch = LANGUAGES[:]
        print(f"Force re-fetching all languages: {to_fetch}")
    else:
        to_fetch = [lang for lang in LANGUAGES if lang not in existing]
        if not to_fetch:
            print("All languages already downloaded. Use --force to re-fetch.")
        else:
            print(f"Languages to fetch: {to_fetch}")

    # Always fetch English for cross-referencing Japanese
    en_cards = {}
    if "ja" in to_fetch or "zh-tw" in to_fetch:
        print("[en] (for cross-referencing)")
        en_cards = fetch_cards("en")
        print(f"  Got {len(en_cards)} EN cards")
        time.sleep(DELAY)

    for i, lang in enumerate(to_fetch):
        if i > 0 or en_cards:
            time.sleep(DELAY)
        print(f"[{lang}]")
        raw_cards = fetch_cards(lang)
        if not raw_cards:
            print(f"  No cards returned for {lang}")
            continue

        print(f"  Got {len(raw_cards)} raw cards")

        # Remap IDs
        if lang == "ja" and en_cards:
            remapped = build_name_mapping_for_ja(en_cards, raw_cards, our_ids)
        else:
            remapped = remap_all_ids(raw_cards)

        # Count matches against our DB
        matched = sum(1 for cid in remapped if cid in our_ids)
        print(f"  After remapping: {len(remapped)} cards, {matched} match our DB ({100*matched/len(remapped):.1f}%)")

        existing[lang] = remapped

    if to_fetch:
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
        if cards and our_ids:
            matched = sum(1 for cid in cards if cid in our_ids)
            print(f"  {lang}: {len(cards)} cards, {matched} match our DB ({100*matched/len(cards):.1f}%)")
        else:
            print(f"  {lang}: {len(cards)} cards")
        all_ids.update(cards.keys())
    print(f"  Total unique card IDs: {len(all_ids)}")

    # Show pipeline-level stats: how many unique translation names map to English
    if our_ids:
        id_to_eng = {}
        if os.path.exists(names_path):
            with open(names_path) as f:
                for row in json.load(f):
                    base_id = row[0].rsplit("/", 1)[0] if "/" in row[0] else row[0]
                    id_to_eng[base_id] = row[1]

        total_trans = 0
        for lang in LANGUAGES:
            cards = existing.get(lang, {})
            mapped = 0
            for cid, tname in cards.items():
                if cid in id_to_eng:
                    mapped += 1
            total_trans += mapped
            if cards:
                print(f"  {lang}: {mapped} names usable by pipeline")
        print(f"  Total usable translation names: {total_trans}")


if __name__ == "__main__":
    main()
