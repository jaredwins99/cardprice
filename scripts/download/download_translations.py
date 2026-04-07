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

# Japanese/Chinese TCGdex set ID → English TCGdex set ID for localId-based matching.
# Japanese/Chinese sets are often split (e.g. SV1S + SV1V → sv01) while English
# combines them. Within these paired sets, localIds match (after stripping zeros).
JA_TCGDEX_TO_EN_TCGDEX_SET = {
    # === Scarlet & Violet era (JA) ===
    "SV1S": "sv01",      # Scarlet ex → Scarlet & Violet
    "SV1V": "sv01",      # Violet ex → Scarlet & Violet
    "SV2D": "sv02",      # Clay Burst → Paldea Evolved
    "SV2P": "sv02",      # Snow Hazard → Paldea Evolved
    "SV2a": "sv03.5",    # Pokemon Card 151 → 151
    "SV3": "sv03",       # Ruler of the Black Flame → Obsidian Flames
    "SV3a": "sv03",      # Raging Surf → Obsidian Flames
    "SV4K": "sv04",      # Ancient Roar → Paradox Rift
    "SV4M": "sv04",      # Future Flash → Paradox Rift
    "SV4a": "sv04.5",    # Shiny Treasure ex → Paldean Fates
    "SV5K": "sv05",      # Wild Force → Temporal Forces
    "SV5a": "sv05",      # Crimson Haze → Temporal Forces
    "SV6": "sv06",       # Mask of Change → Twilight Masquerade
    "SV7": "sv07",       # Stellar Miracle → Stellar Crown
    "SV7a": "sv07",      # Paradise Dragona → Stellar Crown
    "SVK": "sv07",       # Deck Build Box Stellar Miracle → Stellar Crown
    "SVLN": "sv07",      # Starter Set Nymphia ex → Stellar Crown
    "SVLS": "sv07",      # Starter Set Soublades ex → Stellar Crown
    "SV8": "sv08",       # Super Electric Breaker → Surging Sparks
    "SV8a": "sv08.5",    # Terastal Fest ex → Prismatic Evolutions
    "SV9": "sv09",       # Battle Partners → Journey Together
    "SV9a": "sv09",      # Arena of Heat → Journey Together
    "SV10": "sv10",      # Team Rocket's Glory → Destined Rivals
    "SV11B": "sv10.5b",  # Black Bolt → Black Bolt
    "SV11W": "sv10.5w",  # White Flare → White Flare
    # === Sword & Shield era (JA S-series) ===
    "S4": "swsh4",       # Astonishing Volt Tackle → Vivid Voltage
    "S4a": "swsh4",      # Shiny Star V → Vivid Voltage (localIds overlap)
    "S5I": "swsh5",      # Single Strike Master → Battle Styles
    "S5R": "swsh5",      # Rapid Strike Master → Battle Styles
    "S5a": "swsh6",      # Matchless Fighters → Chilling Reign
    "S6H": "swsh6",      # Silver Lance → Chilling Reign
    "S6K": "swsh6",      # Jet-Black Spirit → Chilling Reign
    "S6a": "swsh7",      # Eevee Heroes → Evolving Skies
    "S7D": "swsh7",      # Skyscraping Perfect → Evolving Skies
    "S7R": "swsh7",      # Blue Sky Stream → Evolving Skies
    "S8": "swsh8",       # Fusion Arts → Fusion Strike
    "S8a": "swsh9",      # 25th Anniversary Collection → Brilliant Stars
    "S8b": "swsh12.5",   # VMAX Climax → Crown Zenith
    "S9": "swsh9",       # Star Birth → Brilliant Stars
    "S9a": "swsh10",     # Battle Region → Astral Radiance
    "S10D": "swsh10",    # Time Gazer → Astral Radiance
    "S10P": "swsh10",    # Space Juggler → Astral Radiance
    "S10a": "swsh11",    # Dark Phantasma → Lost Origin
    "S10b": "swsh10.5",  # Pokemon GO → Pokemon GO
    "S11": "swsh11",     # Lost Abyss → Lost Origin
    "S11a": "swsh11",    # Incandescent Arcana → Lost Origin
    "S12": "swsh12",     # Paradigm Trigger → Silver Tempest
    "S12a": "swsh12.5",  # VSTAR Universe → Crown Zenith
    # === Chinese-exclusive sets (SC-series → SV EN sets) ===
    "SC1D": "sv01",      # zh-tw Scarlet → Scarlet & Violet
    "SC1a": "sv01",      # zh-tw SV expansion → Scarlet & Violet
    "SC1b": "sv01",      # zh-tw SV expansion → Scarlet & Violet
    "SC2D": "sv02",      # zh-tw → Paldea Evolved
    "SC2a": "sv03.5",    # zh-tw → 151
    "SC2b": "sv02",      # zh-tw → Paldea Evolved
    "SCA": "sv03",       # zh-tw → Obsidian Flames
    "SCB": "sv04",       # zh-tw → Paradox Rift
    "SCC": "sv05",       # zh-tw → Temporal Forces
    "SCD": "sv06",       # zh-tw → Twilight Masquerade
    "SI": "sv04.5",      # zh-tw → Paldean Fates
    "SH": "sv08.5",      # zh-tw → Prismatic Evolutions
    "SJ": "sv07",        # zh-tw → Stellar Crown
    "SK": "sv09",        # zh-tw → Journey Together
    "SN": "sv10",        # zh-tw → Destined Rivals
    "SVD": "sv08",       # zh-tw → Surging Sparks
    "SVF": "sv06.5",     # zh-tw → Shrouded Fable
    "SVB": "sv08",       # zh-tw → Surging Sparks
    "SVC": "sv08.5",     # zh-tw → Prismatic Evolutions
    "SVEL": "sv10.5w",   # zh-tw → White Flare
    "SVEM": "sv10.5b",   # zh-tw → Black Bolt
    "SVHK": "sv10",      # zh-tw → Destined Rivals
    "SVHM": "sv10",      # zh-tw → Destined Rivals
    "SDM": "sv07",       # zh-tw starter → Stellar Crown
    "SDL": "sv07",       # zh-tw starter → Stellar Crown
    "SLD": "sv07",       # zh-tw starter → Stellar Crown
    "SLL": "sv07",       # zh-tw starter → Stellar Crown
    "SVAL": "sv07",      # zh-tw → Stellar Crown
    "SVAM": "sv07",      # zh-tw → Stellar Crown
    "SVAW": "sv07",      # zh-tw → Stellar Crown
    "SPD": "sv01",       # zh-tw promo → Scarlet & Violet
    "SPZ": "sv01",       # zh-tw promo → Scarlet & Violet
    "SP5": "svp",        # zh-tw promo → SVP Black Star Promos
    "SP6": "svp",        # zh-tw promo → SVP Black Star Promos
    "SVP1": "svp",       # zh-tw promo → SVP Black Star Promos
    # === zh-tw SV-prefix sets (same as JA) ===
    "SV5M": "sv05",      # zh-tw/th Cyber Judge → Temporal Forces
    "SV6a": "sv06.5",    # zh-tw Night Wanderer → Shrouded Fable
    # === Indonesian/Thai "s"-suffix sets ===
    "SV3s": "sv03",      # id/th combined → Obsidian Flames
    "SV4s": "sv04",      # id combined → Paradox Rift
    "SV5s": "sv05",      # id combined → Temporal Forces
    "SV6s": "sv06",      # id combined → Twilight Masquerade
    "SV7s": "sv07",      # id/th combined → Stellar Crown
    "SV8s": "sv08",      # id/th combined → Surging Sparks
    "SV9s": "sv09",      # id/th combined → Journey Together
    "SVDs": "sv10",      # id/th combined → Destined Rivals
    # === Sun & Moon era (JA SM-series) ===
    "SM1S": "sm1",       # Collection Sun → Sun & Moon
    "SM1M": "sm1",       # Collection Moon → Sun & Moon
    "SM2K": "sm2",       # Islands Await You → Guardians Rising
    "SM2L": "sm2",       # Alolan Moonlight → Guardians Rising
    "sm2+": "sm2",       # To Have Seen the Battle Rainbow → Guardians Rising
    "SM3N": "sm3",       # Darkness That Consumes Light → Burning Shadows
    "SM3H": "sm3",       # To Have Seen the Battle Rainbow → Burning Shadows
    "SM3+": "sm3.5",     # Shining Legends → Shining Legends
    "SM4A": "sm4",       # Ultradimensional Beasts → Crimson Invasion
    "SM4S": "sm4",       # Awakened Heroes → Crimson Invasion
    "SM4+": "sm4",       # GX Battle Boost → Crimson Invasion
    "SM5M": "sm5",       # Ultra Moon → Ultra Prism
    "SM5S": "sm5",       # Ultra Sun → Ultra Prism
    "SM5+": "sm5",       # Ultra Force → Ultra Prism
    "SM6": "sm6",        # Forbidden Light → Forbidden Light
    "SM6a": "sm6",       # Dragon Storm → Forbidden Light
    "SM6b": "sm7.5",     # Champion Road → Dragon Majesty
    "SM7": "sm7",        # Charisma of the Wrecked Sky → Celestial Storm
    "SM7a": "sm8",       # Thunderclap Spark → Lost Thunder
    "SM7b": "sm8",       # Fairy Rise → Lost Thunder
    "SM8": "sm8",        # Super-Burst Impact → Lost Thunder
    "SM8a": "sm115",     # Dark Order → Hidden Fates
    "SM8b": "sma",       # GX Ultra Shiny → Hidden Fates Shiny Vault
    "SM9": "sm9",        # Tag Bolt → Team Up
    "SM9a": "sm10",      # Night Unison → Unbroken Bonds
    "SM9b": "sm9",       # Full Metal Wall → Team Up
    "SM10": "sm10",      # Double Blaze → Unbroken Bonds
    "sn10a": "sm11",     # GG End → Unified Minds
    "SM10b": "sm11",     # Sky Legend → Unified Minds
    "sn11": "sm11",      # Miracle Twin → Unified Minds
    "SM11a": "sm11",     # Remix Bout → Unified Minds
    "SM11b": "sm11",     # Dream League → Unified Minds
    "SM12": "sm12",      # Alter Genesis → Cosmic Eclipse
    "SM12a": "sm12",     # Tag All Stars → Cosmic Eclipse
    # === XY era (JA XY-series) ===
    "XY1a": "xy1",       # Collection X → XY
    "XY1b": "xy1",       # Collection Y → XY
    "XY2": "xy2",        # Wild Blaze → Flashfire
    "XY3": "xy3",        # Rising Fist → Furious Fists
    "XY4": "xy4",        # Phantom Gate → Phantom Forces
    "XY5a": "xy5",       # Gaia Volcano / Tidal Storm → Primal Clash
    "CP1": "dc1",        # Double Crisis → Double Crisis
    "XY6": "xy6",        # Emerald Break → Roaring Skies
    "XY7": "xy7",        # Bandit Ring → Ancient Origins
    "XY8a": "xy8",       # Blue Shock → BREAKthrough
    "XY8b": "xy8",       # Red Flash → BREAKthrough
    "XY9": "xy9",        # Rage of the Broken Sky → BREAKpoint
    "CP2": "g1",         # Legendary Shine Collection → Generations
    "CP3": "g1",         # Pokekyun Collection → Generations
    "XY10": "xy10",      # Awakening Psychic King → Fates Collide
    "XY11a": "xy11",     # Cruel Traitor / Explosive Fighter → Steam Siege
    "CP5": "xy11",       # Mythical Legendary Dream Shine Collection → Steam Siege
    "CP6": "xy12",       # 20th Anniversary → Evolutions
    # === HGSS era (JA L-series) ===
    "L1a": "hgss1",      # HeartGold Collection → HeartGold SoulSilver
    "L1b": "hgss1",      # SoulSilver Collection → HeartGold SoulSilver
    "L2": "hgss2",       # Revived Legends → Unleashed
    "LL": "hgss3",       # Lost Link → Undaunted
    "L3": "hgss4",       # Clash at the Summit → Triumphant
    # === Sword & Shield era (JA S-series) — missing ones ===
    "S1H": "swsh1",      # Shield → Sword & Shield
    "S1W": "swsh1",      # Sword → Sword & Shield
    "S1a": "swsh1",      # VMAX Rising → Sword & Shield
    "S2": "swsh2",       # Rebellion Crash → Rebel Clash
    "S2a": "swsh3",      # Explosive Walker → Darkness Ablaze
    "S3": "swsh3",       # Infinity Zone → Darkness Ablaze
    "S3a": "swsh3",      # Legendary Heartbeat → Darkness Ablaze
    # === Mega Evolution era (JA M-series) ===
    "M1S": "me01",       # Mega Symphonia → Mega Evolution
    "M3": "me03",        # Munikis Zero → Perfect Order
    # SV1a (Triplet Beat) intentionally NOT bridged to sv01 — produces
    # bijection collisions on reprinted trainers (e.g. Clavell→Youngster).
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
    foreign_cards: dict[str, str],
    our_ids: set[str],
) -> dict[str, str]:
    """Map Japanese/Chinese cards to our IDs using English TCGdex as bridge.

    These languages use completely different set IDs (e.g. PMCG4 vs base5,
    SV1S vs sv01, SC1D vs sv01). Strategy:
    1. For cards where remapped TCGdex ID matches our ID, keep them (neo1-4).
    2. For modern sets (SV/S/SC era): use JA_TCGDEX_TO_EN_TCGDEX_SET mapping
       to find the corresponding EN TCGdex set, then match by localId within
       that set. This works because localIds match after zero-stripping.
    3. Old sets (PMCG, E, PCG, VS, web) have completely different numbering
       and can't be matched by localId — kept as-is for fuzzy fallback.
    """
    # First pass: direct ID match after standard remapping
    remapped = remap_all_ids(foreign_cards)
    result = {}

    for our_id, name in remapped.items():
        if our_id in our_ids:
            result[our_id] = name

    print(f"    Direct ID match: {len(result)}")

    # Second pass: use JA→EN set mapping + localId matching
    # Build EN TCGdex set → {normalized_localId: remapped_our_id} lookup
    # Normalize localIds by stripping leading zeros (JA/zh-tw use "001", EN uses "1")
    en_by_set: dict[str, dict[str, str]] = {}
    for tcgdex_id, name in en_cards.items():
        parts = tcgdex_id.split("-", 1)
        if len(parts) == 2:
            set_id, local_id = parts
            if set_id not in en_by_set:
                en_by_set[set_id] = {}
            # Remap the EN TCGdex ID to our DB ID
            our_id = remap_card_id(tcgdex_id)
            norm_lid = _strip_leading_zeros(local_id)
            if our_id in our_ids:
                en_by_set[set_id][norm_lid] = our_id

    bridge_matched = 0
    for tcgdex_id, ja_name in foreign_cards.items():
        parts = tcgdex_id.split("-", 1)
        if len(parts) != 2:
            continue
        ja_set, local_id = parts
        local_id = _strip_leading_zeros(local_id)
        en_tcgdex_set = JA_TCGDEX_TO_EN_TCGDEX_SET.get(ja_set)
        if not en_tcgdex_set:
            continue
        en_locals = en_by_set.get(en_tcgdex_set, {})
        our_id = en_locals.get(local_id)
        if our_id and our_id not in result:
            result[our_id] = ja_name
            bridge_matched += 1

    print(f"    Bridge-matched via set+localId: {bridge_matched}")
    print(f"    Total matched: {len(result)}")

    # Merge: result has matched IDs, also keep unmatched for fuzzy fallback
    for k, v in remapped.items():
        if k not in result:
            result[k] = v

    return result


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

    # Always fetch English for cross-referencing non-Western languages
    en_cards = {}
    needs_en = {"ja", "zh-tw", "id", "th"}
    if needs_en & set(to_fetch):
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

        # Remap IDs — languages with non-English set IDs need bridge matching
        if lang in ("ja", "zh-tw", "id", "th") and en_cards:
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
