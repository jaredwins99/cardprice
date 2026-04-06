#!/usr/bin/env python3
"""Build comprehensive JA→EN trainer/energy name mapping from TCGdex.

JA and EN trainer cards have completely different localIds within a set
(JA "Nest Ball" might be SV1S-070, while EN "Nest Ball" is sv01-181), so
the localId-bridge that works for Pokemon species fails for trainers.

This script bridges by (illustrator + trainerType) within a JA→EN set
mapping. Illustrator names are stored in Latin script in both languages,
so they match exactly. Within a single bridged set, (illustrator, trainerType)
is usually unique enough to identify the EN counterpart of a JA trainer.

Process:
  1. Fetch the JA Trainer/Energy category lists (cheap, gives id+name).
  2. For every JA T/E card in a known bridged set, fetch the per-card detail
     to get illustrator + trainerType.
  3. For each bridged EN set, fetch per-card details for its T/E cards.
  4. Match (illustrator, trainerType) within (ja_set → en_set).
  5. Merge into data/jp_en_trainer_energy.json (curated entries win).
"""

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "download"))
from download_translations import JA_TCGDEX_TO_EN_TCGDEX_SET, _strip_leading_zeros  # type: ignore

ROOT = os.path.dirname(HERE)
OUTPUT = os.path.join(ROOT, "data", "jp_en_trainer_energy.json")
API = "https://api.tcgdex.net/v2"
WORKERS = 16


def http_get_json(url: str) -> dict | list | None:
    req = urllib.request.Request(url, headers={"User-Agent": "cardprice/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  ERR {url}: {e}")
        return None


def fetch_category_ids(lang: str, category: str) -> list[dict]:
    d = http_get_json(f"{API}/{lang}/categories/{category}")
    return d["cards"] if d else []


def fetch_card_details(lang: str, ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    print(f"  Fetching {len(ids)} {lang} card details with {WORKERS} workers...")
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(http_get_json, f"{API}/{lang}/cards/{cid}"): cid for cid in ids}
        for fut in as_completed(futs):
            cid = futs[fut]
            d = fut.result()
            if isinstance(d, dict):
                out[cid] = d
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(ids)}")
    return out


def main() -> None:
    existing: dict[str, str] = {}
    if os.path.exists(OUTPUT):
        with open(OUTPUT, encoding="utf-8") as f:
            existing = json.load(f)
    curated = {k: v for k, v in existing.items() if not k.startswith("_")}
    print(f"Loaded {len(curated)} existing curated entries")

    # 1. JA category lists
    ja_cards: list[dict] = []
    for cat in ("Trainer", "Energy"):
        print(f"[ja/{cat}]")
        cards = fetch_category_ids("ja", cat)
        print(f"  {len(cards)} cards")
        ja_cards.extend(cards)

    # Restrict to JA sets we have a bridge for.
    bridged_ja_ids = []
    ja_sets_present = set()
    for c in ja_cards:
        cid = c.get("id", "")
        set_id = cid.split("-", 1)[0]
        if set_id in JA_TCGDEX_TO_EN_TCGDEX_SET:
            bridged_ja_ids.append(cid)
            ja_sets_present.add(set_id)
    print(f"JA T/E in bridged sets: {len(bridged_ja_ids)} (across {len(ja_sets_present)} sets)")

    # 2. JA per-card details
    ja_details = fetch_card_details("ja", bridged_ja_ids)

    # 3. EN per-card details — only for the EN sets we need
    en_sets_needed = {JA_TCGDEX_TO_EN_TCGDEX_SET[s] for s in ja_sets_present}
    print(f"EN sets needed: {len(en_sets_needed)}")
    en_target_ids: list[str] = []
    for cat in ("Trainer", "Energy"):
        print(f"[en/{cat}]")
        en_cards = fetch_category_ids("en", cat)
        print(f"  {len(en_cards)} cards")
        for c in en_cards:
            cid = c.get("id", "")
            set_id = cid.split("-", 1)[0]
            if set_id in en_sets_needed:
                en_target_ids.append(cid)
    print(f"EN T/E to fetch: {len(en_target_ids)}")
    en_details = fetch_card_details("en", en_target_ids)

    # Build per-set indices grouped by (illustrator, trainerType).
    # We accept a match ONLY when that key maps to exactly one card on BOTH
    # sides of a JA→EN set bridge. This guarantees a real bijection — no
    # arbitrary tiebreaking, no cross-contamination.
    def group_by_set(details: dict[str, dict]) -> dict[str, dict[tuple[str, str], list[dict]]]:
        out: dict[str, dict[tuple[str, str], list[dict]]] = {}
        for cid, c in details.items():
            set_id = cid.split("-", 1)[0]
            ill = (c.get("illustrator") or "").strip()
            ttype = (c.get("trainerType") or c.get("category") or "").strip()
            name = (c.get("name") or "").strip()
            if not ill or not name:
                continue
            out.setdefault(set_id, {}).setdefault((ill, ttype), []).append(c)
        return out

    ja_by_set = group_by_set(ja_details)
    en_by_set = group_by_set(en_details)

    new_mappings: dict[str, str] = {}
    ambiguous = 0
    no_match = 0
    matched = 0
    samples: list[str] = []
    for ja_set, ja_groups in ja_by_set.items():
        en_set = JA_TCGDEX_TO_EN_TCGDEX_SET.get(ja_set)
        if not en_set:
            continue
        en_groups = en_by_set.get(en_set, {})
        for key, ja_cards in ja_groups.items():
            en_cards = en_groups.get(key, [])
            # Strict 1:1 bijection only — both sides must have a single card
            # with this (illustrator, trainerType) in the bridged set.
            if len(ja_cards) != 1 or len(en_cards) != 1:
                if not en_cards:
                    no_match += 1
                else:
                    ambiguous += 1
                continue
            ja_name = ja_cards[0].get("name", "").strip()
            en_name = en_cards[0].get("name", "").strip()
            if not ja_name or not en_name:
                continue
            if ja_name in new_mappings and new_mappings[ja_name] != en_name:
                continue
            if ja_name not in new_mappings and len(samples) < 40:
                samples.append(f"  [{ja_cards[0]['id']} ↔ {en_cards[0]['id']}] {ja_name} → {en_name}")
            new_mappings[ja_name] = en_name
            matched += 1

    print(f"\nMatched: {matched}")
    print(f"Unique JA→EN names: {len(new_mappings)}")
    print(f"No (illustrator,trainerType) match: {no_match}")
    print(f"Ambiguous (multi-name, non-numeric localId): {ambiguous}")

    # 5. Merge
    merged = dict(existing)
    added = 0
    overrides_skipped = 0
    for ja, en in new_mappings.items():
        if ja in merged:
            overrides_skipped += 1
            continue
        merged[ja] = en
        added += 1

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)

    print(f"\nWrote {OUTPUT}")
    print(f"  Curated preserved: {len(curated)}")
    print(f"  Added from TCGdex bridge: {added}")
    print(f"  Skipped (already in curated): {overrides_skipped}")
    print(f"  Total entries: {sum(1 for k in merged if not k.startswith('_'))}")
    print("\nSample new mappings:")
    for s in samples:
        print(s)


if __name__ == "__main__":
    main()
