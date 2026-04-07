#!/usr/bin/env python3
"""Backfill dim_cards_jp.name_jp with Japanese names from TCGdex.

dim_cards_jp.name currently holds romanized English names from TCGPlayer.
This script fetches the actual kanji/kana names from TCGdex per-set and
matches by card_number (with leading-zero normalization).

set_id mapping strategy:
  dim_sets_jp.name has the form "<JA_CODE>: <English Name>" (e.g.
  "SV1S: Scarlet ex"). The prefix before the colon is usually the TCGdex
  JA set ID directly. We try it as-is, and skip sets where the prefix
  isn't a clean alphanumeric code.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request

import psycopg2

API_BASE = "https://api.tcgdex.net/v2/ja"
DB_DSN = "dbname=cardprice"
DELAY = 0.2


def fetch_all_ja_cards() -> list[dict]:
    """Fetch the global /v2/ja/cards list (id + name only, ~6k cards)."""
    url = f"{API_BASE}/cards"
    print(f"Fetching {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "cardprice/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_set(ja_set_id: str) -> list[dict] | None:
    url = f"{API_BASE}/sets/{ja_set_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "cardprice/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("cards", [])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        print(f"  HTTP {e.code} for {ja_set_id}")
        return None
    except (urllib.error.URLError, OSError) as e:
        print(f"  ERROR {ja_set_id}: {e}")
        return None


def strip_zeros(s: str) -> str:
    m = re.match(r"^([A-Za-z]*)0*(\d+)$", s)
    if m:
        prefix, num = m.groups()
        return f"{prefix}{num}" if prefix else num
    return s


def normalize_card_number(raw: str) -> str | None:
    """'015/078' -> '15'; 'TG01/030' -> 'TG01'; '015' -> '15'."""
    if not raw:
        return None
    head = raw.split("/", 1)[0].strip()
    if not head:
        return None
    return strip_zeros(head)


_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,7}$")


def derive_ja_set_id(set_name: str) -> str | None:
    """Extract the TCGdex JA set ID prefix from dim_sets_jp.name."""
    if ":" not in set_name:
        return None
    code = set_name.split(":", 1)[0].strip()
    if _CODE_RE.match(code):
        return code
    return None


def main() -> int:
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute(
        """
        SELECT s.set_id, s.name, c.card_id, c.card_number, c.name
        FROM dim_sets_jp s
        JOIN dim_cards_jp c ON c.set_id = s.set_id
        ORDER BY s.set_id
        """
    )
    rows = cur.fetchall()
    print(f"Loaded {len(rows)} JP cards from DB")

    # Group by set
    by_set: dict[str, dict] = {}
    for set_id, set_name, card_id, card_number, name_en in rows:
        d = by_set.setdefault(
            set_id, {"name": set_name, "cards": []}
        )
        d["cards"].append((card_id, card_number, name_en))

    # Determine sets to fetch
    fetchable: list[tuple[str, str, str]] = []  # (set_id, ja_code, set_name)
    skipped_no_code: list[str] = []
    for set_id, info in by_set.items():
        ja_code = derive_ja_set_id(info["name"])
        if ja_code:
            fetchable.append((set_id, ja_code, info["name"]))
        else:
            skipped_no_code.append(info["name"])

    print(f"Sets with parseable JA code: {len(fetchable)}")
    print(f"Sets skipped (no code prefix): {len(skipped_no_code)}")

    # Build global prefix → {stripped_localId: name} lookup from /v2/ja/cards
    # (case-insensitive prefix since TCGdex mixes e.g. "m1S"/"M1S").
    global_cards = fetch_all_ja_cards()
    print(f"Fetched {len(global_cards)} global JA cards")
    global_lookup: dict[str, dict[str, str]] = {}
    for c in global_cards:
        cid = c.get("id") or ""
        name = c.get("name") or ""
        if "-" not in cid or not name:
            continue
        prefix, local = cid.rsplit("-", 1)
        global_lookup.setdefault(prefix.lower(), {})[strip_zeros(local)] = name
    print(f"Global prefixes available: {len(global_lookup)}")

    updates: list[tuple[str, str]] = []
    set_stats: dict[str, tuple[int, int]] = {}
    failed_sets: list[str] = []
    per_set_cache: dict[str, dict[str, str]] = {}

    for set_id, ja_code, set_name in fetchable:
        ja_key = ja_code.lower()
        ja_lookup = dict(global_lookup.get(ja_key, {}))

        # Enrich via per-set endpoint (has more localIds e.g. secret rares).
        # Only try if the prefix exists in global OR the set looks SV/S/M era
        # modern enough to be worth a network call.
        try_per_set = ja_key in global_lookup or re.match(
            r"^(sv|s|m|pm|pcg|e|neo|web|vs)[0-9a-z]*$", ja_key
        )
        if try_per_set and ja_code not in per_set_cache:
            cards_ja = fetch_set(ja_code)
            time.sleep(DELAY)
            extra: dict[str, str] = {}
            if cards_ja:
                for c in cards_ja:
                    lid = c.get("localId") or ""
                    nm = c.get("name") or ""
                    if lid and nm:
                        extra[strip_zeros(lid)] = nm
            per_set_cache[ja_code] = extra
        if ja_code in per_set_cache:
            for k, v in per_set_cache[ja_code].items():
                ja_lookup.setdefault(k, v)

        if not ja_lookup:
            failed_sets.append(f"{set_name} (tried {ja_code})")
            set_stats[set_id] = (0, len(by_set[set_id]["cards"]))
            continue

        matched = 0
        for card_id, card_number, _name_en in by_set[set_id]["cards"]:
            key = normalize_card_number(card_number) if card_number else None
            if key and key in ja_lookup:
                updates.append((ja_lookup[key], card_id))
                matched += 1
        set_stats[set_id] = (matched, len(by_set[set_id]["cards"]))
        if matched:
            print(
                f"  {set_name[:50]:<50} [{ja_code:<8}] {matched}/{len(by_set[set_id]['cards'])}"
            )

    print(f"\nTotal updates: {len(updates)}")
    print("Writing to DB...")

    BATCH = 1000
    for i in range(0, len(updates), BATCH):
        batch = updates[i : i + BATCH]
        cur.executemany(
            "UPDATE dim_cards_jp SET name_jp = %s WHERE card_id = %s",
            batch,
        )
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM dim_cards_jp WHERE name_jp IS NOT NULL")
    total_set = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM dim_cards_jp")
    total = cur.fetchone()[0]
    print(f"\ndim_cards_jp.name_jp populated: {total_set}/{total} ({100*total_set/total:.1f}%)")

    cur.execute(
        """
        SELECT card_id, name, name_jp
        FROM dim_cards_jp
        WHERE name_jp IS NOT NULL
        ORDER BY random()
        LIMIT 15
        """
    )
    print("\nSample (card_id, name_en, name_jp):")
    for r in cur.fetchall():
        print(f"  {r[0]:<12} {r[1][:30]:<30} {r[2]}")

    if failed_sets:
        print(f"\nFailed/empty sets ({len(failed_sets)}):")
        for s in failed_sets[:30]:
            print(f"  {s}")
        if len(failed_sets) > 30:
            print(f"  ... and {len(failed_sets) - 30} more")

    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
