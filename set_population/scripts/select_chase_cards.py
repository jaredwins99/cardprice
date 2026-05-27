#!/usr/bin/env python3
"""Select each Pokemon set's top ~3 "chase" cards from our own price data.

Part of the set_population sub-project. Chase cards are the highest-value cards
in a set; their graded population (collected separately by collect_psa_pop.py)
is the core anchor signal for relative print-population estimation.

SELECTION RULE
--------------
1. For every card, take its LATEST market price across all subtype printings,
   preferring the highest-value printing (Holofoil / 1st Edition / secret / full
   art). We do this by taking, per card_id, the max market_price over the most
   recent price_date per (card_id, subtype_name) and then the single
   highest-value subtype. This biases toward the chase printing of a card.
2. Assign each card a RARITY TIER (see RARITY_TIER below). Higher tier = rarer /
   chase-ier. We want the top-3 to be COMPARABLE (same pull-rate denominator), so
   within a set we:
     a. find the highest rarity tier present that has >= 1 priced card,
     b. take the top-3 by price from cards in that tier;
     c. if that tier has < 3 priced cards, fall back to ALSO include cards from
        the next-lower tier(s), but never mix a holo/ultra/secret with a plain
        Common/Uncommon unless the set genuinely has nothing rarer.
   This keeps the three chase cards in (nearly) the same rarity class so their
   pull-rate denominators are comparable across sets.
3. Rank the resulting pool by price desc, take top 3.

The per-set output records the rule outcome (tier used, whether a fallback /
tier-mix happened) so downstream code can down-weight low-confidence sets.

Read-only on Postgres (dim_sets, dim_cards, fact_market_prices). stdlib +
psycopg2 only. Writes set_population/data/chase_cards.json.
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import psycopg2

SUBPROJ_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_OUT = os.path.abspath(os.path.join(SUBPROJ_DATA, "chase_cards.json"))
PG_DSN = "dbname=cardprice"  # peer auth via unix socket

# Rarity -> chase tier. Higher number = rarer / more chase-worthy. Cards whose
# rarity isn't listed default to tier 0 (treated like a common). The grouping is
# deliberately coarse: we only need "is this card in the same chase class as the
# other top cards in its set", not a fine-grained rarity ladder.
RARITY_TIER = {
    # tier 5: the apex chase rarities (secret / rainbow / hyper / special art)
    "Rare Secret": 5,
    "Rare Rainbow": 5,
    "Hyper Rare": 5,
    "Mega Hyper Rare": 5,
    "Special Illustration Rare": 5,
    "Rare Shiny": 5,
    "Shiny Rare": 5,
    "Rare Shiny GX": 5,
    "Shiny Ultra Rare": 5,
    "Rare Holo Star": 5,
    "Rare Shining": 5,
    "Rare Prism Star": 5,
    "LEGEND": 5,
    "Classic Collection": 5,
    "Black White Rare": 5,
    # tier 4: ultra / full-art / mechanic chase (EX/GX/V/VMAX/VSTAR/ex/LV.X)
    "Rare Ultra": 4,
    "Ultra Rare": 4,
    "Rare Holo EX": 4,
    "Rare Holo GX": 4,
    "Rare Holo V": 4,
    "Rare Holo VMAX": 4,
    "Rare Holo VSTAR": 4,
    "Rare Holo LV.X": 4,
    "Double Rare": 4,
    "Illustration Rare": 4,
    "Rare BREAK": 4,
    "Rare Prime": 4,
    "Rare ACE": 4,
    "ACE SPEC Rare": 4,
    "Amazing Rare": 4,
    "Radiant Rare": 4,
    "Trainer Gallery Rare Holo": 4,
    "MEGA_ATTACK_RARE": 4,
    # tier 3: standard holo rares
    "Rare Holo": 3,
    # tier 2: non-holo rares
    "Rare": 2,
    # tier 1: uncommons
    "Uncommon": 1,
    # tier 0: commons / promos / unknown
    "Common": 0,
    "Promo": 0,
}

# Preference order among subtype printings of the SAME card when picking the
# single representative chase price. Earlier = preferred (highest-value printing).
SUBTYPE_PRIORITY = [
    "1st Edition Holofoil",
    "1st Edition",
    "Holofoil",
    "Unlimited Holofoil",
    "Foil",
    "Reverse Holofoil",
    "Unlimited",
    "Normal",
]


def tier_of(rarity):
    return RARITY_TIER.get(rarity, 0)


def subtype_rank(subtype):
    try:
        return SUBTYPE_PRIORITY.index(subtype)
    except ValueError:
        return len(SUBTYPE_PRIORITY)  # unknown subtypes sort last


def load_cards(verbose=False):
    """Return {set_id: {"set_name":..,"release_date":..,"cards":[card dict,..]}}.

    Each card dict: card_id, name, card_number, rarity, tier, tcg_product_id,
    price (the representative chase price), subtype (printing the price came from).
    Only cards with at least one non-null market price are included.
    """
    conn = psycopg2.connect(PG_DSN)
    try:
        cur = conn.cursor()
        cur.execute("SELECT set_id, name, release_date FROM dim_sets;")
        sets = {}
        for set_id, name, rd in cur.fetchall():
            sets[set_id] = {
                "set_name": name,
                "release_date": rd.isoformat() if rd else None,
                "cards": [],
            }

        # For each (card_id, subtype) get the latest priced row, then we reduce
        # to one representative price per card in Python (preferring the
        # highest-value printing, breaking ties by price).
        # DISTINCT ON gives us the most recent row per (card_id, subtype_name).
        cur.execute(
            """
            SELECT DISTINCT ON (f.card_id, f.subtype_name)
                   f.card_id, f.subtype_name, f.market_price,
                   c.name, c.set_id, c.card_number, c.rarity, c.tcg_product_id
            FROM fact_market_prices f
            JOIN dim_cards c ON c.card_id = f.card_id
            WHERE f.market_price IS NOT NULL
              AND c.set_id IS NOT NULL
            ORDER BY f.card_id, f.subtype_name, f.price_date DESC;
            """
        )
        # gather per-card printings
        by_card = defaultdict(list)
        meta = {}
        for (card_id, subtype, price, name, set_id, num, rarity, pid) in cur.fetchall():
            by_card[card_id].append((subtype, float(price)))
            meta[card_id] = (name, set_id, num, rarity, pid)
    finally:
        conn.close()

    for card_id, printings in by_card.items():
        name, set_id, num, rarity, pid = meta[card_id]
        if set_id not in sets:
            continue
        # representative printing: prefer the highest-value printing by our
        # SUBTYPE_PRIORITY, but if a non-preferred printing is dramatically more
        # valuable (e.g. a secret-foil variant), price still wins. We choose the
        # max price overall, then record which subtype produced it.
        best_subtype, best_price = max(
            printings,
            key=lambda sp: (sp[1], -subtype_rank(sp[0])),
        )
        sets[set_id]["cards"].append({
            "card_id": card_id,
            "name": name,
            "card_number": num,
            "rarity": rarity,
            "tier": tier_of(rarity),
            "tcg_product_id": pid,
            "price": round(best_price, 2),
            "subtype": best_subtype,
        })

    if verbose:
        n = sum(len(s["cards"]) for s in sets.values())
        print(f"  loaded {n} priced cards across {len(sets)} sets")
    return sets


def pick_chase(cards, top_n=3):
    """Given the priced cards of one set, return (chase_list, selection_meta).

    Implements the tier rule: take the highest tier that has cards, fill from
    lower tiers only if needed to reach top_n. Sort the final pool by price.
    """
    if not cards:
        return [], {"tier_used": None, "tier_mixed": False, "fallback": False,
                    "n_priced_cards": 0}

    # tiers present, high -> low
    tiers = sorted({c["tier"] for c in cards}, reverse=True)
    pool = []
    tiers_used = []
    for t in tiers:
        tier_cards = sorted([c for c in cards if c["tier"] == t],
                            key=lambda c: c["price"], reverse=True)
        pool.extend(tier_cards)
        tiers_used.append(t)
        if len(pool) >= top_n:
            break

    pool.sort(key=lambda c: c["price"], reverse=True)
    chase = pool[:top_n]
    chase_tiers = {c["tier"] for c in chase}
    meta = {
        "tier_used": max(chase_tiers) if chase_tiers else None,
        "tier_mixed": len(chase_tiers) > 1,
        "fallback": len(tiers_used) > 1,  # had to dip below top tier to fill
        "n_priced_cards": len(cards),
    }
    return chase, meta


def build(top_n=3, verbose=False):
    sets = load_cards(verbose=verbose)
    out = {}
    n_mixed = 0
    n_empty = 0
    for set_id, s in sets.items():
        chase, meta = pick_chase(s["cards"], top_n=top_n)
        if not chase:
            n_empty += 1
            continue
        if meta["tier_mixed"]:
            n_mixed += 1
        out[set_id] = {
            "set_name": s["set_name"],
            "release_date": s["release_date"],
            "selection": meta,
            "chase": [
                {
                    "card_id": c["card_id"],
                    "name": c["name"],
                    "card_number": c["card_number"],
                    "rarity": c["rarity"],
                    "price": c["price"],
                    "tcg_product_id": c["tcg_product_id"],
                    "subtype": c["subtype"],
                }
                for c in chase
            ],
        }
    summary = {
        "sets_with_chase": len(out),
        "sets_without_priced_cards": n_empty,
        "sets_with_mixed_tiers": n_mixed,
    }
    return out, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--top-n", type=int, default=3,
                    help="number of chase cards per set (default 3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print summary but do not write JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    sets_out, summary = build(top_n=args.top_n, verbose=args.verbose)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "Per set: take the highest rarity tier present; rank its cards by "
            "latest representative market price (highest-value printing); fill "
            "from next-lower tiers only if fewer than top_n cards exist. "
            "tier_mixed flags sets where the top-N spans >1 rarity tier."
        ),
        "summary": summary,
        "sets": sets_out,
    }

    if not args.dry_run:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"Wrote {args.out}")

    print(f"\nSets with chase cards: {summary['sets_with_chase']}")
    print(f"Sets with no priced cards: {summary['sets_without_priced_cards']}")
    print(f"Sets where top-{args.top_n} mixes rarity tiers: "
          f"{summary['sets_with_mixed_tiers']}")

    # show a few examples
    print("\nExamples:")
    for set_id in ["base1", "neo1", "swsh1", "sv4"]:
        s = sets_out.get(set_id)
        if not s:
            continue
        print(f"  {set_id} ({s['set_name']}) tier={s['selection']['tier_used']} "
              f"mixed={s['selection']['tier_mixed']}")
        for c in s["chase"]:
            print(f"      {c['name']:<22} #{c['card_number'] or '?':<6} "
                  f"{c['rarity'] or '?':<20} ${c['price']:<9.2f} {c['subtype']}")


if __name__ == "__main__":
    main()
