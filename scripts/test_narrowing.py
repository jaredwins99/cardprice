#!/usr/bin/env python3
"""Analyze how much each signal (name, HP, type) narrows the candidate pool.

For each ground truth card in binder_eval.json, count how many cards in dim_cards
share the same name, name+HP, name+type, and name+HP+type.
"""

import json
import statistics
from pathlib import Path
from sqlalchemy import create_engine, text

EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "binder_eval.json"
OUTPUT_PATH = EVAL_PATH.parent / "narrowing_results.json"
DB_URL = "postgresql+psycopg2://godli@/cardprice"


def load_eval_cards():
    """Load non-empty eval cards from binder_eval.json."""
    with open(EVAL_PATH) as f:
        data = json.load(f)

    cards = []
    for page in data["pages"]:
        for card in page["cards"]:
            if card["card_id"] is not None:
                cards.append(card)
    return cards


def run_queries(engine, cards):
    """Run narrowing queries for each eval card."""
    results = []

    with engine.connect() as conn:
        for card in cards:
            card_id = card["card_id"]
            name = card["name"]

            # Get ground truth card's HP, pokemon_id, and types
            row = conn.execute(
                text("""
                    SELECT c.name, c.hp, p.types
                    FROM dim_cards c
                    LEFT JOIN dim_pokemon p ON c.pokemon_id = p.pokemon_id
                    WHERE c.card_id = :card_id
                """),
                {"card_id": card_id},
            ).fetchone()

            if row is None:
                print(f"WARNING: card_id {card_id} not found in DB, skipping")
                continue

            db_name, hp, types = row
            primary_type = types[0] if types else None

            # 1. Count cards sharing same name (case insensitive)
            name_count = conn.execute(
                text("SELECT count(*) FROM dim_cards WHERE lower(name) = lower(:name)"),
                {"name": db_name},
            ).scalar()

            # 2. Count cards sharing name + HP
            if hp is not None:
                name_hp_count = conn.execute(
                    text("""
                        SELECT count(*) FROM dim_cards
                        WHERE lower(name) = lower(:name) AND hp = :hp
                    """),
                    {"name": db_name, "hp": hp},
                ).scalar()
            else:
                name_hp_count = name_count  # HP is NULL, can't narrow

            # 3. Count cards sharing name + type (primary type via dim_pokemon)
            if primary_type is not None:
                name_type_count = conn.execute(
                    text("""
                        SELECT count(*) FROM dim_cards c
                        JOIN dim_pokemon p ON c.pokemon_id = p.pokemon_id
                        WHERE lower(c.name) = lower(:name)
                          AND :ptype = ANY(p.types)
                    """),
                    {"name": db_name, "ptype": primary_type},
                ).scalar()
            else:
                name_type_count = name_count  # No type info

            # 4. Count cards sharing name + HP + type
            if hp is not None and primary_type is not None:
                name_hp_type_count = conn.execute(
                    text("""
                        SELECT count(*) FROM dim_cards c
                        JOIN dim_pokemon p ON c.pokemon_id = p.pokemon_id
                        WHERE lower(c.name) = lower(:name)
                          AND c.hp = :hp
                          AND :ptype = ANY(p.types)
                    """),
                    {"name": db_name, "hp": hp, "ptype": primary_type},
                ).scalar()
            else:
                name_hp_type_count = name_hp_count if hp is not None else name_type_count

            entry = {
                "card_id": card_id,
                "name": db_name,
                "hp": hp,
                "primary_type": primary_type,
                "candidates_by_name": name_count,
                "candidates_by_name_hp": name_hp_count,
                "candidates_by_name_type": name_type_count,
                "candidates_by_name_hp_type": name_hp_type_count,
            }
            results.append(entry)
            print(
                f"{db_name:25s} | name={name_count:3d} | +hp={name_hp_count:3d} | "
                f"+type={name_type_count:3d} | +hp+type={name_hp_type_count:3d}"
            )

    return results


def compute_stats(results, key):
    vals = [r[key] for r in results]
    return {
        "mean": round(statistics.mean(vals), 2),
        "median": round(statistics.median(vals), 2),
        "min": min(vals),
        "max": max(vals),
        "stdev": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0,
    }


def main():
    engine = create_engine(DB_URL)
    cards = load_eval_cards()
    print(f"Loaded {len(cards)} non-empty eval cards\n")

    results = run_queries(engine, cards)

    signals = [
        "candidates_by_name",
        "candidates_by_name_hp",
        "candidates_by_name_type",
        "candidates_by_name_hp_type",
    ]

    summary = {}
    print("\n=== SUMMARY ===")
    for sig in signals:
        stats = compute_stats(results, sig)
        summary[sig] = stats
        print(f"{sig:30s} | mean={stats['mean']:6.1f} | median={stats['median']:5.1f} | "
              f"min={stats['min']:3d} | max={stats['max']:3d}")

    output = {
        "description": "Candidate pool narrowing analysis for binder_eval.json",
        "num_cards": len(results),
        "per_card": results,
        "summary": summary,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
