#!/usr/bin/env python3
"""Benchmark: SequenceMatcher vs rapidfuzz for attack name matching.

Loads the attack index from data/card_attacks.json, generates 50 realistic
OCR-garbled attack names, and times both approaches head-to-head.

Usage:
    python scripts/bench/bench_rapidfuzz.py
"""

from __future__ import annotations

import json
import random
import string
import time
from difflib import SequenceMatcher
from pathlib import Path

from rapidfuzz import fuzz as rfuzz
from rapidfuzz import process as rprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ATTACK_JSON = PROJECT_ROOT / "data" / "card_attacks.json"


# ---------------------------------------------------------------------------
# OCR garbling: simulate realistic OCR errors
# ---------------------------------------------------------------------------

def garble_name(name: str, rng: random.Random) -> str:
    """Apply 1-3 realistic OCR-style mutations to an attack name.

    Mutations: char substitution, char deletion, char insertion, char swap.
    """
    chars = list(name)
    if not chars:
        return name

    n_mutations = rng.randint(1, 3)
    for _ in range(n_mutations):
        if not chars:
            break
        op = rng.choice(["sub", "del", "ins", "swap"])
        idx = rng.randint(0, len(chars) - 1)

        if op == "sub":
            # Common OCR confusions
            confusions = {
                "l": "1", "1": "l", "O": "0", "0": "O",
                "I": "l", "S": "5", "5": "S", "B": "8",
                "g": "9", "Z": "2", "m": "rn", "n": "ri",
            }
            c = chars[idx]
            if c in confusions:
                chars[idx] = confusions[c]
            else:
                chars[idx] = rng.choice(string.ascii_lowercase)
        elif op == "del" and len(chars) > 2:
            chars.pop(idx)
        elif op == "ins":
            chars.insert(idx, rng.choice(string.ascii_lowercase))
        elif op == "swap" and idx < len(chars) - 1:
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

    return "".join(chars)


# ---------------------------------------------------------------------------
# Old approach: loop + SequenceMatcher
# ---------------------------------------------------------------------------

def match_sequencematcher(
    query: str, choices: list[str], threshold: float = 0.60,
) -> tuple[str | None, float]:
    """Brute-force SequenceMatcher scan over all choices."""
    best_match = None
    best_score = 0.0
    q = query.lower().strip()
    for choice in choices:
        score = SequenceMatcher(None, q, choice.lower().strip()).ratio()
        if score > best_score:
            best_score = score
            best_match = choice
    if best_score >= threshold:
        return best_match, best_score
    return None, 0.0


# ---------------------------------------------------------------------------
# New approach: rapidfuzz.process.extractOne
# ---------------------------------------------------------------------------

def match_rapidfuzz(
    query: str, choices: list[str], threshold: float = 0.60,
) -> tuple[str | None, float]:
    """rapidfuzz extractOne (C-accelerated)."""
    result = rprocess.extractOne(
        query.lower().strip(),
        choices,
        scorer=rfuzz.ratio,
        score_cutoff=threshold * 100,
        processor=lambda s: s.lower().strip(),
    )
    if result:
        return result[0], result[1] / 100.0
    return None, 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load attack names
    if not ATTACK_JSON.exists():
        print(f"ERROR: {ATTACK_JSON} not found")
        return

    with open(ATTACK_JSON) as f:
        data = json.load(f)

    # Expect {"attack_to_cards": {...}, ...} or a flat dict/list
    if isinstance(data, dict) and "attack_to_cards" in data:
        all_attacks = sorted(data["attack_to_cards"].keys())
    elif isinstance(data, dict):
        all_attacks = sorted(data.keys())
    elif isinstance(data, list):
        all_attacks = sorted(set(data))
    else:
        print("ERROR: unexpected JSON structure")
        return

    print(f"Loaded {len(all_attacks)} unique attack names from {ATTACK_JSON.name}")

    # Generate 50 garbled OCR fragments
    rng = random.Random(42)
    sample_size = min(50, len(all_attacks))
    sampled_attacks = rng.sample(all_attacks, sample_size)
    queries = [(garble_name(name, rng), name) for name in sampled_attacks]

    print(f"Generated {len(queries)} garbled OCR queries\n")
    print(f"{'Query':<30} {'Original':<25} {'SM Match':<25} {'RF Match':<25} {'Same?'}")
    print("-" * 130)

    # --- Benchmark SequenceMatcher ---
    sm_results = []
    t0 = time.perf_counter()
    for query, _original in queries:
        match, score = match_sequencematcher(query, all_attacks)
        sm_results.append((match, score))
    sm_time = time.perf_counter() - t0

    # --- Benchmark rapidfuzz ---
    rf_results = []
    t0 = time.perf_counter()
    for query, _original in queries:
        match, score = match_rapidfuzz(query, all_attacks)
        rf_results.append((match, score))
    rf_time = time.perf_counter() - t0

    # --- Compare results ---
    identical = 0
    for i, (query, original) in enumerate(queries):
        sm_match, sm_score = sm_results[i]
        rf_match, rf_score = rf_results[i]
        same = sm_match == rf_match
        if same:
            identical += 1

        sm_display = f"{sm_match} ({sm_score:.2f})" if sm_match else "(none)"
        rf_display = f"{rf_match} ({rf_score:.2f})" if rf_match else "(none)"
        marker = "OK" if same else "DIFF"

        print(f"{query:<30} {original:<25} {sm_display:<25} {rf_display:<25} {marker}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Attack names in index:   {len(all_attacks)}")
    print(f"Queries tested:          {len(queries)}")
    print()
    print(f"SequenceMatcher time:    {sm_time:.4f}s  ({sm_time / len(queries) * 1000:.2f} ms/query)")
    print(f"rapidfuzz time:          {rf_time:.4f}s  ({rf_time / len(queries) * 1000:.2f} ms/query)")
    print()
    speedup = sm_time / rf_time if rf_time > 0 else float("inf")
    print(f"Speedup:                 {speedup:.1f}x")
    print(f"Identical matches:       {identical}/{len(queries)} ({100 * identical / len(queries):.1f}%)")

    # Show any differences in detail
    diffs = [
        (i, queries[i], sm_results[i], rf_results[i])
        for i in range(len(queries))
        if sm_results[i][0] != rf_results[i][0]
    ]
    if diffs:
        print(f"\n--- {len(diffs)} differences ---")
        for i, (query, original), (sm_m, sm_s), (rf_m, rf_s) in diffs:
            print(f"  Query: {query!r} (from {original!r})")
            print(f"    SM: {sm_m!r} ({sm_s:.3f})  |  RF: {rf_m!r} ({rf_s:.3f})")


if __name__ == "__main__":
    main()
