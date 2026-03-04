"""Card identification using Claude Code CLI vision.

Spawns `claude -p` subprocesses to visually identify Pokemon cards
from segment images.  Strips CLAUDE* env vars so the child process
doesn't detect nesting.

No API key needed — uses the user's Claude Code subscription.

Two modes:
  - Monolithic: single prompt identifies everything (legacy)
  - Multi-step: 5 focused prompts in parallel, each reading one field
"""

import json
import logging
import os
import pickle
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from thefuzz import fuzz

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Prompt optimized for card identification from binder segment photos
_IDENTIFY_PROMPT = (
    "Read the image at {image_path} and identify this Pokemon trading card. "
    "These are cropped segments from phone photos of binder pages — they may "
    "be dark, blurry, have glare, or be partially cropped. "
    "IMPORTANT: Look carefully at the bottom-right corner for the collector "
    "number (e.g. 90/144). Also note the set symbol next to it. "
    "Look at the top for the card name and HP. "
    "Return ONLY valid JSON (no markdown fences, no extra text): "
    '{{"pokemon_name": "name as printed on card", '
    '"card_name": "full name including suffixes like ex, delta, LV.X etc", '
    '"set_name": "set name if identifiable from symbol or text", '
    '"card_number": "collector number like 90/144 from bottom right", '
    '"hp": null, '
    '"attacks": ["attack1 name", "attack2 name"], '
    '"era": "one of: Base/Jungle/Fossil, Neo, e-card, EX, DP, Platinum, '
    'HGSS, BW, XY, SM, SWSH, SV, or unknown", '
    '"confidence": 0.0}}'
)

# Clean env: strip all CLAUDE* vars to avoid nested-session detection
_CLEAN_ENV = None


def _get_clean_env():
    global _CLEAN_ENV
    if _CLEAN_ENV is None:
        _CLEAN_ENV = {
            k: v for k, v in os.environ.items()
            if 'CLAUDE' not in k.upper()
        }
        _CLEAN_ENV['TERM'] = 'dumb'
    return _CLEAN_ENV


def _parse_claude_json(text: str) -> dict | None:
    """Extract JSON from Claude's response, handling markdown fences."""
    if not text:
        return None
    text = text.strip()
    # Strip ```json ... ``` wrappers
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines[1:] if l.strip() != "```"]
        text = "\n".join(lines).strip()
    # Find first { ... } block
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        logger.warning("Failed to parse Claude JSON: %s", text[start:end][:200])
        return None


def identify_card_vision(
    image_path: str | Path,
    model: str = "sonnet",
    timeout_s: int = 90,
) -> dict | None:
    """Identify a Pokemon card by having Claude visually examine the image.

    Spawns a claude CLI subprocess with CLAUDE* env vars stripped.

    Returns parsed identification dict, or None on failure.
    """
    image_path = str(Path(image_path).resolve())
    if not Path(image_path).exists():
        logger.error("Image not found: %s", image_path)
        return None

    prompt = _IDENTIFY_PROMPT.format(image_path=image_path)

    cmd = [
        'claude', '-p', prompt,
        '--allowedTools', 'Read',
        '--dangerously-skip-permissions',
        '--no-session-persistence',
        '--model', model,
    ]

    try:
        t0 = time.time()
        proc = subprocess.run(
            cmd,
            env=_get_clean_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed = time.time() - t0

        output = proc.stdout.strip()
        if not output:
            logger.warning("Claude vision: no output for %s (rc=%d, %.1fs)",
                           Path(image_path).name, proc.returncode, elapsed)
            if proc.stderr:
                logger.debug("Claude vision stderr: %s", proc.stderr[:300])
            return None

        result = _parse_claude_json(output)
        if result:
            result["_source"] = "claude_vision"
            result["_model"] = model
            logger.info("Claude vision: %s (conf=%.2f) from %s in %.1fs",
                        result.get("pokemon_name"), result.get("confidence", 0),
                        Path(image_path).name, elapsed)
        else:
            logger.warning("Claude vision: unparseable response from %s: %s",
                           Path(image_path).name, output[:200])
        return result

    except subprocess.TimeoutExpired:
        logger.warning("Claude vision: timed out after %ds for %s",
                       timeout_s, Path(image_path).name)
        return None
    except FileNotFoundError:
        logger.error("Claude vision: 'claude' CLI not found in PATH")
        return None
    except Exception as e:
        logger.error("Claude vision: unexpected error for %s: %s",
                     Path(image_path).name, e)
        return None


def identify_cards_vision_parallel(
    image_paths: list[str | Path],
    model: str = "sonnet",
    timeout_s: int = 90,
    max_workers: int = 4,
) -> list[dict | None]:
    """Identify multiple cards in parallel using Claude vision.

    Returns list of results in same order as input paths.
    """
    results = [None] * len(image_paths)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {}
        for i, path in enumerate(image_paths):
            f = pool.submit(identify_card_vision, path, model, timeout_s)
            future_to_idx[f] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error("Claude vision failed for %s: %s",
                             image_paths[idx], e)
                results[idx] = None

    identified = sum(1 for r in results if r is not None)
    logger.info("Claude vision parallel: %d/%d identified",
                identified, len(image_paths))
    return results


def match_vision_to_db(
    vision_result: dict,
    session=None,
) -> tuple[str | None, float]:
    """Match a Claude vision result to the card database.

    Uses pokemon_name + card_number + set/era + HP for matching.
    Returns (card_id, confidence) or (None, 0.0).
    """
    if not vision_result:
        return None, 0.0

    from cardprice.db.session import SessionLocal
    from sqlalchemy import text as sa_text

    own_session = session is None
    if own_session:
        session = SessionLocal()

    try:
        pokemon_name = vision_result.get("pokemon_name") or ""
        card_name = vision_result.get("card_name") or pokemon_name
        card_number = vision_result.get("card_number")
        era = vision_result.get("era")
        hp = vision_result.get("hp")

        if not pokemon_name or len(pokemon_name) < 2:
            return None, 0.0

        # Normalize card number: "41/115" -> "41"
        number_only = None
        if card_number:
            number_only = card_number.split("/")[0].strip().lstrip("0") or None

        # Strategy 1: exact name + number match
        if number_only:
            rows = session.execute(
                sa_text("""
                    SELECT c.card_id, c.name, s.name as set_name
                    FROM dim_cards c
                    JOIN dim_sets s ON c.set_id = s.set_id
                    WHERE LOWER(c.name) = LOWER(:name)
                      AND LTRIM(c.card_number, '0') = :number
                """),
                {"name": pokemon_name, "number": number_only},
            ).fetchall()
            if len(rows) == 1:
                return rows[0][0], 0.95
            elif rows:
                # Multiple matches — use era/set to disambiguate
                for row in rows:
                    if era and era.lower() in row[2].lower():
                        return row[0], 0.90
                # Return first match with lower confidence
                return rows[0][0], 0.75

        # Strategy 2: get all candidates matching the name
        rows = session.execute(
            sa_text("""
                SELECT c.card_id, c.name, c.card_number, c.hp,
                       s.name as set_name, s.release_date
                FROM dim_cards c
                JOIN dim_sets s ON c.set_id = s.set_id
                WHERE LOWER(c.name) LIKE LOWER(:pattern)
                ORDER BY s.release_date
            """),
            {"pattern": f"%{pokemon_name[:20]}%"},
        ).fetchall()

        if not rows:
            # Try with just the pokemon base name (strip ex, delta, etc.)
            base_name = re.sub(
                r'\s*(ex|EX|δ|delta|V|VSTAR|VMAX|GX|LV\.\w+)\s*',
                '', pokemon_name,
            ).strip()
            if base_name and base_name != pokemon_name:
                rows = session.execute(
                    sa_text("""
                        SELECT c.card_id, c.name, c.card_number, c.hp,
                               s.name as set_name, s.release_date
                        FROM dim_cards c
                        JOIN dim_sets s ON c.set_id = s.set_id
                        WHERE LOWER(c.name) LIKE LOWER(:pattern)
                        ORDER BY s.release_date
                    """),
                    {"pattern": f"%{base_name}%"},
                ).fetchall()

        if not rows:
            return None, 0.0

        # Score each candidate
        best_id = None
        best_score = 0.0

        for row in rows:
            cid, name, cnum, card_hp, set_name, release_date = row
            score = 0.0

            # Fuzzy name match (most important signal)
            name_score = fuzz.ratio(card_name.lower(), name.lower()) / 100.0
            score += name_score * 0.5

            # Card number match
            if number_only and cnum:
                cnum_norm = cnum.lstrip("0") or "0"
                if cnum_norm == number_only:
                    score += 0.25

            # HP match
            if hp and card_hp:
                try:
                    if int(hp) == int(card_hp):
                        score += 0.15
                except (ValueError, TypeError):
                    pass

            # Era hint
            if era and era.lower() != "unknown":
                _ERA_KEYWORDS = {
                    "ecard": ["expedition", "aquapolis", "skyridge"],
                    "ex": ["ex "],
                    "dp": ["diamond", "pearl", "mysterious",
                           "secret wonders", "great encounters",
                           "majestic dawn", "legends awakened",
                           "stormfront"],
                    "platinum": ["platinum", "rising rivals",
                                 "supreme victors", "arceus"],
                    "bw": ["black", "white", "noble victories",
                           "next destinies", "dark explorers",
                           "dragons exalted", "boundaries", "plasma"],
                    "xy": ["xy", "flashfire", "furious", "phantom",
                           "primal", "roaring", "ancient", "breakthrough",
                           "breakpoint", "fates", "steam", "evolutions"],
                    "sm": ["sun", "moon", "guardians", "burning",
                           "shining", "crimson", "ultra", "forbidden",
                           "celestial", "dragon majesty", "lost thunder",
                           "team up", "unbroken", "unified", "cosmic"],
                    "swsh": ["sword", "shield", "rebel", "darkness",
                             "champion", "vivid", "battle styles",
                             "chilling", "evolving", "fusion",
                             "brilliant", "astral", "lost origin",
                             "silver", "crown"],
                }
                era_key = era.lower().replace("-", "").replace(" ", "")
                for ek, keywords in _ERA_KEYWORDS.items():
                    if ek == era_key:
                        if any(kw in set_name.lower() for kw in keywords):
                            score += 0.10
                        break

            if score > best_score:
                best_score = score
                best_id = cid

        if best_id and best_score >= 0.35:
            return best_id, min(best_score, 0.99)

        return None, 0.0

    finally:
        if own_session:
            session.close()


# ---------------------------------------------------------------------------
# Multi-step focused prompts
# ---------------------------------------------------------------------------

_MULTI_PROMPTS = {
    "name": (
        "Read the image at {image_path}. "
        "What Pokemon is shown on this card? Include suffixes like ex, V, VSTAR, "
        "VMAX, GX, delta, LV.X if present. "
        'Return ONLY: {{"name": "...", "confidence": 0.0}}'
    ),
    "attacks": (
        "Read the image at {image_path}. "
        "List the attack or move names printed on this Pokemon card. "
        "Ignore ability names — only attacks with damage numbers or energy costs. "
        'Return ONLY: {{"attacks": ["attack1", "attack2"], "confidence": 0.0}}'
    ),
    "number": (
        "Read the image at {image_path}. "
        "Read the collector number in the bottom-left or bottom-right corner of "
        "this Pokemon card. It looks like XX/YYY or XXX. "
        'Return ONLY: {{"number": "XX/YYY", "confidence": 0.0}}'
    ),
    "era": (
        "Read the image at {image_path}. "
        "Classify this Pokemon card's era by its border style and layout. "
        "Choose exactly one: Base/Jungle/Fossil, Neo, e-card, EX-era, DP, "
        "Platinum, HGSS, BW, XY, SM, SWSH, SV, or unknown. "
        'Return ONLY: {{"era": "...", "confidence": 0.0}}'
    ),
    "hp": (
        "Read the image at {image_path}. "
        "What is the HP number shown near the top-right of this Pokemon card? "
        'Return ONLY: {{"hp": 120, "confidence": 0.0}}'
    ),
}


def _run_vision_sub(key: str, image_path: str, model: str, timeout_s: int):
    """Run a single focused vision sub-prompt."""
    prompt = _MULTI_PROMPTS[key].format(image_path=image_path)
    cmd = [
        'claude', '-p', prompt,
        '--allowedTools', 'Read',
        '--dangerously-skip-permissions',
        '--no-session-persistence',
        '--model', model,
    ]
    try:
        t0 = time.time()
        proc = subprocess.run(
            cmd,
            env=_get_clean_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed = time.time() - t0
        result = _parse_claude_json(proc.stdout.strip())
        logger.debug("Vision sub[%s]: %s (%.1fs)", key, result, elapsed)
        return key, result
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning("Vision sub[%s] failed: %s", key, e)
        return key, None


def identify_card_multi_step(
    image_path: str | Path,
    model: str = "sonnet",
    timeout_s: int = 45,
    max_workers: int = 5,
) -> dict | None:
    """Identify a card using 5 parallel focused vision prompts.

    Runs name, attacks, number, era, HP prompts concurrently.
    Combines results into a single dict compatible with match_vision_to_db().
    """
    image_path = str(Path(image_path).resolve())
    if not Path(image_path).exists():
        logger.error("Image not found: %s", image_path)
        return None

    sub_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_run_vision_sub, k, image_path, model, timeout_s)
            for k in _MULTI_PROMPTS
        ]
        for f in as_completed(futures):
            key, result = f.result()
            sub_results[key] = result

    return _combine_sub_results(sub_results, model)


def _combine_sub_results(sub_results: dict, model: str) -> dict | None:
    """Merge 5 sub-prompt results into one identification dict."""
    CONFIDENCE_FLOOR = 0.3

    name_result = sub_results.get("name")
    if not name_result or not name_result.get("name"):
        return None

    def _get(key, field, default=None):
        r = sub_results.get(key)
        if not r:
            return default, 0.0
        conf = r.get("confidence", 0.0)
        if conf < CONFIDENCE_FLOOR:
            return default, conf
        return r.get(field, default), conf

    pokemon_name, name_conf = _get("name", "name")
    attacks, atk_conf = _get("attacks", "attacks", [])
    card_number, num_conf = _get("number", "number")
    era, era_conf = _get("era", "era", "unknown")
    hp, hp_conf = _get("hp", "hp")

    sub_confidences = {
        "name": name_conf,
        "attacks": atk_conf,
        "number": num_conf,
        "era": era_conf,
        "hp": hp_conf,
    }

    # Weighted aggregate
    weights = {"name": 0.35, "attacks": 0.25, "number": 0.20, "era": 0.10, "hp": 0.10}
    aggregate = sum(sub_confidences[k] * weights[k] for k in weights)

    return {
        "pokemon_name": pokemon_name,
        "card_name": pokemon_name,
        "attacks": attacks if isinstance(attacks, list) else [],
        "card_number": card_number,
        "set_name": None,
        "era": era,
        "hp": hp,
        "confidence": round(aggregate, 3),
        "_sub_confidences": sub_confidences,
        "_source": "claude_vision_multi",
        "_model": model,
    }


def identify_cards_multi_step_parallel(
    image_paths: list[str | Path],
    model: str = "sonnet",
    timeout_s: int = 45,
    max_workers: int = 4,
) -> list[dict | None]:
    """Identify multiple cards using multi-step vision, cards processed in parallel.

    Each card runs 5 sub-prompts internally. Cards are processed max_workers at a time.
    """
    results = [None] * len(image_paths)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {}
        for i, path in enumerate(image_paths):
            f = pool.submit(identify_card_multi_step, path, model, timeout_s)
            future_to_idx[f] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error("Multi-step vision failed for %s: %s",
                             image_paths[idx], e)

    identified = sum(1 for r in results if r is not None)
    logger.info("Multi-step vision: %d/%d identified", identified, len(image_paths))
    return results


# ---------------------------------------------------------------------------
# Attack-based matching
# ---------------------------------------------------------------------------

_attack_index = None


def _get_attack_index():
    """Lazy-load the attack index from pickle."""
    global _attack_index
    if _attack_index is None:
        idx_path = _PROJECT_ROOT / "data" / "attack_index.pkl"
        if idx_path.exists():
            with open(idx_path, "rb") as f:
                _attack_index = pickle.load(f)
            logger.info("Loaded attack index: %d attacks, %d cards",
                        len(_attack_index.get("attack_to_cards", {})),
                        len(_attack_index.get("card_to_attacks", {})))
        else:
            logger.warning("Attack index not found at %s", idx_path)
            _attack_index = {"attack_to_cards": {}, "card_to_attacks": {}}
    return _attack_index


def match_attacks_to_db(
    attacks: list[str],
    pokemon_name: str | None = None,
    hp: int | str | None = None,
    card_number: str | None = None,
    session=None,
) -> tuple[str | None, float]:
    """Match a card by its attack names, optionally filtered by name/HP/number.

    Intersects the card_id sets for each attack name. Combined with
    pokemon_name, this is often a unique match.

    Returns (card_id, confidence) or (None, 0.0).
    """
    if not attacks:
        return None, 0.0

    idx = _get_attack_index()
    atk_to_cards = idx.get("attack_to_cards", {})

    # Find cards that have ALL the given attacks
    candidate_sets = []
    for atk in attacks:
        atk_lower = atk.strip().lower()
        if atk_lower in atk_to_cards:
            candidate_sets.append(set(atk_to_cards[atk_lower]))

    if not candidate_sets:
        return None, 0.0

    # Intersect all attack sets
    candidates = candidate_sets[0]
    for s in candidate_sets[1:]:
        candidates = candidates & s

    if not candidates:
        # Fall back to union with scoring
        candidates = set()
        for s in candidate_sets:
            candidates |= s

    if not candidates:
        return None, 0.0

    # Filter by pokemon name if provided
    if pokemon_name and len(pokemon_name) >= 2:
        from cardprice.db.session import SessionLocal
        from sqlalchemy import text as sa_text

        own_session = session is None
        if own_session:
            session = SessionLocal()

        try:
            name_lower = pokemon_name.lower()
            base_name = re.sub(
                r'\s*(ex|EX|δ|delta|V|VSTAR|VMAX|GX|LV\.\w+|Star)\s*$',
                '', pokemon_name,
            ).strip().lower()

            # Get names for candidates from DB
            placeholders = ", ".join(f":c{i}" for i in range(len(candidates)))
            cid_list = list(candidates)
            params = {f"c{i}": cid for i, cid in enumerate(cid_list)}

            rows = session.execute(
                sa_text(f"""
                    SELECT card_id, name, card_number, hp
                    FROM dim_cards
                    WHERE card_id IN ({placeholders})
                """),
                params,
            ).fetchall()

            # Score candidates
            best_id = None
            best_score = 0.0

            for row in rows:
                cid, db_name, db_num, db_hp = row
                score = 0.0
                db_name_lower = (db_name or "").lower()
                db_base = re.sub(
                    r'\s*(ex|EX|δ|delta|V|VSTAR|VMAX|GX|LV\.\w+|Star)\s*$',
                    '', db_name or '',
                ).strip().lower()

                # Name match
                if db_base == base_name or name_lower in db_name_lower:
                    score += 0.50
                elif fuzz.ratio(name_lower, db_name_lower) > 80:
                    score += 0.35

                # Attack intersection score
                card_attacks = set(idx.get("card_to_attacks", {}).get(cid, []))
                given_attacks = {a.strip().lower() for a in attacks}
                if card_attacks and given_attacks:
                    overlap = len(card_attacks & given_attacks)
                    score += 0.30 * (overlap / max(len(given_attacks), 1))

                # Card number match
                if card_number and db_num:
                    num_only = card_number.split("/")[0].strip().lstrip("0") or "0"
                    db_num_norm = db_num.lstrip("0") or "0"
                    if num_only == db_num_norm:
                        score += 0.15

                # HP match
                if hp and db_hp:
                    try:
                        if int(hp) == int(db_hp):
                            score += 0.05
                    except (ValueError, TypeError):
                        pass

                if score > best_score:
                    best_score = score
                    best_id = cid

            if best_id and best_score >= 0.40:
                return best_id, min(best_score, 0.99)
            return None, 0.0

        finally:
            if own_session:
                session.close()

    # No name filter — just return the best from intersection
    if len(candidates) == 1:
        return list(candidates)[0], 0.85
    return list(candidates)[0], 0.50


def match_multi_step_to_db(
    vision_result: dict,
    session=None,
) -> tuple[str | None, float]:
    """Match a multi-step vision result using all available signals.

    Tries attack-based matching first (strongest signal), then falls back
    to name+number matching via match_vision_to_db().
    """
    if not vision_result:
        return None, 0.0

    attacks = vision_result.get("attacks", [])
    name = vision_result.get("pokemon_name")
    hp = vision_result.get("hp")
    number = vision_result.get("card_number")

    # Try attack-based matching first
    if attacks and len(attacks) >= 1:
        card_id, conf = match_attacks_to_db(
            attacks, pokemon_name=name, hp=hp,
            card_number=number, session=session,
        )
        if card_id and conf >= 0.60:
            logger.info("Attack match: %s (conf=%.2f) for %s [%s]",
                        card_id, conf, name, ", ".join(attacks))
            return card_id, conf

    # Fall back to name+number matching
    return match_vision_to_db(vision_result, session=session)
