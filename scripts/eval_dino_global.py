#!/usr/bin/env python3
"""Test: if OCR fails, how well does raw DINOv2 over all 20k cards do?

For each eval card:
  1. If OCR name is strong (conf >= 0.85): use name path (current behavior)
  2. Else: attack OCR fallback — read attack names, find matching cards, DINOv2 pick
  3. Else: DINOv2 dot product against ALL 20,026 reference embeddings, pick top-1
"""

import json
import os
import sys
import time
import pickle
import numpy as np
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.json")
INBOX_DIR = os.path.join(DATA_DIR, "inbox")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "ref_embeddings.pkl")


def fuzzy_name_match(predicted, expected):
    if not predicted or not expected:
        return False
    p = predicted.lower().strip()
    e = expected.lower().strip()
    if p == e:
        return True
    if p in e or e in p:
        return True
    return SequenceMatcher(None, p, e).ratio() >= 0.75


def load_card_names_lookup():
    lookup = {}
    cn_path = os.path.join(DATA_DIR, "card_names.json")
    if os.path.exists(cn_path):
        with open(cn_path) as f:
            for row in json.load(f):
                lookup[row[0]] = row[1]
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine("postgresql+psycopg2://godli@/cardprice")
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT card_id, name FROM dim_cards")).fetchall()
            for r in rows:
                lookup[r[0]] = r[1]
    except Exception:
        pass
    return lookup


def _lookup_name(card_id, lookup):
    """Look up card name from card_id, trying full ID then base ID."""
    if card_id in lookup:
        return lookup[card_id]
    base = card_id.split("/")[0] if "/" in card_id else card_id
    if base in lookup:
        return lookup[base]
    # Search prefix
    for cid, name in lookup.items():
        if cid.startswith(base + "/") or cid.startswith(base + "-"):
            return name
    return card_id


def main():
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)
    pages = gt["pages"]
    sorted_pages = sorted(pages.items(), key=lambda x: x[0])
    card_names_lookup = load_card_names_lookup()

    # Load reference embeddings
    print("Loading reference embeddings...")
    with open(EMBEDDINGS_PATH, "rb") as f:
        ref_embs = pickle.load(f)
    ref_ids = list(ref_embs.keys())
    ref_matrix = np.array([ref_embs[k] for k in ref_ids], dtype=np.float32)
    # Normalize
    norms = np.linalg.norm(ref_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    ref_matrix = ref_matrix / norms
    print(f"  {len(ref_ids)} reference embeddings loaded ({ref_matrix.shape})")

    # Import DINOv2 for computing query embeddings
    print("Loading DINOv2 model...")
    t0 = time.time()
    from cardprice.ml.dino_matcher import extract_embedding as get_dino_embedding
    # Warm up
    print(f"  Model ready in {time.time() - t0:.1f}s")

    # Import OCR and DB helpers
    from cardprice.ml import (
        _paddle_ocr_name_and_hp, _run_name_and_hp, _get_candidates_from_db,
    )
    from cardprice.ml.attack_ocr import (
        extract_attack_names_paddle as extract_attacks,
        fuzzy_match_attacks,
        _load_structured_attacks,
    )

    # Load structured attacks for HP filtering
    print("Loading structured attacks...")
    struct_atk_to_cards, struct_card_to_atks = _load_structured_attacks()
    # Build HP lookup from structured_attacks.json
    struct_attacks_path = os.path.join(DATA_DIR, "structured_attacks.json")
    card_hp_lookup = {}
    if os.path.exists(struct_attacks_path):
        with open(struct_attacks_path) as f:
            struct_data = json.load(f)
        for cid, entry in struct_data.items():
            hp_str = entry.get("hp", "")
            if hp_str and hp_str.isdigit():
                card_hp_lookup[cid] = int(hp_str)
                card_hp_lookup[f"{cid}/normal"] = int(hp_str)
    print(f"  HP lookup: {len(card_hp_lookup)} cards")

    # Load multilingual card translations (name -> card IDs)
    translations_path = os.path.join(DATA_DIR, "card_translations.json")
    trans_name_to_ids: dict[str, list[str]] = {}
    if os.path.exists(translations_path):
        with open(translations_path) as f:
            trans_data = json.load(f)
        for lang, cards in trans_data.items():
            for cid, tname in cards.items():
                lower = tname.lower().strip()
                if lower not in trans_name_to_ids:
                    trans_name_to_ids[lower] = []
                trans_name_to_ids[lower].append(cid)
        print(f"  Translations: {sum(len(v) for v in trans_data.values())} names across {len(trans_data)} languages, {len(trans_name_to_ids)} unique names")
    else:
        print("  No translations file found")

    # Also need DB for name-path candidates
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    engine = create_engine("postgresql+psycopg2://godli@/cardprice")

    total_correct = 0
    total_cards = 0
    results_by_method = {
        "name_path": 0, "attack_path": 0, "dino_global": 0,
        "name_correct": 0, "attack_correct": 0, "dino_correct": 0,
    }

    all_rows = []

    for page_name, page_gt in sorted_pages:
        page_dir = os.path.join(INBOX_DIR, page_name)
        if not os.path.isdir(page_dir):
            continue

        card_paths = []
        gt_cards = {}
        for key, val in page_gt.items():
            if key.startswith("card_") and isinstance(val, dict):
                img_path = os.path.join(page_dir, f"{key}.png")
                if os.path.exists(img_path):
                    card_paths.append(img_path)
                    gt_cards[len(card_paths) - 1] = val

        if not card_paths:
            continue

        desc = page_gt.get("description", "")
        print(f"\n{'='*70}")
        print(f"{page_name} ({len(card_paths)} cards) - {desc}")
        print("="*70)

        for i, (gt_idx, gt_card) in enumerate(gt_cards.items()):
            img_path = card_paths[gt_idx]
            expected = gt_card["name"]
            if gt_card.get("empty_slot"):
                continue
            total_cards += 1

            # Step 1: Try OCR name
            try:
                ocr_name, ocr_conf, ocr_raw, hp_value = _run_name_and_hp(img_path)
            except Exception:
                ocr_name, ocr_conf, ocr_raw, hp_value = None, 0.0, None, None

            if ocr_name and len(ocr_name) < 3:
                ocr_name = None
                ocr_conf = 0.0

            # Step 2: Decide path
            method = ""
            predicted_name = ""
            predicted_id = ""
            dino_score = 0.0
            top5_str = ""

            if ocr_name and ocr_conf >= 0.85:
                # NAME PATH: use OCR name to get candidates, DINOv2 to pick
                method = "name_path"
                results_by_method["name_path"] += 1

                # Get candidates from DB
                with Session(engine) as session:
                    candidates = _get_candidates_from_db(
                        ocr_name, hp=hp_value, session=session
                    )

                # If no English candidates, try multilingual translations
                if not candidates:
                    trans_cands = trans_name_to_ids.get(ocr_name.lower().strip(), [])
                    if trans_cands:
                        candidates = trans_cands

                if candidates and len(candidates) == 1:
                    predicted_id = candidates[0]
                    predicted_name = _lookup_name(predicted_id, card_names_lookup)
                    dino_score = 1.0  # single candidate, trust it
                elif candidates:
                    # DINOv2 pick among candidates
                    query_emb = get_dino_embedding(img_path)
                    if query_emb is not None:
                        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)
                        best_score = -1
                        best_cid = None
                        for cid in candidates:
                            ref_e = ref_embs.get(cid)
                            if ref_e is None:
                                base_cid = cid.split("/")[0] if "/" in cid else cid
                                ref_e = ref_embs.get(base_cid)
                            if ref_e is not None:
                                ref_e = ref_e / (np.linalg.norm(ref_e) + 1e-9)
                                s = float(np.dot(query_emb, ref_e))
                                if s > best_score:
                                    best_score = s
                                    best_cid = cid
                        if best_cid:
                            predicted_id = best_cid
                            predicted_name = _lookup_name(best_cid, card_names_lookup)
                            dino_score = best_score
                else:
                    # No candidates from DB or translations, fall through
                    method = "dino_global"
                    results_by_method["dino_global"] += 1
                    results_by_method["name_path"] -= 1

            if method != "name_path":
                # ATTACK PATH: try attack OCR -> candidate cards -> DINOv2 pick
                attack_tried = False
                attack_candidates_str = ""
                attack_names_found = []

                try:
                    # Step 1: Extract attack names from card image
                    ocr_attacks = extract_attacks(img_path)
                    if ocr_attacks:
                        # Step 2: Fuzzy-match against all known attacks
                        matched = fuzzy_match_attacks(ocr_attacks)
                        attack_names_found = [atk for _, atk, _ in matched]

                        if matched:
                            attack_tried = True
                            # Step 3: Find candidate cards that have ALL matched attacks
                            # Intersect per-attack candidate sets; fall back to union if
                            # intersection is empty (OCR may have misread one attack)
                            per_attack_sets = []
                            for _, atk_name, _ in matched:
                                per_attack_sets.append(
                                    set(struct_atk_to_cards.get(atk_name, []))
                                )
                            atk_candidate_ids = per_attack_sets[0]
                            for s in per_attack_sets[1:]:
                                intersected = atk_candidate_ids & s
                                if intersected:
                                    atk_candidate_ids = intersected
                            # If intersection is empty somehow, use union
                            if not atk_candidate_ids:
                                atk_candidate_ids = set()
                                for s in per_attack_sets:
                                    atk_candidate_ids.update(s)

                            # HP filtering: if we have HP, remove candidates with wrong HP
                            if hp_value and atk_candidate_ids:
                                hp_filtered = set()
                                for cid in atk_candidate_ids:
                                    card_hp = card_hp_lookup.get(cid)
                                    if card_hp is None:
                                        hp_filtered.add(cid)  # keep unknowns
                                    elif card_hp == hp_value:
                                        hp_filtered.add(cid)
                                if hp_filtered:
                                    atk_candidate_ids = hp_filtered

                            # If OCR read a name (even low conf), filter attack
                            # candidates to those matching the OCR name
                            if ocr_name and ocr_conf >= 0.70 and atk_candidate_ids:
                                name_filtered = set()
                                ocr_lower = ocr_name.lower().strip()
                                for cid in atk_candidate_ids:
                                    cname = _lookup_name(cid, card_names_lookup).lower()
                                    if not cname:
                                        continue
                                    if ocr_lower in cname or cname in ocr_lower:
                                        name_filtered.add(cid)
                                if name_filtered:
                                    atk_candidate_ids = name_filtered
                                else:
                                    # Attack candidates don't match OCR name;
                                    # add name-path candidates so DINOv2 can pick
                                    try:
                                        with Session(engine) as session:
                                            name_cands = _get_candidates_from_db(
                                                ocr_name, hp=hp_value, session=session
                                            )
                                        if name_cands:
                                            atk_candidate_ids.update(name_cands)
                                    except Exception:
                                        pass

                            attack_candidates_str = f"{len(atk_candidate_ids)} candidates"

                            # Step 4: DINOv2 pick among attack candidates
                            if atk_candidate_ids:
                                query_emb = get_dino_embedding(img_path)
                                if query_emb is not None:
                                    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)
                                    best_score = -1
                                    best_cid = None
                                    for cid in atk_candidate_ids:
                                        ref_e = ref_embs.get(cid)
                                        if ref_e is None:
                                            base_cid = cid.split("/")[0] if "/" in cid else cid
                                            ref_e = ref_embs.get(base_cid)
                                        if ref_e is not None:
                                            ref_e = ref_e / (np.linalg.norm(ref_e) + 1e-9)
                                            s = float(np.dot(query_emb, ref_e))
                                            if s > best_score:
                                                best_score = s
                                                best_cid = cid
                                    if best_cid:
                                        method = "attack_path"
                                        results_by_method["attack_path"] += 1
                                        predicted_id = best_cid
                                        predicted_name = _lookup_name(best_cid, card_names_lookup)
                                        dino_score = best_score
                except Exception as e:
                    print(f"         attack fallback error: {e}")

                if method != "attack_path":
                    # DINO GLOBAL: dot product against all 20k (last resort)
                    method = "dino_global"
                    results_by_method["dino_global"] += 1

                    query_emb = get_dino_embedding(img_path)
                    if query_emb is not None:
                        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)
                        scores = ref_matrix @ query_emb
                        top_indices = np.argsort(scores)[::-1][:5]

                        best_idx = top_indices[0]
                        predicted_id = ref_ids[best_idx]
                        dino_score = float(scores[best_idx])
                        predicted_name = _lookup_name(predicted_id, card_names_lookup)

                        # Top 5 for display
                        top5_parts = []
                        for idx in top_indices:
                            cid = ref_ids[idx]
                            name = _lookup_name(cid, card_names_lookup)
                            top5_parts.append(f"{name}({scores[idx]:.3f})")
                        top5_str = " | ".join(top5_parts)

            correct = fuzzy_name_match(predicted_name, expected)
            if correct:
                total_correct += 1
                if method == "name_path":
                    results_by_method["name_correct"] += 1
                elif method == "attack_path":
                    results_by_method["attack_correct"] += 1
                else:
                    results_by_method["dino_correct"] += 1

            status = "OK" if correct else ("WRONG" if predicted_name else "MISS")

            ocr_display = f"{ocr_name}({ocr_conf:.2f})" if ocr_name else "(none)"
            hp_display = str(hp_value) if hp_value else ""

            print(f"  card_{i:02d} {status:<5} exp={expected:<18} got={predicted_name:<18} "
                  f"method={method:<12} ocr={ocr_display:<25} hp={hp_display:<5} "
                  f"dino={dino_score:.3f}")
            if method == "attack_path":
                print(f"         attacks found: {attack_names_found}  ({attack_candidates_str})")
            if method == "dino_global" and top5_str:
                print(f"         top5: {top5_str}")
                if attack_names_found:
                    print(f"         attacks found (no match): {attack_names_found}  ({attack_candidates_str})")

    print(f"\n{'='*70}")
    print(f"OVERALL: {total_correct}/{total_cards} ({100*total_correct/total_cards:.1f}%)")
    print(f"  Name path:     {results_by_method['name_correct']}/{results_by_method['name_path']} correct")
    print(f"  Attack path:   {results_by_method['attack_correct']}/{results_by_method['attack_path']} correct")
    print(f"  DINOv2 global: {results_by_method['dino_correct']}/{results_by_method['dino_global']} correct")
    wrong = total_cards - total_correct
    print(f"  Wrong/miss: {wrong}")


if __name__ == "__main__":
    main()
