#!/usr/bin/env python3
"""Run pipeline on all ground truth pages and output a detailed table."""

import json
import os
import sys
import time
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.json")
INBOX_DIR = os.path.join(DATA_DIR, "inbox")


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


def get_name_from_result(result, card_names_lookup):
    if not result:
        return None
    raw = result.get("raw_response", {})
    signals = raw.get("signals", {})
    name_used = signals.get("name_used")
    if name_used:
        return name_used
    card_id = result.get("card_id")
    if card_id:
        if card_id in card_names_lookup:
            return card_names_lookup[card_id]
        base_id = card_id.split("/")[0] if "/" in card_id else card_id
        for cid, name in card_names_lookup.items():
            if cid.startswith(base_id + "/") or cid == base_id:
                return name
    return card_id


def main():
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    pages = gt["pages"]
    card_names_lookup = load_card_names_lookup()

    # Sort pages by their numeric timestamp
    sorted_pages = sorted(pages.items(), key=lambda x: x[0])

    print("Importing ML pipeline...")
    t0 = time.time()
    from cardprice.ml import identify_page_v2
    print(f"Import done in {time.time() - t0:.1f}s\n")

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    engine = create_engine("postgresql+psycopg2://godli@/cardprice")

    all_rows = []
    total_correct = 0
    total_cards = 0

    for page_name, page_gt in sorted_pages:
        page_dir = os.path.join(INBOX_DIR, page_name)
        if not os.path.isdir(page_dir):
            print(f"SKIP {page_name}")
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
        print(f"Processing {page_name} ({len(card_paths)} cards) - {desc}")

        t_start = time.time()
        with Session(engine) as session:
            results = identify_page_v2(card_paths, session=session)
        elapsed = time.time() - t_start
        print(f"  {elapsed:.1f}s ({elapsed/len(card_paths):.1f}s/card)")

        for i, (result, (gt_idx, gt_card)) in enumerate(zip(results, gt_cards.items())):
            expected = gt_card["name"]
            predicted = get_name_from_result(result, card_names_lookup)
            card_id = result.get("card_id", "") if result else ""
            confidence = result.get("confidence", 0) if result else 0
            method = result.get("method", "") if result else ""

            raw = result.get("raw_response", {}) if result else {}

            # Extract intermediate values
            ocr_name = raw.get("ocr_name") or raw.get("signals", {}).get("ocr_name") or ""
            ocr_conf = raw.get("ocr_confidence") or raw.get("signals", {}).get("ocr_confidence") or 0
            ocr_raw_text = raw.get("ocr_raw") or ""
            hp = raw.get("hp") or raw.get("signals", {}).get("hp") or ""
            color_type = raw.get("color_type") or ""
            n_cands = raw.get("n_candidates") or ""
            attack_names = raw.get("attack_names") or ""

            # Combined results (top candidates with scores)
            combined = raw.get("combined_results", [])
            top1_score = ""
            top1_detail = ""
            if combined:
                top1_cid, top1_s, top1_d = combined[0]
                top1_score = f"{top1_s:.4f}"
                if isinstance(top1_d, dict):
                    top1_detail = f"dino={top1_d.get('dino_score',0):.3f} atk={top1_d.get('attack_score',0):.3f}"

            # DINOv2 top from signals
            dino_top = raw.get("dino_top10", raw.get("signals", {}).get("dino_name_vote", ""))

            correct = fuzzy_name_match(predicted, expected)
            if correct:
                total_correct += 1
            total_cards += 1

            status = "OK" if correct else ("WRONG" if card_id else "MISS")

            # Truncate ocr_raw for display
            ocr_raw_short = str(ocr_raw_text)[:80] if ocr_raw_text else ""

            all_rows.append({
                "page": page_name.replace("page_", "").replace("_cards", ""),
                "slot": f"card_{i:02d}",
                "expected": expected,
                "predicted": predicted or "",
                "status": status,
                "ocr_name": ocr_name,
                "ocr_conf": f"{ocr_conf:.2f}" if ocr_conf else "",
                "ocr_raw": ocr_raw_short,
                "hp": str(hp),
                "color": color_type,
                "n_cands": str(n_cands),
                "attacks": str(attack_names)[:40] if attack_names else "",
                "top1_score": top1_score,
                "top1_detail": top1_detail,
                "card_id": card_id or "",
                "conf": f"{confidence:.3f}" if confidence else "",
                "method": method,
            })

    # Print summary
    print(f"\n{'='*70}")
    print(f"OVERALL: {total_correct}/{total_cards} ({100*total_correct/total_cards:.1f}%)")
    wrong = [r for r in all_rows if r["status"] == "WRONG"]
    miss = [r for r in all_rows if r["status"] == "MISS"]
    print(f"WRONG: {len(wrong)}, MISS: {len(miss)}")
    print(f"{'='*70}\n")

    # Print table
    # Header
    print(f"{'Page':<20} {'Slot':<8} {'Expected':<18} {'Predicted':<18} {'St':<5} "
          f"{'OCR Name':<16} {'OConf':<6} {'HP':<5} {'Color':<8} {'#Cand':<6} "
          f"{'Attacks':<30} {'Top1':<8} {'Top1 Detail':<22} "
          f"{'Card ID':<20} {'Conf':<6} {'Method':<16} "
          f"{'OCR Raw (first 80 chars)'}")
    print("-" * 260)

    cur_page = ""
    for r in all_rows:
        if r["page"] != cur_page:
            if cur_page:
                print("-" * 260)
            cur_page = r["page"]

        status_mark = r["status"]
        if status_mark == "OK":
            st = "  OK"
        elif status_mark == "WRONG":
            st = "WRONG"
        else:
            st = " MISS"

        print(f"{r['page']:<20} {r['slot']:<8} {r['expected']:<18} {r['predicted']:<18} {st:<5} "
              f"{r['ocr_name']:<16} {r['ocr_conf']:<6} {r['hp']:<5} {r['color']:<8} {r['n_cands']:<6} "
              f"{r['attacks']:<30} {r['top1_score']:<8} {r['top1_detail']:<22} "
              f"{r['card_id']:<20} {r['conf']:<6} {r['method']:<16} "
              f"{r['ocr_raw']}")

    print(f"\n{'='*70}")
    print(f"OVERALL: {total_correct}/{total_cards} ({100*total_correct/total_cards:.1f}%)")
    print(f"WRONG: {len(wrong)}, MISS: {len(miss)}")

    if wrong:
        print(f"\nWRONG ANSWERS:")
        for r in wrong:
            print(f"  {r['page']} {r['slot']}: expected={r['expected']!r} got={r['predicted']!r} ({r['card_id']}, {r['method']})")


if __name__ == "__main__":
    main()
