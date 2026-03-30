#!/usr/bin/env python3
"""Benchmark detect_all_variants timing per card and per check.

Prints results incrementally with flush so we can see progress.
"""

import sys
import time
import cv2

sys.path.insert(0, '/home/godli/cardprice')
import cardprice.ml.stamp_detection as sd


def p(msg):
    print(msg, flush=True)


def timed_detect(image_path, card_id, fast=False):
    """Wraps detect_all_variants with per-check timing, printing as we go."""
    set_id = sd._extract_set_id(card_id)
    era = sd._get_era(card_id)
    variant_suffix = card_id.split("/", 1)[1] if "/" in card_id else ""

    result = {
        "stamps_detected": [],
        "stamp_details": {},
        "stamps_checked": [],
        "variant_flags": {},
        "_check_times": {},
    }

    img = cv2.imread(str(image_path))
    if img is None:
        p(f"  ERROR: could not read {image_path}")
        return result

    def _run(name, checker_fn, *args, **kwargs):
        result["stamps_checked"].append(name)
        t0 = time.perf_counter()
        try:
            detail = checker_fn(*args, **kwargs)
        except Exception as e:
            dt = time.perf_counter() - t0
            result["_check_times"][name] = dt
            p(f"  {name:<25s} {dt*1000:8.1f}ms  ERROR: {e}")
            return None
        dt = time.perf_counter() - t0
        result["_check_times"][name] = dt

        detected = detail.get("detected", False) if detail else False
        marker = " DETECTED" if detected else ""
        p(f"  {name:<25s} {dt*1000:8.1f}ms{marker}")

        if not detected:
            return None

        result["stamps_detected"].append(name)
        info = {
            "confidence": detail["confidence"],
            "position": detail.get("position", "unknown"),
        }
        result["stamp_details"][name] = info
        return detail

    # === TIER 0 ===
    wc = _run("world_championship", sd._check_world_championship, img, set_id)
    if wc:
        result["variant_flags"]["is_reproduction"] = True
        return result

    if set_id == "base1":
        _run("shadowless", sd._check_shadowless, img, set_id)

    if fast:
        return result

    # === TIER 1: WotC stamps ===
    if set_id in sd._FIRST_EDITION_SETS:
        ed = _run("1st_edition", sd._check_1st_edition, img)
        if ed:
            result["variant_flags"]["1st_edition"] = True

    if set_id == "base1":
        _run("ghost_stamp", sd._check_ghost_stamp, img, set_id)
        _run("copyright_year", sd._check_copyright_year, img, set_id)

    if set_id == "base2" and "holo" in variant_suffix.lower():
        _run("no_symbol_error", sd._check_no_symbol_error_as_stamp,
             img, set_id, variant_suffix)

    # === TIER 2: EX era ===
    if set_id in sd._EX_STAMPED_SETS:
        _run("ex_set_stamp", sd._check_ex_set_stamp, img, set_id)

    # === TIER 3: Prerelease/Staff/Winner ===
    prerelease_found = False
    if set_id in sd._PRERELEASE_TEXT_SETS:
        pr = _run("prerelease", sd._check_prerelease, img, set_id, era)
        if pr:
            prerelease_found = True
    elif set_id in sd._PRERELEASE_LOGO_SETS:
        pr = _run("prerelease", sd._check_prerelease, img, set_id, era)
        if pr:
            prerelease_found = True

    if era >= 3 or era == 0 or set_id in sd._PRERELEASE_TEXT_SETS or prerelease_found:
        _run("staff_stamp", sd._check_staff_stamp, img, set_id, era)

    if set_id in sd._WINNER_STAMP_SETS:
        _run("winner_stamp", sd._check_winner_stamp, img, set_id, era)

    # === TIER 4: Promo sets ===
    if set_id == "basep":
        _run("black_star_promo", sd._check_black_star_promo, img)
        bare_id = card_id.split("/")[0]
        if bare_id in sd._W_STAMP_ELIGIBLE_CARDS:
            _run("w_stamp", sd._check_w_stamp, img, set_id)

    if set_id == "np":
        _run("black_star_promo", sd._check_black_star_promo, img)

    if set_id in sd._PROMO_SETS:
        _run("promo_stamp", sd._check_promo_stamp, img)

    if set_id in sd._MODERN_PROMO_SETS:
        _run("modern_promo", sd._check_modern_promo, img)
        _run("build_battle", sd._check_build_battle_stamp, img, set_id, era)
        if set_id == "svp":
            _run("pokemon_center", sd._check_pokemon_center_stamp, img, set_id, era)

    # === TIER 5: Retailer exclusives ===
    if era in sd._TOYS_R_US_ERAS or era == 0:
        _run("toys_r_us", sd._toys_r_us_as_dict, img, set_id, era)

    if era in (6, 7) or era == 0:
        _run("build_a_bear", sd._build_a_bear_as_dict, img, set_id, era)

    # === TIER 6: Special product ===
    if set_id in sd._MCDONALDS_SETS or set_id.startswith("mcd"):
        _run("mcdonalds_holo", sd._check_mcdonalds_holo, img, set_id, era)

    if set_id == "pgo":
        _run("peelable_ditto", sd._check_peelable_ditto, img, set_id)

    # === TIER 7: League/tournament ===
    if era >= 3 or era == 0:
        _run("league_stamps", sd._check_league_stamps, img, set_id, era)
        _run("crosshatch_holo", sd._check_crosshatch_holo, img, set_id, era)

    # === TIER 8: Holo pattern ===
    _holo_cache = {}

    def _hf_check(img_bgr):
        if "hf" not in _holo_cache:
            finish, conf = sd._check_holo_finish(img_bgr, set_id, era)
            _holo_cache["hf"] = (finish, conf)
        finish, conf = _holo_cache["hf"]
        return {
            "detected": finish == "holofoil",
            "confidence": conf,
            "position": "artwork",
            "holo_type": finish,
        }

    _run("holo_finish", _hf_check, img)

    if era >= sd._REVERSE_HOLO_MIN_ERA or era == 0:
        def _rh_check(img_bgr):
            if "rh" not in _holo_cache:
                label, conf = sd._check_reverse_holo(img_bgr, set_id, era)
                _holo_cache["rh"] = (label, conf)
            label, conf = _holo_cache["rh"]
            return {
                "detected": label == "reverse_holo",
                "confidence": conf,
                "position": "body",
                "holo_type": label,
            }
        _run("reverse_holo", _rh_check, img)

    if era >= sd._CRACKED_ICE_MIN_ERA or era == 0:
        def _ci_check(img_bgr):
            detected, conf = sd._check_cracked_ice_holo(img_bgr, set_id, era)
            return {
                "detected": detected,
                "confidence": conf,
                "position": "artwork",
            }
        _run("cracked_ice_holo", _ci_check, img)

    return result


if __name__ == "__main__":
    test_cards = [
        ('data/inbox/page_20260305_094228_cards/card_00.png', 'ex15-10/normal', 'EX era'),
        ('data/inbox/page_20260307_014406_cards/card_00.png', 'gym2-87/normal', 'WotC era'),
        ('data/inbox/page_20260320_223702_cards/card_00.png', 'ex1-64/normal', 'EX normal'),
        ('data/inbox/page_20260307_132359_cards/card_00.png', 'dpp-DP09/normal', 'DP promo'),
        ('data/inbox/page_20260307_020047_cards/card_05.png', 'basep-34/normal', 'WotC promo'),
    ]

    all_results = []

    for path, card_id, era_desc in test_cards:
        p(f'\n{"="*60}')
        p(f'CARD: {era_desc}  id={card_id}')
        p(f'  {"Check":<25s} {"Time":>8s}')
        p(f'  {"-"*25} {"-"*8}')

        t0 = time.perf_counter()
        result = timed_detect(path, card_id)
        dt = time.perf_counter() - t0

        check_times = result.get('_check_times', {})
        total_check = sum(check_times.values())
        p(f'  {"-"*25} {"-"*8}')
        p(f'  {"TOTAL":<25s} {dt*1000:8.1f}ms  (check sum: {total_check*1000:.1f}ms)')
        p(f'  Stamps detected: {result.get("stamps_detected", [])}')

        all_results.append((era_desc, card_id, dt, check_times))

    # Summary
    p(f'\n{"="*60}')
    p('SUMMARY TABLE')
    p(f'{"="*60}')
    p(f'{"Era":<15s} {"Total":>8s} {"Checks":>7s} {"Slowest":<25s} {"Time":>8s}')
    p(f'{"-"*15} {"-"*8} {"-"*7} {"-"*25} {"-"*8}')

    for era_desc, card_id, dt, ct in all_results:
        if ct:
            s_name, s_time = max(ct.items(), key=lambda x: x[1])
            p(f'{era_desc:<15s} {dt*1000:7.0f}ms {len(ct):>7d} {s_name:<25s} {s_time*1000:7.0f}ms')

    # Aggregate per-check
    p(f'\n{"="*60}')
    p('PER-CHECK AGGREGATE')
    p(f'{"="*60}')
    from collections import defaultdict
    agg = defaultdict(list)
    for _, _, _, ct in all_results:
        for name, t in ct.items():
            agg[name].append(t)

    p(f'{"Check":<25s} {"N":>3s} {"Mean":>8s} {"Max":>8s} {"Total":>8s}')
    p(f'{"-"*25} {"-"*3} {"-"*8} {"-"*8} {"-"*8}')
    for name, times in sorted(agg.items(), key=lambda x: -max(x[1])):
        p(f'{name:<25s} {len(times):>3d} {sum(times)/len(times)*1000:7.1f}ms {max(times)*1000:7.1f}ms {sum(times)*1000:7.1f}ms')

    # Fast vs full
    p(f'\n{"="*60}')
    p('FAST vs FULL MODE')
    p(f'{"="*60}')
    for path, card_id, era_desc in test_cards:
        t0 = time.perf_counter()
        r_fast = timed_detect(path, card_id, fast=True)
        dt_fast = time.perf_counter() - t0

        t0 = time.perf_counter()
        r_full = timed_detect(path, card_id, fast=False)
        dt_full = time.perf_counter() - t0

        speedup = dt_full / dt_fast if dt_fast > 0 else 0
        p(f'{era_desc:<15s}  Fast: {dt_fast*1000:6.1f}ms ({len(r_fast.get("stamps_checked",[])):2d})  '
          f'Full: {dt_full*1000:6.1f}ms ({len(r_full.get("stamps_checked",[])):2d})  '
          f'Speedup: {speedup:.0f}x')
