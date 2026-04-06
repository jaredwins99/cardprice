"""Test PaddleOCR vs EasyOCR for HP detection on all 45 eval card segments."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ground truth: (page, card_idx, card_id, name, hp)
# hp=None means trainer/energy/item card (no HP)
GROUND_TRUTH = [
    (0, 0, "ex10-91", "Sitrus Berry", None),
    (0, 1, "sm75-76", "Fiery Flint", None),
    (0, 2, "swsh45sv-SV011", "Eldegoss", 80),
    (0, 3, "swsh6-10", "Abomasnow", 140),
    (0, 4, "ecard2-137", "Traveling Salesman", None),
    (0, 5, "np-28", "Championship Arena", None),
    (0, 6, "xyp-XY143", "Magikarp", 30),
    (0, 7, "xy1-76", "Malamar", 100),
    (0, 8, "sv8-171", "Deduction Kit", None),
    (1, 0, "swsh10-111", "Bronzor", 60),
    (1, 1, "base5-27", "Dark Machamp", 70),
    (1, 2, "swsh3-69", "Mew V", 180),
    (1, 3, "sm7-178", "Acro Bike", None),
    (1, 4, "me2pt5-60", "Eelektrik", 90),
    (1, 5, "sv5-128", "Dunsparce", 60),
    (1, 6, "rsv10pt5-76", "Stoutland", 160),
    (1, 7, "sv1-222", "Skwovet", 60),
    (1, 8, "sv10-34", "Ethan's Typhlosion", 170),
    (2, 0, "bw10-7", "Shelmet", 60),
    (2, 1, "sv3-144", "Bronzor", 70),
    (2, 2, "sv2-239", "Dedenne ex", 170),
    (2, 3, "swsh11-30", "Poliwag", 60),
    (2, 4, "sv8-219", "Pikachu ex", 200),
    (2, 5, "dp6-97", "Gloom", 80),
    (2, 6, "sv5-174", "Excadrill", 130),
    (2, 7, "xy6-53", "Altaria", 80),
    (2, 8, "xy1-146", "Xerneas-EX", 170),
    (3, 0, "swsh6-143", "Justified Gloves", None),
    (3, 1, "swsh6-81", "Gallade", 170),
    (3, 2, "bw10-99", "Dialga-EX", 180),
    (3, 3, "xy4-111", "Double Colorless Energy", None),
    (3, 4, "pgo-29", "Zapdos", 120),
    (3, 5, "ex13-72", "Numel", 50),
    (3, 6, "bw7-3", "Vileplume", 140),
    (3, 7, "swsh7-236", "Darkness Energy", None),
    (3, 8, "ecard2-135", "Time Shard", None),
    (4, 0, "swshp-SWSH231", "Bulbasaur", 70),
    (4, 1, "swsh45-68", "Gym Trainer", None),
    (4, 2, "swsh2-102", "Galarian Runerigus", 100),
    (4, 3, "sv5-70", "Solosis", 40),
    (4, 4, "sv9-14", "Durant", 90),
    (4, 5, "me2-9", "Nymble", 50),
    (4, 6, "sm3-79", "Passimian", 110),
    (4, 7, "sm11-192", "Coach Trainer", None),
    (4, 8, "base4-50", "Magikarp", 30),
]

# HP region definitions (same as hp_detector.py)
_HP_REGION = {"x1": 0.35, "y1": 0.0, "x2": 1.0, "y2": 0.12}


def crop_region(img, region):
    h, w = img.shape[:2]
    x1 = int(region["x1"] * w)
    y1 = int(region["y1"] * h)
    x2 = int(region["x2"] * w)
    y2 = int(region["y2"] * h)
    return img[y1:y2, x1:x2]


def get_hp_crops(img):
    """Return the same crops as hp_detector.detect_hp()."""
    h, w = img.shape[:2]
    return [
        ("narrow", img[0:int(h * 0.10), int(w * 0.55):int(w * 0.95)]),
        ("default", crop_region(img, _HP_REGION)),
        ("wide_right", img[0:int(h * 0.13), int(w * 0.45):w]),
    ]


def paddle_ocr_on_crop(det, rec, crop, upscale=3):
    """Run PaddleOCR detection+recognition on a crop. Returns list of (text, conf)."""
    if crop.size == 0:
        return []

    # Upscale for better detection (same strategy as ocr_matcher)
    if upscale > 1:
        crop = cv2.resize(crop, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)

    try:
        det_result = det.predict(crop, batch_size=1)
        if not det_result:
            return []

        polys = det_result[0].get('dt_polys')
        if polys is None or (hasattr(polys, 'size') and polys.size == 0) or (isinstance(polys, list) and len(polys) == 0):
            return []

        texts = []
        for poly in polys:
            pts = np.array(poly, dtype=np.float32)
            x_min, y_min = pts.min(axis=0).astype(int)
            x_max, y_max = pts.max(axis=0).astype(int)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(crop.shape[1], x_max)
            y_max = min(crop.shape[0], y_max)
            if x_max <= x_min or y_max <= y_min:
                continue
            text_crop = crop[y_min:y_max, x_min:x_max]
            rec_result = rec.predict(text_crop, batch_size=1)
            if rec_result and rec_result[0].get('rec_text'):
                text = rec_result[0]['rec_text']
                score = rec_result[0].get('rec_score', 0.0)
                texts.append((text, score))
        return texts
    except Exception as e:
        print(f"  PaddleOCR error: {e}")
        import traceback; traceback.print_exc()
        return []


def easyocr_on_crop(reader, crop):
    """Run EasyOCR on a crop. Returns list of (text, conf)."""
    if crop.size == 0:
        return []
    try:
        results = reader.readtext(crop, batch_size=8)
        return [(r[1], float(r[2])) for r in results]
    except Exception as e:
        print(f"  EasyOCR error: {e}")
        return []


def parse_hp(texts):
    """Reuse hp_detector parsing logic."""
    from cardprice.ml.hp_detector import _parse_hp_from_texts
    return _parse_hp_from_texts(texts)


def main():
    base = Path("/home/godli/cardprice/data/test_binder_pages")

    # Import the updated detect_hp (combined EasyOCR + PaddleOCR)
    from cardprice.ml.hp_detector import detect_hp as detect_hp_combined
    from cardprice.ml.hp_detector import extract_hp_paddle

    # Initialize PaddleOCR
    print("Loading PaddleOCR engines...")
    t0 = time.time()
    from cardprice.ml.ocr_matcher import get_paddle_engines
    det, rec = get_paddle_engines()
    print(f"PaddleOCR loaded in {time.time()-t0:.1f}s")

    # Initialize EasyOCR
    print("Loading EasyOCR...")
    t0 = time.time()
    import easyocr
    easy_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    print(f"EasyOCR loaded in {time.time()-t0:.1f}s")

    # Results tracking
    easy_results = []  # (card_id, expected, detected)
    paddle_results = []
    combined_results = []

    # Only test Pokemon cards (have HP), skip trainers/energies
    pokemon_cards = [(p, ci, cid, name, hp) for p, ci, cid, name, hp in GROUND_TRUTH if hp is not None]
    trainer_cards = [(p, ci, cid, name, hp) for p, ci, cid, name, hp in GROUND_TRUTH if hp is None]

    print(f"\n{'='*80}")
    print(f"Testing {len(pokemon_cards)} Pokemon cards (with HP) + {len(trainer_cards)} trainers/energies (no HP)")
    print(f"{'='*80}\n")

    for page, card_idx, card_id, name, expected_hp in GROUND_TRUTH:
        seg_path = base / f"binder_page_{page:02d}_cards" / f"card_{card_idx:02d}.png"
        if not seg_path.exists():
            print(f"SKIP {card_id} ({name}) - segment not found")
            continue

        img = cv2.imread(str(seg_path))
        if img is None:
            print(f"SKIP {card_id} ({name}) - could not read image")
            continue

        crops = get_hp_crops(img)

        # --- EasyOCR ---
        easy_hp = None
        easy_raw = []
        for crop_name, hp_crop in crops:
            texts = easyocr_on_crop(easy_reader, hp_crop)
            if texts:
                easy_raw.extend(texts)
                hp = parse_hp(texts)
                if hp is not None:
                    easy_hp = hp
                    break

        # --- PaddleOCR ---
        paddle_hp = None
        paddle_raw = []
        for crop_name, hp_crop in crops:
            texts = paddle_ocr_on_crop(det, rec, hp_crop)
            if texts:
                paddle_raw.extend(texts)
                hp = parse_hp(texts)
                if hp is not None:
                    paddle_hp = hp
                    break

        # --- Combined (updated detect_hp) ---
        combined_hp = detect_hp_combined(str(seg_path))

        # Record results
        easy_results.append((card_id, name, expected_hp, easy_hp, easy_raw))
        paddle_results.append((card_id, name, expected_hp, paddle_hp, paddle_raw))
        combined_results.append((card_id, name, expected_hp, combined_hp, []))

        # Display
        is_pokemon = expected_hp is not None
        easy_status = ""
        paddle_status = ""
        combined_status = ""
        if is_pokemon:
            easy_status = "OK" if easy_hp == expected_hp else f"WRONG({easy_hp})" if easy_hp else "MISS"
            paddle_status = "OK" if paddle_hp == expected_hp else f"WRONG({paddle_hp})" if paddle_hp else "MISS"
            combined_status = "OK" if combined_hp == expected_hp else f"WRONG({combined_hp})" if combined_hp else "MISS"
        else:
            easy_status = "OK(None)" if easy_hp is None else f"FALSE_POS({easy_hp})"
            paddle_status = "OK(None)" if paddle_hp is None else f"FALSE_POS({paddle_hp})"
            combined_status = "OK(None)" if combined_hp is None else f"FALSE_POS({combined_hp})"

        print(f"{card_id:25s} {name:25s} HP={str(expected_hp):>4s}  "
              f"Easy={str(easy_hp):>4s} [{easy_status:>12s}]  "
              f"Paddle={str(paddle_hp):>4s} [{paddle_status:>12s}]  "
              f"Combined={str(combined_hp):>4s} [{combined_status:>12s}]")
        if any(s not in ("OK", "OK(None)") for s in (easy_status, paddle_status, combined_status)):
            print(f"  EasyOCR raw: {easy_raw}")
            print(f"  Paddle  raw: {paddle_raw}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # Pokemon cards only (have HP)
    def compute_stats(results, label):
        pokemon = [(cid, name, exp, det_, raw) for cid, name, exp, det_, raw in results if exp is not None]
        trainers = [(cid, name, exp, det_, raw) for cid, name, exp, det_, raw in results if exp is None]
        correct = sum(1 for _, _, exp, det_, _ in pokemon if det_ == exp)
        miss = sum(1 for _, _, exp, det_, _ in pokemon if det_ is None)
        wrong = sum(1 for _, _, exp, det_, _ in pokemon if det_ is not None and det_ != exp)
        fp = sum(1 for _, _, _, det_, _ in trainers if det_ is not None)
        total_correct = correct + (len(trainers) - fp)
        total = len(results)
        return correct, miss, wrong, fp, len(pokemon), len(trainers), total_correct, total

    for label, results in [("EasyOCR", easy_results), ("PaddleOCR", paddle_results), ("Combined", combined_results)]:
        correct, miss, wrong, fp, n_pokemon, n_trainer, total_correct, total = compute_stats(results, label)
        print(f"\n  {label}:")
        print(f"    Pokemon HP: {correct}/{n_pokemon} correct ({correct/n_pokemon*100:.1f}%), {miss} missed, {wrong} wrong")
        print(f"    Trainer false positives: {fp}/{n_trainer}")
        print(f"    Overall: {total_correct}/{total} ({total_correct/total*100:.1f}%)")

    # Show disagreements
    print(f"\n{'='*80}")
    print("DISAGREEMENTS (EasyOCR != PaddleOCR)")
    print(f"{'='*80}")
    for (cid1, name1, exp1, det1, raw1), (cid2, name2, exp2, det2, raw2) in zip(easy_results, paddle_results):
        if det1 != det2:
            print(f"  {cid1:25s} {name1:20s} expected={exp1}  easy={det1}  paddle={det2}")
            print(f"    Easy raw: {raw1}")
            print(f"    Paddle raw: {raw2}")


if __name__ == "__main__":
    main()
