"""Test script for card number OCR from the bottom of Pokemon cards.

Tests the extract_card_number function from ocr_matcher.py on actual
card segment images from binder page scans.
"""
import os
import sys

BASE = '/home/godli/cardprice/data/inbox'


def main():
    """Test card number OCR on both page 2 and page 3 binder scan cards."""
    from cardprice.ml.ocr_matcher import extract_card_number

    dirs = [
        os.path.join(BASE, 'page_20260228_202134_cards'),
        os.path.join(BASE, 'page_20260228_195512_cards'),
    ]

    total = 0
    found = 0

    for d in dirs:
        if not os.path.isdir(d):
            print(f"Skipping {d}: not found")
            continue
        print(f"\n=== {os.path.basename(d)} ===")
        for f in sorted(os.listdir(d)):
            if not f.endswith('.png'):
                continue
            fpath = os.path.join(d, f)
            total += 1

            card_num, set_total, conf = extract_card_number(fpath)

            if card_num:
                found += 1
                print(f"  {f}: FOUND {card_num}/{set_total} (conf={conf:.2f})")
            else:
                print(f"  {f}: no number found")

    print(f"\n=== SUMMARY: {found}/{total} card numbers found ===")


def test_parse_card_number():
    """Test the _parse_card_number function with simulated OCR results."""
    from cardprice.ml.ocr_matcher import _parse_card_number

    test_cases = [
        ([("16/132", 0.9)], "16", "132"),
        ([("22/110", 0.85)], "22", "110"),
        ([("16 / 132", 0.8)], "16", "132"),
        ([("l6/l32", 0.5)], "16", "132"),
        ([("16|132", 0.6)], "16", "132"),
        ([("DP293 2007Pokemon 16/132", 0.7)], "16", "132"),
        ([("FoketOM Nenene", 0.02)], None, None),
        ([("200/50", 0.5)], None, None),
        ([("7/132", 0.9)], "7", "132"),
        ([("100/130", 0.9)], "100", "130"),
    ]

    passed = 0
    for inputs, exp_n, exp_t in test_cases:
        num, total, conf = _parse_card_number(inputs)
        ok = num == exp_n and total == exp_t
        passed += ok
        status = "OK" if ok else "FAIL"
        print(f"  {status}: {repr(inputs[0][0]):30} -> {num}/{total} (expect {exp_n}/{exp_t})")

    print(f"\nParsing tests: {passed}/{len(test_cases)} passed")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--parse-test':
        test_parse_card_number()
    else:
        main()
