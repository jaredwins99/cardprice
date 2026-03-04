#!/usr/bin/env python3
"""Test card number OCR with aggressive upscale + preprocessing.

Research script: Can we read the small card numbers (e.g. "9/101") from
630x880 card segments using upscaling + CLAHE + Tesseract/EasyOCR?

RESULT (2026-03-01): 0/26 across ALL preprocessing pipelines tested.

Pipelines tested:
  1. Bottom 15%, left/right half, 4x INTER_CUBIC, unsharp mask, Otsu: 0/26
  2. Bottom 12%, left/right half, 6x INTER_LANCZOS4, sharpen, Otsu: 0/26
  3. Bottom 12%, full width, 6x CLAHE + inverted + 8x raw: 0/26 (EasyOCR)
  4. Same as #3 but with Tesseract PSM 6 and PSM 11: 1/26 (false positive)
  5. Bottom 4%, left/right, 10x CLAHE: 0/26
  6. Full binder page (3024x4032), grid extraction, bottom 5%, 6x CLAHE: 0/26

Root cause: Card numbers are ~5-8px tall at 630px card width (~10-16px at
full binder resolution of ~1008px width). Even with 10x upscale, the
original pixels are too few and too blurred from phone camera to reconstruct
readable glyphs. The text is below the Shannon-Nyquist limit for the
camera's effective resolution at this print size.

Pixel analysis of Vibrava card_03 bottom-right (rows 861-877) showed
character-shaped dark regions from copyright text and card number, but
individual digits merge into amorphous blobs.

Key numbers:
  - Segment resolution: 630x880
  - Full binder card: ~1008x1344 (3024x4032 / 3x3 grid)
  - Card number text height: ~5-8px (segment) / ~10-16px (binder)
  - Minimum for OCR: ~20-30px character height
  - Resolution gap: 2-4x too little

Conclusion: Card number OCR is NOT viable at current phone camera
distance/resolution for binder pages. Would require either:
  a) Single-card close-up photos (not binder pages)
  b) Higher resolution camera / closer distance
  c) A specialized super-resolution ML model trained on card number fonts
"""
import cv2
import numpy as np
import json
import re
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

EVAL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'eval', 'binder_eval.json')
SAVE = '/tmp/card_number_crops'


def get_crops(img):
    """Get multiple bottom crops at different scales and preprocessing."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crops = {}

    # Bottom 12%, full width, 6x, CLAHE enhanced
    y1 = int(h * 0.88)
    bottom = gray[y1:h, :]
    bh, bw = bottom.shape
    up = cv2.resize(bottom, (bw * 6, bh * 6), interpolation=cv2.INTER_LANCZOS4)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(up)
    crops['clahe_6x'] = enhanced
    crops['inv_clahe_6x'] = 255 - enhanced

    # Bottom 8%, full width, 8x, raw gray
    y2 = int(h * 0.92)
    bottom2 = gray[y2:h, :]
    bh2, bw2 = bottom2.shape
    up2 = cv2.resize(bottom2, (bw2 * 8, bh2 * 8), interpolation=cv2.INTER_LANCZOS4)
    crops['raw_8x_narrow'] = up2

    return crops


def run_all_ocr(crops, name, reader):
    """Run OCR on all crop variants."""
    all_texts = []
    os.makedirs(SAVE, exist_ok=True)

    for label, crop_img in crops.items():
        cv2.imwrite(f'{SAVE}/{name}_{label}.png', crop_img)

        # EasyOCR
        results = reader.readtext(crop_img, detail=1, paragraph=False)
        for bbox, text, conf in results:
            all_texts.append((label + '_easy', text, conf))

        # Tesseract if available
        try:
            import pytesseract
            for psm in [6, 11]:
                tess_text = pytesseract.image_to_string(
                    crop_img,
                    config=f'--psm {psm} -c tessedit_char_whitelist=0123456789/H'
                ).strip()
                if tess_text:
                    all_texts.append((f'{label}_tess_psm{psm}', tess_text, 0.5))
        except ImportError:
            pass

    return all_texts


def find_number(texts, expected):
    """Try to find expected card number in OCR results."""
    for method, text, conf in texts:
        c = (text.replace('l', '1').replace('L', '1').replace('O', '0')
             .replace('o', '0').replace('|', '/').replace('I', '1')
             .replace('\\', '/'))
        m = re.search(r'(H?\d{1,3})\s*/\s*(H?\d{1,3})', c)
        if m:
            num = m.group(1)
            if num.upper() == expected.upper() or num.lstrip('0') == expected.lstrip('0'):
                return True, m.group(0), text, conf, method
        if re.search(r'(?<!\d)' + re.escape(expected) + r'(?!\d)', c):
            return True, expected, text, conf, method
    return False, None, None, 0, None


def main():
    import easyocr
    print("Loading EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False)
    print("Loaded.\n")

    with open(EVAL_PATH) as f:
        eval_data = json.load(f)

    total = found = 0
    for page in eval_data['pages']:
        seg_dir = page['segments_dir']
        pname = os.path.basename(seg_dir)
        print(f"\n=== {pname} ===")
        for card in page['cards']:
            if not card['card_id']:
                continue
            seg_path = os.path.join(
                os.path.dirname(__file__), '..', seg_dir, card['segment']
            )
            img = cv2.imread(seg_path)
            if img is None:
                continue
            total += 1
            expected = card['card_id'].split('/')[0].split('-')[-1]
            cname = card['segment'].replace('.png', '')

            crops = get_crops(img)
            all_texts = run_all_ocr(crops, f'{pname}_{cname}', reader)

            ok, ms, raw, conf, method = find_number(all_texts, expected)
            if ok:
                found += 1
                print(f"  {card['segment']}: {card['name']} #{expected}"
                      f" -> MATCH '{ms}' via {method} (raw='{raw}')")
            else:
                readable = [f'[{m}]"{t}"' for m, t, c in all_texts if c > 0.05][:8]
                print(f"  {card['segment']}: {card['name']} #{expected}"
                      f" -> MISS  {'; '.join(readable) or '(nothing)'}")

    print(f"\n=== RESULT: {found}/{total} matched ({100*found/total:.0f}%) ===")


if __name__ == '__main__':
    main()
