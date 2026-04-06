#!/usr/bin/env python3
"""
Check resolution/dimensions of reference images vs eval card segments.
"""

import os
import glob
import random
from PIL import Image
import psycopg2
from datetime import datetime

print(f"\n{'='*80}")
print(f"IMAGE DIMENSION ANALYSIS - {datetime.now().isoformat()}")
print(f"{'='*80}\n")

# ============================================================================
# 1. SAMPLE 5 REFERENCE IMAGES FROM DIFFERENT SETS
# ============================================================================
print("1. REFERENCE IMAGES (5 random from different sets)")
print("-" * 80)

card_images_dir = "/home/godli/cardprice/data/card_images"
set_dirs = [d for d in os.listdir(card_images_dir)
            if os.path.isdir(os.path.join(card_images_dir, d))]

sampled_sets = random.sample(set_dirs, min(5, len(set_dirs)))
ref_stats = []

for set_id in sampled_sets:
    set_path = os.path.join(card_images_dir, set_id)
    images = glob.glob(os.path.join(set_path, "*.jpg")) + glob.glob(os.path.join(set_path, "*.png"))

    if images:
        img_path = random.choice(images)
        try:
            img = Image.open(img_path)
            width, height = img.size
            filename = os.path.basename(img_path)
            ref_stats.append({"width": width, "height": height, "set": set_id})
            print(f"  {set_id:10s} -> {filename:40s} {width:4d}x{height:4d}")
        except Exception as e:
            print(f"  {set_id:10s} -> ERROR: {e}")

# ============================================================================
# 2. ALL EVAL CARD SEGMENTS DIMENSIONS
# ============================================================================
print(f"\n2. EVAL CARD SEGMENTS")
print("-" * 80)

segment_dirs = [
    "/home/godli/cardprice/data/test_binder_segments",
    "/home/godli/cardprice/data/test_binder_segments_rotated",
    "/home/godli/cardprice/data/test_segments"
]

eval_stats = []
total_eval = 0

for seg_dir in segment_dirs:
    if not os.path.exists(seg_dir):
        continue

    images = glob.glob(os.path.join(seg_dir, "*.jpg")) + glob.glob(os.path.join(seg_dir, "*.png"))
    if not images:
        continue

    dir_name = os.path.basename(seg_dir)
    print(f"\n  {dir_name}/ ({len(images)} segments):")

    for img_path in sorted(images):
        try:
            img = Image.open(img_path)
            width, height = img.size
            filename = os.path.basename(img_path)
            eval_stats.append({"width": width, "height": height, "dir": dir_name})
            print(f"    {filename:40s} {width:4d}x{height:4d}")
            total_eval += 1
        except Exception as e:
            print(f"    {filename:40s} ERROR: {e}")

# ============================================================================
# 3. CHECK FOR IMAGE_LARGE URLS IN DIM_CARDS
# ============================================================================
print(f"\n3. IMAGE_LARGE URLS IN DIM_CARDS")
print("-" * 80)

try:
    conn = psycopg2.connect("dbname=cardprice user=godli")
    cur = conn.cursor()

    # Count cards with image_large
    cur.execute("SELECT COUNT(*) FROM dim_cards WHERE image_large IS NOT NULL")
    large_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM dim_cards")
    total_count = cur.fetchone()[0]

    print(f"  Total cards: {total_count}")
    print(f"  Cards with image_large URL: {large_count}")
    print(f"  Coverage: {large_count/total_count*100:.1f}%")

    # Get a sample with image_large
    cur.execute("""
        SELECT card_id, name, set_id, image_small, image_large
        FROM dim_cards
        WHERE image_large IS NOT NULL
        LIMIT 3
    """)

    print(f"\n  Sample cards with image_large:")
    samples = cur.fetchall()
    for card_id, name, set_id, img_small, img_large in samples:
        print(f"    {card_id:20s} {name:30s}")
        print(f"      small: {img_small[:60]}...")
        print(f"      large: {img_large[:60]}...")

    conn.close()

except Exception as e:
    print(f"  ERROR connecting to database: {e}")

# ============================================================================
# 4. DOWNLOAD AND TEST ONE LARGE IMAGE
# ============================================================================
print(f"\n4. TEST DOWNLOADING ONE LARGE IMAGE")
print("-" * 80)

try:
    import requests

    conn = psycopg2.connect("dbname=cardprice user=godli")
    cur = conn.cursor()

    # Get one sample with both URLs
    cur.execute("""
        SELECT card_id, image_small, image_large
        FROM dim_cards
        WHERE image_large IS NOT NULL
        LIMIT 1
    """)

    result = cur.fetchone()
    if result:
        card_id, img_small_url, img_large_url = result

        print(f"  Testing card: {card_id}")

        # Download small image
        try:
            print(f"\n  Downloading small image...")
            resp_small = requests.get(img_small_url, timeout=10)
            resp_small.raise_for_status()
            small_img = Image.open(__import__('io').BytesIO(resp_small.content))
            small_width, small_height = small_img.size
            small_size_bytes = len(resp_small.content)
            print(f"    Dimensions: {small_width}x{small_height}")
            print(f"    File size: {small_size_bytes:,} bytes ({small_size_bytes/1024:.1f} KB)")
        except Exception as e:
            print(f"    ERROR: {e}")

        # Download large image
        try:
            print(f"\n  Downloading large image...")
            resp_large = requests.get(img_large_url, timeout=10)
            resp_large.raise_for_status()
            large_img = Image.open(__import__('io').BytesIO(resp_large.content))
            large_width, large_height = large_img.size
            large_size_bytes = len(resp_large.content)
            print(f"    Dimensions: {large_width}x{large_height}")
            print(f"    File size: {large_size_bytes:,} bytes ({large_size_bytes/1024:.1f} KB)")

            # Calculate difference
            if 'small_width' in locals():
                width_ratio = large_width / small_width
                height_ratio = large_height / small_height
                area_ratio = (large_width * large_height) / (small_width * small_height)
                print(f"\n  Resolution improvement:")
                print(f"    Width ratio: {width_ratio:.2f}x")
                print(f"    Height ratio: {height_ratio:.2f}x")
                print(f"    Area ratio: {area_ratio:.2f}x")

        except Exception as e:
            print(f"    ERROR: {e}")

    conn.close()

except ImportError:
    print("  ERROR: requests library not available (install: pip install requests)")
except Exception as e:
    print(f"  ERROR: {e}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print(f"\n5. SUMMARY STATISTICS")
print("-" * 80)

if ref_stats:
    avg_ref_w = sum(s['width'] for s in ref_stats) / len(ref_stats)
    avg_ref_h = sum(s['height'] for s in ref_stats) / len(ref_stats)
    print(f"  Reference images (n={len(ref_stats)}):")
    print(f"    Average dimensions: {avg_ref_w:.0f}x{avg_ref_h:.0f}")
    print(f"    Range: {min(s['width'] for s in ref_stats)}-{max(s['width'] for s in ref_stats)}w, "
          f"{min(s['height'] for s in ref_stats)}-{max(s['height'] for s in ref_stats)}h")

if eval_stats:
    avg_eval_w = sum(s['width'] for s in eval_stats) / len(eval_stats)
    avg_eval_h = sum(s['height'] for s in eval_stats) / len(eval_stats)
    print(f"\n  Eval card segments (n={len(eval_stats)}):")
    print(f"    Average dimensions: {avg_eval_w:.0f}x{avg_eval_h:.0f}")
    print(f"    Range: {min(s['width'] for s in eval_stats)}-{max(s['width'] for s in eval_stats)}w, "
          f"{min(s['height'] for s in eval_stats)}-{max(s['height'] for s in eval_stats)}h")

if ref_stats and eval_stats:
    print(f"\n  Comparison:")
    print(f"    Ref avg area: {avg_ref_w * avg_ref_h:.0f} pixels")
    print(f"    Eval avg area: {avg_eval_w * avg_eval_h:.0f} pixels")
    print(f"    Ratio: {(avg_eval_w * avg_eval_h) / (avg_ref_w * avg_ref_h):.2f}x")

print(f"\n{'='*80}\n")
