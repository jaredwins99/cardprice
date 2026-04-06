#!/usr/bin/env python3
"""Test SIFT+RANSAC card alignment for condition assessment.

Tests alignment quality on eval segments against reference images,
including edge cases (wrong reference, blur, rotation/skew).

Outputs visual results to data/eval/alignment_test/.
"""

import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cardprice.ml.card_aligner import align_cards, compute_difference_map

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "data", "eval", "alignment_test")
CARD_IMAGES = os.path.join(BASE, "data", "card_images")

os.makedirs(OUT_DIR, exist_ok=True)


# --- Test pairs: (label, segment_path, card_id) ---
# Selected from v2 eval results -- correctly matched cards across different sets/eras
TEST_PAIRS = [
    ("ex15_metagross",
     "data/inbox/page_20260228_174819_cards_hires/card_01.png",
     "ex15-42"),
    ("ecard3_raikou",
     "data/inbox/page_20260228_195512_cards_hires/card_00.png",
     "ecard3-80"),
    ("dp3_gardevoir",
     "data/inbox/page_20260228_202134_cards_hires/card_05.png",
     "dp3-16"),
    ("pl3_blaziken",
     "data/inbox/page_20260228_202134_cards_hires/card_03.png",
     "pl3-13"),
    ("dp1_infernape",
     "data/inbox/page_20260228_202134_cards_hires/card_08.png",
     "dp1-16"),
]


def ref_path_for(card_id: str) -> str:
    """Get reference image path for a card ID like 'ex15-42'."""
    set_id = card_id.rsplit("-", 1)[0]
    return os.path.join(CARD_IMAGES, set_id, f"{card_id}_normal.png")


def compute_reprojection_rmse(result) -> float:
    """Compute RMSE of the homography reprojection on inlier matches.

    This measures how well the homography fits -- sub-pixel RMSE means
    excellent alignment.
    """
    if not result.success:
        return float("inf")

    # We can estimate from the RANSAC threshold and inlier ratio
    # For a proper RMSE we'd need the actual keypoint coords, which
    # aren't stored in AlignmentResult. Instead, we use a proxy:
    # inlier_ratio * ransac_threshold gives a rough upper bound.
    inlier_ratio = result.num_inliers / max(result.num_good_matches, 1)
    # Good alignments have >80% inlier ratio
    return inlier_ratio


def save_comparison(label: str, query_img, ref_img, result, suffix=""):
    """Save a side-by-side comparison: query | ref | aligned | diff_map."""
    tag = f"{label}{suffix}"

    if not result.success:
        print(f"  [{tag}] FAILED: {result.error}")
        return

    # Resize all to same height for side-by-side
    h = max(query_img.shape[0], ref_img.shape[0], result.aligned.shape[0])
    target_w = int(h * 245 / 342)  # roughly card aspect ratio

    def resize(img):
        return cv2.resize(img, (target_w, h), interpolation=cv2.INTER_AREA)

    q_resized = resize(query_img)
    r_resized = resize(ref_img)
    a_resized = resize(result.aligned)

    diff_map = compute_difference_map(result.aligned, ref_img)
    d_resized = resize(diff_map)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for img, text in [(q_resized, "Query"), (r_resized, "Reference"),
                      (a_resized, "Aligned"), (d_resized, "Diff Map")]:
        cv2.putText(img, text, (10, 30), font, 0.8, (0, 255, 0), 2)

    combined = np.hstack([q_resized, r_resized, a_resized, d_resized])
    out_path = os.path.join(OUT_DIR, f"{tag}_comparison.jpg")
    cv2.imwrite(out_path, combined)
    print(f"  [{tag}] saved: {out_path}")


def test_normal_alignment():
    """Test 1: Normal alignment on correctly identified eval segments."""
    print("\n=== TEST 1: Normal alignment on eval segments ===\n")

    results_table = []

    for label, seg_rel, card_id in TEST_PAIRS:
        seg_path = os.path.join(BASE, seg_rel)
        rp = ref_path_for(card_id)

        if not os.path.exists(seg_path):
            print(f"  SKIP {label}: segment not found at {seg_path}")
            continue
        if not os.path.exists(rp):
            print(f"  SKIP {label}: reference not found at {rp}")
            continue

        t0 = time.time()
        result = align_cards(seg_path, rp)
        elapsed = time.time() - t0

        query_img = cv2.imread(seg_path)
        ref_img = cv2.imread(rp)
        save_comparison(label, query_img, ref_img, result)

        inlier_ratio = result.num_inliers / max(result.num_good_matches, 1) if result.success else 0
        results_table.append({
            "label": label,
            "card_id": card_id,
            "success": result.success,
            "keypoints_query": result.num_keypoints_query,
            "keypoints_ref": result.num_keypoints_ref,
            "good_matches": result.num_good_matches,
            "inliers": result.num_inliers,
            "inlier_ratio": round(inlier_ratio, 3),
            "time_ms": round(elapsed * 1000, 1),
            "error": result.error,
        })

        status = "OK" if result.success else f"FAIL: {result.error}"
        print(f"  {label}: {status} | kp={result.num_keypoints_query}/{result.num_keypoints_ref} "
              f"matches={result.num_good_matches} inliers={result.num_inliers} "
              f"ratio={inlier_ratio:.3f} time={elapsed*1000:.0f}ms")

    return results_table


def test_wrong_reference():
    """Test 2: What happens when we align against the WRONG reference image?

    Should produce fewer inliers and/or fail validation.
    """
    print("\n=== TEST 2: Wrong reference image ===\n")

    # Use first segment but align against a completely different card
    seg_path = os.path.join(BASE, TEST_PAIRS[0][1])
    correct_ref = ref_path_for(TEST_PAIRS[0][2])

    # Pick a wrong reference from a different set
    wrong_refs = [
        ("wrong_dp1_1", os.path.join(CARD_IMAGES, "dp1", "dp1-1_normal.png")),
        ("wrong_base1_4", os.path.join(CARD_IMAGES, "base1", "base1-4_normal.png")),
        ("wrong_ecard2_95", os.path.join(CARD_IMAGES, "ecard2", "ecard2-95_normal.png")),
    ]

    query_img = cv2.imread(seg_path)
    if query_img is None:
        print("  SKIP: query segment not found")
        return []

    results = []

    # First, correct reference for comparison
    result_correct = align_cards(seg_path, correct_ref)
    print(f"  CORRECT ref ({TEST_PAIRS[0][2]}): success={result_correct.success} "
          f"matches={result_correct.num_good_matches} inliers={result_correct.num_inliers}")

    for wrong_label, wrong_ref_path in wrong_refs:
        if not os.path.exists(wrong_ref_path):
            print(f"  SKIP {wrong_label}: not found")
            continue

        result = align_cards(seg_path, wrong_ref_path)
        ref_img = cv2.imread(wrong_ref_path)
        save_comparison(TEST_PAIRS[0][0], query_img, ref_img, result, suffix=f"_{wrong_label}")

        inlier_ratio = result.num_inliers / max(result.num_good_matches, 1) if result.success else 0
        print(f"  {wrong_label}: success={result.success} matches={result.num_good_matches} "
              f"inliers={result.num_inliers} ratio={inlier_ratio:.3f}"
              + (f" error={result.error}" if result.error else ""))

        results.append({
            "label": wrong_label,
            "success": result.success,
            "good_matches": result.num_good_matches,
            "inliers": result.num_inliers,
            "inlier_ratio": round(inlier_ratio, 3),
            "error": result.error,
        })

    return results


def test_blurry_segment():
    """Test 3: Alignment with artificially blurred segment.

    Simulates a blurry photo by applying Gaussian blur to the query.
    """
    print("\n=== TEST 3: Blurry segment ===\n")

    seg_path = os.path.join(BASE, TEST_PAIRS[0][1])
    rp = ref_path_for(TEST_PAIRS[0][2])

    query_img = cv2.imread(seg_path)
    ref_img = cv2.imread(rp)
    if query_img is None or ref_img is None:
        print("  SKIP: images not found")
        return []

    results = []
    blur_levels = [0, 5, 11, 21, 31, 51]

    for ksize in blur_levels:
        if ksize == 0:
            blurred = query_img.copy()
            blur_label = "original"
        else:
            blurred = cv2.GaussianBlur(query_img, (ksize, ksize), 0)
            blur_label = f"blur_{ksize}"

        from cardprice.ml.card_aligner import align_cards_images
        t0 = time.time()
        result = align_cards_images(blurred, ref_img)
        elapsed = time.time() - t0

        save_comparison(TEST_PAIRS[0][0], blurred, ref_img, result, suffix=f"_{blur_label}")

        inlier_ratio = result.num_inliers / max(result.num_good_matches, 1) if result.success else 0
        print(f"  {blur_label}: success={result.success} kp={result.num_keypoints_query} "
              f"matches={result.num_good_matches} inliers={result.num_inliers} "
              f"ratio={inlier_ratio:.3f} time={elapsed*1000:.0f}ms"
              + (f" error={result.error}" if result.error else ""))

        results.append({
            "blur_kernel": ksize,
            "success": result.success,
            "keypoints_query": result.num_keypoints_query,
            "good_matches": result.num_good_matches,
            "inliers": result.num_inliers,
            "inlier_ratio": round(inlier_ratio, 3),
            "time_ms": round(elapsed * 1000, 1),
            "error": result.error,
        })

    return results


def test_rotated_skewed():
    """Test 4: Alignment with artificially rotated/skewed segments.

    Tests robustness to slight rotation and perspective skew.
    """
    print("\n=== TEST 4: Rotated and skewed segments ===\n")

    seg_path = os.path.join(BASE, TEST_PAIRS[0][1])
    rp = ref_path_for(TEST_PAIRS[0][2])

    query_img = cv2.imread(seg_path)
    ref_img = cv2.imread(rp)
    if query_img is None or ref_img is None:
        print("  SKIP: images not found")
        return []

    from cardprice.ml.card_aligner import align_cards_images
    results = []
    h, w = query_img.shape[:2]

    # Rotation tests
    for angle in [2, 5, 10, 15, 30]:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(query_img, M, (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
        label = f"rot_{angle}deg"

        result = align_cards_images(rotated, ref_img)
        save_comparison(TEST_PAIRS[0][0], rotated, ref_img, result, suffix=f"_{label}")

        inlier_ratio = result.num_inliers / max(result.num_good_matches, 1) if result.success else 0
        print(f"  {label}: success={result.success} matches={result.num_good_matches} "
              f"inliers={result.num_inliers} ratio={inlier_ratio:.3f}"
              + (f" error={result.error}" if result.error else ""))

        results.append({
            "transform": label,
            "success": result.success,
            "good_matches": result.num_good_matches,
            "inliers": result.num_inliers,
            "inlier_ratio": round(inlier_ratio, 3),
            "error": result.error,
        })

    # Perspective skew tests
    for skew_px, skew_label in [(20, "skew_mild"), (50, "skew_moderate"), (100, "skew_severe")]:
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32([
            [skew_px, skew_px // 2],
            [w - skew_px // 2, skew_px],
            [w - skew_px, h - skew_px // 2],
            [skew_px // 2, h - skew_px],
        ])
        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        skewed = cv2.warpPerspective(query_img, M_persp, (w, h),
                                      borderMode=cv2.BORDER_REPLICATE)

        result = align_cards_images(skewed, ref_img)
        save_comparison(TEST_PAIRS[0][0], skewed, ref_img, result, suffix=f"_{skew_label}")

        inlier_ratio = result.num_inliers / max(result.num_good_matches, 1) if result.success else 0
        print(f"  {skew_label}: success={result.success} matches={result.num_good_matches} "
              f"inliers={result.num_inliers} ratio={inlier_ratio:.3f}"
              + (f" error={result.error}" if result.error else ""))

        results.append({
            "transform": skew_label,
            "success": result.success,
            "good_matches": result.num_good_matches,
            "inliers": result.num_inliers,
            "inlier_ratio": round(inlier_ratio, 3),
            "error": result.error,
        })

    return results


def main():
    print("=" * 70)
    print("CARD ALIGNMENT TEST SUITE")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 70)

    all_results = {}

    all_results["normal"] = test_normal_alignment()
    all_results["wrong_ref"] = test_wrong_reference()
    all_results["blur"] = test_blurry_segment()
    all_results["rotation_skew"] = test_rotated_skewed()

    # Save JSON summary
    summary_path = os.path.join(OUT_DIR, "alignment_test_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON summary saved to: {summary_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nNormal alignment:")
    for r in all_results["normal"]:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['label']}: {r['inliers']} inliers, "
              f"ratio={r['inlier_ratio']}, {r['time_ms']}ms")

    print("\nWrong reference (should have fewer inliers or fail):")
    for r in all_results["wrong_ref"]:
        flag = "GOOD" if not r["success"] or r["inliers"] < 20 else "WARN"
        print(f"  [{flag}] {r['label']}: success={r['success']} "
              f"inliers={r['inliers']} ratio={r['inlier_ratio']}")

    print("\nBlur degradation:")
    for r in all_results["blur"]:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] kernel={r['blur_kernel']}: kp={r['keypoints_query']} "
              f"matches={r['good_matches']} inliers={r['inliers']}")

    print("\nRotation/skew robustness:")
    for r in all_results["rotation_skew"]:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['transform']}: matches={r['good_matches']} "
              f"inliers={r['inliers']} ratio={r['inlier_ratio']}")


if __name__ == "__main__":
    main()
