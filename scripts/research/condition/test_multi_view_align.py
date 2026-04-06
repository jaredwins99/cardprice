#!/usr/bin/env python3
"""
Multi-view card photo alignment prototype using SIFT feature matching.

Purpose: Align 2+ photos of the same card taken at different angles to a
common reference frame, enabling pixel-level comparison for condition assessment.

Pipeline:
  1. SIFT keypoint detection on both images
  2. FLANN-based feature matching with Lowe's ratio test
  3. Homography estimation with RANSAC
  4. Optional ECC (Enhanced Correlation Coefficient) refinement
  5. Warp image to reference frame
  6. Compute difference map

Usage:
  python scripts/test_multi_view_align.py [--img1 PATH] [--img2 PATH] [--no-ecc] [--save-dir DIR]

If only img1 is provided (or neither), a simulated second view is generated
from the first image with perspective distortion + noise.

RESEARCH FINDINGS (2026-03-06, 630x880 card segments):
=========================================================

1. SIFT Keypoints per card segment:
   - Energy cards (low detail): ~284 keypoints
   - Pokemon cards (typical):   2500-3500 keypoints (nfeatures=5000 cap)
   - With nfeatures=2000 cap:   1000-2000 good features on detailed cards

2. Good matches after Lowe's ratio test (threshold=0.75):
   - Energy card:  127-169 matches (45-60% of raw)
   - Pokemon card: 871-1161 matches (43-58% of raw)
   - RANSAC inlier rate: 86-97%

3. Alignment error (geometric, corner reprojection):
   - Energy card (284 kps):  0.12-0.50 px mean error across all severities
   - Pokemon card (2000 kps): 0.05-0.14 px mean error across all severities
   - Even "heavy" distortion (20-30px corner shift) recovers to <0.5 px

4. ECC refinement:
   - Does NOT improve alignment. SIFT+RANSAC already achieves sub-pixel
     geometric accuracy, and ECC slightly degrades it (-0.1% to -2% RMSE)
   - ECC is confused by intensity differences (noise, blur, brightness)
     between the simulated views that SIFT feature matching ignores
   - ECC adds significant time: 85-4200ms (varies with convergence)
   - RECOMMENDATION: Skip ECC for this use case. Use --no-ecc flag.

5. Total processing time (without ECC):
   - Energy card:  ~135 ms (detection + matching + homography)
   - Pokemon card: ~140 ms
   - With ECC:     +85 to +4200 ms (unpredictable)

CONCLUSION: SIFT + FLANN + RANSAC is sufficient for card alignment.
Sub-pixel accuracy achieved without ECC. For real multi-angle photos,
the main challenge will be occlusion and specular highlights, not
geometric alignment precision.
"""

import argparse
import time
import sys
import os
import cv2
import numpy as np
from pathlib import Path


def create_simulated_view(img, severity="moderate"):
    """Create a simulated second view with perspective distortion and noise."""
    h, w = img.shape[:2]

    if severity == "mild":
        # Small perspective shift (~5px corners)
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        pts2 = np.float32([[5, 3], [w - 4, 6], [w - 3, h - 5], [4, h - 4]])
        blur_k = 3
        noise_std = 5
    elif severity == "moderate":
        # Medium perspective shift (~10-15px corners)
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        pts2 = np.float32([[10, 5], [w - 10, 15], [w - 5, h - 10], [5, h - 5]])
        blur_k = 3
        noise_std = 10
    elif severity == "heavy":
        # Large perspective shift (~20-30px corners) + more noise
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        pts2 = np.float32([[20, 12], [w - 15, 25], [w - 8, h - 18], [12, h - 10]])
        blur_k = 5
        noise_std = 15
    else:
        raise ValueError(f"Unknown severity: {severity}")

    M_true = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M_true, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Add Gaussian blur (simulates slight defocus)
    warped = cv2.GaussianBlur(warped, (blur_k, blur_k), 0)

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, warped.shape).astype(np.float32)
    warped = np.clip(warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Slight brightness shift (simulates different lighting angle)
    warped = cv2.convertScaleAbs(warped, alpha=1.05, beta=8)

    return warped, M_true


def detect_sift_features(img_gray, nfeatures=2000, contrast_threshold=0.04,
                         edge_threshold=10):
    """Detect SIFT keypoints and compute descriptors."""
    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
    )
    kps, descs = sift.detectAndCompute(img_gray, None)
    return kps, descs


def match_features_flann(descs1, descs2, ratio_threshold=0.75):
    """Match features using FLANN with Lowe's ratio test."""
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(descs1, descs2, k=2)

    good = []
    for m, n in raw_matches:
        if m.distance < ratio_threshold * n.distance:
            good.append(m)

    return good, raw_matches


def compute_homography(kps1, kps2, good_matches, ransac_thresh=5.0):
    """Compute homography using RANSAC."""
    if len(good_matches) < 4:
        return None, None

    src_pts = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    inliers = int(mask.sum()) if mask is not None else 0

    return H, mask, inliers


def refine_with_ecc(img1_gray, img2_gray, H_inv_init, max_iter=200, eps=1e-6):
    """Refine alignment using Enhanced Correlation Coefficient (ECC).

    ECC works on grayscale intensity patterns and can achieve sub-pixel
    accuracy, complementing SIFT's feature-level alignment.

    Strategy: Pre-warp img2 using the SIFT homography so it is already
    close to img1. Then run ECC with an identity seed to find the small
    residual correction. Finally compose the two transforms.

    Args:
        img1_gray: Reference image (grayscale)
        img2_gray: Second view (grayscale)
        H_inv_init: Initial homography that warps img2 -> img1 frame
    Returns:
        H_composed: Refined homography (img2 -> img1)
        cc: ECC correlation coefficient (or None on failure)
    """
    h, w = img1_gray.shape

    # Pre-warp img2 into img1's frame using SIFT homography
    img2_prewarped = cv2.warpPerspective(
        img2_gray, H_inv_init, (w, h), borderMode=cv2.BORDER_REPLICATE
    )

    # Start ECC from identity -- it only needs to find the small residual
    warp_matrix = np.eye(3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max_iter,
        eps,
    )

    try:
        # findTransformECC finds W such that warpPerspective(img2_prewarped, W) ~ img1
        cc, H_residual = cv2.findTransformECC(
            img1_gray, img2_prewarped, warp_matrix,
            motionType=cv2.MOTION_HOMOGRAPHY,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
        # Compose: final = H_residual @ H_inv_init
        H_composed = H_residual.astype(np.float64) @ H_inv_init.astype(np.float64)
        return H_composed, cc
    except cv2.error as e:
        print(f"  ECC refinement failed: {e}")
        return H_inv_init, None


def compute_alignment_error(img1_gray, img2_warped_gray):
    """Compute alignment error metrics."""
    # Only compare valid (non-zero) regions
    mask = (img2_warped_gray > 0).astype(np.uint8)
    # Erode mask to avoid border artifacts
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)

    if mask.sum() == 0:
        return float('inf'), float('inf'), None

    diff = cv2.absdiff(img1_gray, img2_warped_gray)
    diff_masked = diff[mask > 0].astype(np.float64)

    rmse = np.sqrt(np.mean(diff_masked ** 2))
    mae = np.mean(diff_masked)

    # Create visual difference map
    diff_map = np.zeros_like(img1_gray)
    diff_map[mask > 0] = diff[mask > 0]

    return rmse, mae, diff_map


def compute_geometric_error(H_estimated, H_true, img_shape):
    """Compute geometric alignment error using corner reprojection."""
    h, w = img_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    # Where the true transform maps corners
    corners_true = cv2.perspectiveTransform(corners, H_true)
    # Where our estimated inverse maps those corners back
    H_est_inv = np.linalg.inv(H_estimated)
    corners_recovered = cv2.perspectiveTransform(corners_true, H_est_inv)

    errors = np.sqrt(np.sum((corners.reshape(-1, 2) - corners_recovered.reshape(-1, 2)) ** 2, axis=1))
    return errors.mean(), errors.max(), errors


def run_alignment(img1_path, img2_path=None, use_ecc=True, save_dir=None,
                  severities=None):
    """Run the full alignment pipeline and report results."""
    print("=" * 70)
    print("MULTI-VIEW CARD ALIGNMENT PROTOTYPE")
    print("=" * 70)

    # Load reference image
    img1 = cv2.imread(img1_path)
    if img1 is None:
        print(f"ERROR: Could not load image: {img1_path}")
        sys.exit(1)
    print(f"\nReference image: {img1_path}")
    print(f"  Dimensions: {img1.shape[1]}x{img1.shape[0]}")

    if severities is None:
        severities = ["mild", "moderate", "heavy"]

    simulated = img2_path is None

    for severity in severities:
        print(f"\n{'=' * 70}")
        if simulated:
            print(f"SIMULATED VIEW  --  severity={severity}")
        else:
            print(f"REAL SECOND VIEW")
        print("=" * 70)

        t_total_start = time.perf_counter()

        # --- Load / generate second image ---
        if simulated:
            img2, M_true = create_simulated_view(img1, severity=severity)
        else:
            img2 = cv2.imread(img2_path)
            if img2 is None:
                print(f"ERROR: Could not load image: {img2_path}")
                continue
            M_true = None
            # Resize img2 to match img1 if needed
            if img2.shape[:2] != img1.shape[:2]:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # --- Step 1: SIFT feature detection ---
        print("\n[1] SIFT Feature Detection")
        t0 = time.perf_counter()
        kps1, descs1 = detect_sift_features(gray1)
        t1 = time.perf_counter()
        kps2, descs2 = detect_sift_features(gray2)
        t2 = time.perf_counter()

        print(f"  Image 1: {len(kps1)} keypoints  ({(t1-t0)*1000:.1f} ms)")
        print(f"  Image 2: {len(kps2)} keypoints  ({(t2-t1)*1000:.1f} ms)")

        if descs1 is None or descs2 is None or len(kps1) < 10 or len(kps2) < 10:
            print("  ERROR: Too few keypoints for matching")
            continue

        # --- Step 2: Feature matching ---
        print("\n[2] Feature Matching (FLANN + Lowe's ratio test)")
        t3 = time.perf_counter()
        good_matches, raw_matches = match_features_flann(descs1, descs2)
        t4 = time.perf_counter()

        print(f"  Raw matches: {len(raw_matches)}")
        print(f"  Good matches (ratio < 0.75): {len(good_matches)}")
        print(f"  Match rate: {len(good_matches)/len(raw_matches)*100:.1f}%")
        print(f"  Matching time: {(t4-t3)*1000:.1f} ms")

        if len(good_matches) < 10:
            print("  ERROR: Too few good matches for homography")
            continue

        # --- Step 3: Homography with RANSAC ---
        print("\n[3] Homography Estimation (RANSAC)")
        t5 = time.perf_counter()
        H_sift, mask, inliers = compute_homography(kps1, kps2, good_matches)
        t6 = time.perf_counter()

        if H_sift is None:
            print("  ERROR: Could not compute homography")
            continue

        print(f"  RANSAC inliers: {inliers}/{len(good_matches)} ({inliers/len(good_matches)*100:.1f}%)")
        print(f"  Homography time: {(t6-t5)*1000:.1f} ms")

        # Warp with SIFT homography
        h, w = gray1.shape
        # H maps img1 keypoints -> img2 keypoints
        # To warp img2 into img1's frame, we need inverse
        H_sift_inv = np.linalg.inv(H_sift)
        warped_sift = cv2.warpPerspective(img2, H_sift_inv, (w, h),
                                          borderMode=cv2.BORDER_REPLICATE)
        warped_sift_gray = cv2.cvtColor(warped_sift, cv2.COLOR_BGR2GRAY)

        # Alignment error after SIFT
        rmse_sift, mae_sift, diff_sift = compute_alignment_error(gray1, warped_sift_gray)
        print(f"\n  SIFT-only alignment error:")
        print(f"    RMSE: {rmse_sift:.2f} px intensity")
        print(f"    MAE:  {mae_sift:.2f} px intensity")

        if M_true is not None:
            geo_mean, geo_max, geo_corners = compute_geometric_error(H_sift, M_true, img1.shape)
            print(f"    Geometric RMSE (corner reprojection): {geo_mean:.3f} px")
            print(f"    Geometric max error: {geo_max:.3f} px")
            print(f"    Per-corner errors: {[f'{e:.3f}' for e in geo_corners]}")

        # --- Step 4: ECC refinement ---
        H_final = H_sift
        if use_ecc:
            print("\n[4] ECC Refinement")
            t7 = time.perf_counter()
            # ECC refines warp from img2 -> img1, so use inverse
            H_ecc, cc = refine_with_ecc(gray1, gray2, H_sift_inv.copy())
            t8 = time.perf_counter()

            if cc is not None:
                print(f"  ECC correlation coefficient: {cc:.6f}")
                print(f"  ECC time: {(t8-t7)*1000:.1f} ms")

                # Warp with ECC-refined homography
                warped_ecc = cv2.warpPerspective(img2, H_ecc, (w, h),
                                                 borderMode=cv2.BORDER_REPLICATE)
                warped_ecc_gray = cv2.cvtColor(warped_ecc, cv2.COLOR_BGR2GRAY)

                rmse_ecc, mae_ecc, diff_ecc = compute_alignment_error(gray1, warped_ecc_gray)
                print(f"\n  ECC-refined alignment error:")
                print(f"    RMSE: {rmse_ecc:.2f} px intensity")
                print(f"    MAE:  {mae_ecc:.2f} px intensity")

                rmse_improvement = (rmse_sift - rmse_ecc) / rmse_sift * 100
                print(f"    RMSE improvement: {rmse_improvement:.1f}%")

                if M_true is not None:
                    H_ecc_fwd = np.linalg.inv(H_ecc)
                    geo_mean_ecc, geo_max_ecc, geo_corners_ecc = compute_geometric_error(
                        H_ecc_fwd, M_true, img1.shape)
                    print(f"    Geometric RMSE (corner reprojection): {geo_mean_ecc:.3f} px")
                    print(f"    Geometric max error: {geo_max_ecc:.3f} px")

                    geo_improve = (geo_mean - geo_mean_ecc) / geo_mean * 100
                    print(f"    Geometric improvement: {geo_improve:.1f}%")

                H_final_inv = H_ecc
            else:
                print("  ECC failed, using SIFT-only homography")
                H_final_inv = H_sift_inv
                diff_ecc = diff_sift
                warped_ecc = warped_sift
        else:
            H_final_inv = H_sift_inv
            diff_ecc = diff_sift
            warped_ecc = warped_sift

        t_total = time.perf_counter() - t_total_start

        # --- Step 5: Difference map ---
        print(f"\n[5] Difference Map")
        diff_color = cv2.applyColorMap(
            cv2.normalize(diff_ecc if diff_ecc is not None else diff_sift,
                          None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        print(f"  Generated heatmap difference map")

        # --- Timing summary ---
        print(f"\n--- TIMING SUMMARY ---")
        print(f"  SIFT detection (2 images): {(t2-t0)*1000:.1f} ms")
        print(f"  Feature matching:          {(t4-t3)*1000:.1f} ms")
        print(f"  Homography (RANSAC):       {(t6-t5)*1000:.1f} ms")
        if use_ecc and cc is not None:
            print(f"  ECC refinement:            {(t8-t7)*1000:.1f} ms")
        print(f"  Total:                     {t_total*1000:.1f} ms")

        # --- Save outputs ---
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            tag = severity if simulated else "real"

            cv2.imwrite(str(save_path / f"img1_reference.jpg"), img1)
            cv2.imwrite(str(save_path / f"img2_{tag}.jpg"), img2)
            cv2.imwrite(str(save_path / f"warped_{tag}.jpg"), warped_ecc)
            cv2.imwrite(str(save_path / f"diff_{tag}.jpg"), diff_color)

            # Draw matches visualization
            match_img = cv2.drawMatches(
                img1, kps1, img2, kps2, good_matches[:50], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv2.imwrite(str(save_path / f"matches_{tag}.jpg"), match_img)

            # Side-by-side comparison: reference | warped | diff
            comparison = np.hstack([
                img1,
                warped_ecc,
                diff_color,
            ])
            cv2.imwrite(str(save_path / f"comparison_{tag}.jpg"), comparison)
            print(f"\n  Outputs saved to {save_path}/")

        if not simulated:
            break  # Only one pass for real image pairs

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-view card alignment prototype using SIFT + ECC"
    )
    parser.add_argument("--img1", default="data/test_binder_segments/card_00.png",
                        help="Reference image path")
    parser.add_argument("--img2", default=None,
                        help="Second view path (omit to simulate)")
    parser.add_argument("--no-ecc", action="store_true",
                        help="Skip ECC refinement step")
    parser.add_argument("--save-dir", default="data/align_test_output",
                        help="Directory to save output images")
    parser.add_argument("--severity", choices=["mild", "moderate", "heavy", "all"],
                        default="all",
                        help="Simulation severity (default: run all three)")
    args = parser.parse_args()

    severities = None
    if args.severity != "all":
        severities = [args.severity]

    run_alignment(
        img1_path=args.img1,
        img2_path=args.img2,
        use_ecc=not args.no_ecc,
        save_dir=args.save_dir,
        severities=severities,
    )


if __name__ == "__main__":
    main()
