"""Card image alignment using SIFT feature matching.

Aligns two photos of the same card (e.g. front/back, different angles) to a
common reference frame for pixel-level comparison. Primary use cases:

- Front/back alignment for condition assessment
- Oblique difference maps between scan and reference
- Defect heatmap generation

Pipeline:
  1. SIFT keypoint detection on both images
  2. FLANN-based feature matching with Lowe's ratio test
  3. Homography estimation with RANSAC
  4. Warp query image into reference frame
  5. Compute difference heatmap

Research findings (2026-03-06):
  - SIFT+RANSAC achieves sub-pixel accuracy (0.05-0.50 px RMSE)
  - Even low-detail cards (Energy) produce 284+ keypoints
  - ECC refinement NOT used -- confirmed to make alignment worse
  - Total processing time: ~140ms per alignment pair
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Alignment quality thresholds
MIN_KEYPOINTS = 10
MIN_GOOD_MATCHES = 10
MIN_HOMOGRAPHY_DET = 1e-6  # Reject degenerate homographies


@dataclass
class AlignmentResult:
    """Result of aligning a query image to a reference image."""

    aligned: np.ndarray
    """Query image warped into the reference frame (BGR)."""

    homography: np.ndarray
    """3x3 homography matrix that maps query coords to reference coords."""

    num_keypoints_query: int
    """Number of SIFT keypoints detected in the query image."""

    num_keypoints_ref: int
    """Number of SIFT keypoints detected in the reference image."""

    num_good_matches: int
    """Number of matches surviving Lowe's ratio test."""

    num_inliers: int
    """Number of RANSAC inliers."""

    success: bool
    """Whether alignment succeeded."""

    error: Optional[str] = None
    """Error message if alignment failed."""


def _detect_sift(gray: np.ndarray, nfeatures: int = 2000,
                 contrast_threshold: float = 0.04,
                 edge_threshold: int = 10):
    """Detect SIFT keypoints and compute descriptors on a grayscale image."""
    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
    )
    kps, descs = sift.detectAndCompute(gray, None)
    return kps, descs


def _match_flann(descs1: np.ndarray, descs2: np.ndarray,
                 ratio_threshold: float = 0.75):
    """Match descriptors using FLANN with Lowe's ratio test.

    Returns list of good DMatch objects.
    """
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    raw_matches = flann.knnMatch(descs1, descs2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good.append(m)

    return good


def _compute_homography(kps_query, kps_ref, good_matches,
                        ransac_thresh: float = 5.0):
    """Compute homography from query to reference using RANSAC.

    Returns (H, inlier_count) or (None, 0) on failure.
    H maps points in query image to corresponding points in reference image.
    """
    if len(good_matches) < 4:
        return None, 0

    src_pts = np.float32(
        [kps_query[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kps_ref[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    if H is None or mask is None:
        return None, 0

    inliers = int(mask.sum())
    return H, inliers


def _validate_homography(H: np.ndarray) -> Optional[str]:
    """Check that a homography matrix is geometrically valid.

    Returns an error message string if invalid, None if OK.
    """
    det = np.linalg.det(H)

    # Degenerate (near-singular)
    if abs(det) < MIN_HOMOGRAPHY_DET:
        return f"degenerate homography (det={det:.2e})"

    # Negative determinant means the transform includes a reflection,
    # which is physically impossible for card photos.
    if det < 0:
        return f"reflective homography (det={det:.4f})"

    return None


def align_cards(query_path: str, ref_path: str, *,
                nfeatures: int = 2000,
                ratio_threshold: float = 0.75,
                ransac_thresh: float = 5.0) -> AlignmentResult:
    """Align a query card image to a reference card image.

    Uses SIFT keypoint detection, FLANN matching with Lowe's ratio test,
    and RANSAC homography estimation.

    Args:
        query_path: Path to the query image (e.g. photo of a card).
        ref_path: Path to the reference image.
        nfeatures: Maximum SIFT keypoints to detect per image.
        ratio_threshold: Lowe's ratio test threshold (lower = stricter).
        ransac_thresh: RANSAC reprojection error threshold in pixels.

    Returns:
        AlignmentResult with the warped image, homography, and diagnostics.
    """
    fail = lambda msg: AlignmentResult(
        aligned=np.array([]),
        homography=np.eye(3),
        num_keypoints_query=0,
        num_keypoints_ref=0,
        num_good_matches=0,
        num_inliers=0,
        success=False,
        error=msg,
    )

    # Load images
    query_img = cv2.imread(query_path)
    ref_img = cv2.imread(ref_path)

    if query_img is None:
        return fail(f"could not load query image: {query_path}")
    if ref_img is None:
        return fail(f"could not load reference image: {ref_path}")

    return align_cards_images(
        query_img, ref_img,
        nfeatures=nfeatures,
        ratio_threshold=ratio_threshold,
        ransac_thresh=ransac_thresh,
    )


def align_cards_images(query_img: np.ndarray, ref_img: np.ndarray, *,
                       nfeatures: int = 2000,
                       ratio_threshold: float = 0.75,
                       ransac_thresh: float = 5.0) -> AlignmentResult:
    """Align a query card image (numpy array) to a reference card image.

    Same as align_cards() but accepts pre-loaded BGR images instead of paths.
    """
    fail = lambda msg, nkq=0, nkr=0, ngm=0: AlignmentResult(
        aligned=np.array([]),
        homography=np.eye(3),
        num_keypoints_query=nkq,
        num_keypoints_ref=nkr,
        num_good_matches=ngm,
        num_inliers=0,
        success=False,
        error=msg,
    )

    # Convert to grayscale
    gray_query = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # SIFT detection
    kps_query, descs_query = _detect_sift(gray_query, nfeatures=nfeatures)
    kps_ref, descs_ref = _detect_sift(gray_ref, nfeatures=nfeatures)

    nkq = len(kps_query) if kps_query else 0
    nkr = len(kps_ref) if kps_ref else 0

    if nkq < MIN_KEYPOINTS or nkr < MIN_KEYPOINTS:
        return fail(
            f"too few keypoints (query={nkq}, ref={nkr}, min={MIN_KEYPOINTS})",
            nkq, nkr,
        )

    if descs_query is None or descs_ref is None:
        return fail("SIFT produced no descriptors", nkq, nkr)

    # Feature matching
    good_matches = _match_flann(descs_query, descs_ref,
                                ratio_threshold=ratio_threshold)

    if len(good_matches) < MIN_GOOD_MATCHES:
        return fail(
            f"too few good matches ({len(good_matches)}, min={MIN_GOOD_MATCHES})",
            nkq, nkr, len(good_matches),
        )

    # Homography estimation
    H, inliers = _compute_homography(kps_query, kps_ref, good_matches,
                                     ransac_thresh=ransac_thresh)

    if H is None:
        return fail("homography estimation failed", nkq, nkr, len(good_matches))

    # Validate homography
    err = _validate_homography(H)
    if err is not None:
        return fail(err, nkq, nkr, len(good_matches))

    # Warp query into reference frame
    h, w = ref_img.shape[:2]
    aligned = cv2.warpPerspective(
        query_img, H, (w, h),
        borderMode=cv2.BORDER_REPLICATE,
    )

    logger.debug(
        "alignment: %d/%d keypoints, %d matches, %d inliers",
        nkq, nkr, len(good_matches), inliers,
    )

    return AlignmentResult(
        aligned=aligned,
        homography=H,
        num_keypoints_query=nkq,
        num_keypoints_ref=nkr,
        num_good_matches=len(good_matches),
        num_inliers=inliers,
        success=True,
    )


def compute_difference_map(aligned: np.ndarray,
                           reference: np.ndarray, *,
                           blur_ksize: int = 5,
                           colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Compute a pixel-level difference heatmap between aligned and reference.

    Both images must be the same size (the aligned image should come from
    align_cards / align_cards_images).

    Args:
        aligned: Query image already warped into the reference frame (BGR).
        reference: Reference image (BGR).
        blur_ksize: Gaussian blur kernel size applied to the difference
            before colormapping, to reduce noise. Set to 0 to disable.
        colormap: OpenCV colormap constant for the heatmap.

    Returns:
        BGR heatmap image where bright regions indicate large differences.
    """
    gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_aligned, gray_ref)

    # Optional smoothing to suppress per-pixel noise
    if blur_ksize > 0:
        diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)

    # Normalize to full 0-255 range for visualization
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    heatmap = cv2.applyColorMap(diff_norm, colormap)
    return heatmap
