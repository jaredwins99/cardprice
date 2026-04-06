#!/usr/bin/env python3
"""
Edge whitening detector prototype for Pokemon cards.

Detects wear on card borders by looking for white/bright pixels along
the card edges in LAB color space. Edge whitening occurs when the colored
border is worn, exposing the white paper layer underneath.

Traditional CV approach - no ML required.
"""

import cv2
import numpy as np
import glob
import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional


@dataclass
class EdgeResult:
    """Result for a single edge strip."""
    side: str           # top, bottom, left, right
    total_pixels: int
    white_pixels: int
    whitening_ratio: float
    mean_lightness: float
    max_lightness: float
    # Spatial: longest contiguous white run (in pixels)
    max_white_run: int
    # How many distinct white clusters
    cluster_count: int


@dataclass
class WhiteningReport:
    """Full whitening analysis for one card."""
    path: str
    image_shape: Tuple[int, int, int]
    edges: List[EdgeResult]
    overall_ratio: float
    severity: str       # none, light, moderate, heavy
    worst_edge: str
    worst_ratio: float
    # Per-edge severity
    edge_severities: Dict[str, str]


def extract_edge_strips(img: np.ndarray, strip_width: int = 30) -> Dict[str, np.ndarray]:
    """Extract 4 edge strips from a card image.

    For 1008x1530 images, strip_width=30 is ~3% of width.
    For smaller images, we scale proportionally.
    """
    h, w = img.shape[:2]

    # Scale strip width proportionally to image size
    # Reference: 30px for 1008-wide image = ~3%
    sw = max(10, int(strip_width * min(w, h) / 1008))

    strips = {
        'top':    img[0:sw, :],
        'bottom': img[h-sw:h, :],
        'left':   img[:, 0:sw],
        'right':  img[:, w-sw:w],
    }
    return strips


def analyze_strip(strip: np.ndarray, side: str,
                  L_threshold: float = 200,
                  saturation_threshold: float = 30) -> EdgeResult:
    """Analyze a single edge strip for whitening.

    Uses LAB color space:
    - L channel: lightness (0-255). High L = bright/white.
    - a, b channels: color. Near 128 = neutral/gray/white.

    White pixels: L > L_threshold AND low saturation in ab channels.

    We also check saturation in HSV to filter out bright-but-colored pixels
    (e.g., yellow energy cards have bright borders that aren't whitening).
    """
    lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

    L = lab[:, :, 0].astype(float)
    a = lab[:, :, 1].astype(float)
    b = lab[:, :, 2].astype(float)
    S_hsv = hsv[:, :, 1].astype(float)

    # White = high lightness + low color saturation
    # LAB: a,b near 128 = neutral
    ab_deviation = np.sqrt((a - 128)**2 + (b - 128)**2)

    # A pixel is "white" if:
    # 1. Very light (L > threshold)
    # 2. Low color (ab deviation small OR hsv saturation low)
    white_mask = (L > L_threshold) & (ab_deviation < saturation_threshold) & (S_hsv < 50)

    total = strip.shape[0] * strip.shape[1]
    white_count = int(np.sum(white_mask))
    ratio = white_count / total if total > 0 else 0.0

    # Compute longest contiguous white run along the edge direction
    # For top/bottom strips: project onto x-axis (columns)
    # For left/right strips: project onto y-axis (rows)
    if side in ('top', 'bottom'):
        projection = np.any(white_mask, axis=0).astype(int)
    else:
        projection = np.any(white_mask, axis=1).astype(int)

    max_run = _longest_run(projection)

    # Count distinct clusters using connected components
    white_mask_u8 = white_mask.astype(np.uint8) * 255
    n_labels, _ = cv2.connectedComponents(white_mask_u8)
    cluster_count = max(0, n_labels - 1)  # subtract background

    return EdgeResult(
        side=side,
        total_pixels=total,
        white_pixels=white_count,
        whitening_ratio=ratio,
        mean_lightness=float(np.mean(L)),
        max_lightness=float(np.max(L)),
        max_white_run=int(max_run),
        cluster_count=cluster_count,
    )


def _longest_run(arr: np.ndarray) -> int:
    """Find longest contiguous run of 1s in a 1D array."""
    if len(arr) == 0:
        return 0
    max_run = 0
    current = 0
    for v in arr:
        if v:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def classify_severity(ratio: float) -> str:
    """Map whitening ratio to severity label."""
    if ratio < 0.005:
        return 'none'
    elif ratio < 0.02:
        return 'light'
    elif ratio < 0.06:
        return 'moderate'
    else:
        return 'heavy'


def detect_whitening(img: np.ndarray, path: str = '',
                     strip_width: int = 30,
                     L_threshold: float = 200,
                     saturation_threshold: float = 30) -> WhiteningReport:
    """Full whitening analysis for a card image."""
    strips = extract_edge_strips(img, strip_width)

    edges = []
    total_white = 0
    total_pixels = 0

    for side, strip in strips.items():
        result = analyze_strip(strip, side, L_threshold, saturation_threshold)
        edges.append(result)
        total_white += result.white_pixels
        total_pixels += result.total_pixels

    overall_ratio = total_white / total_pixels if total_pixels > 0 else 0.0

    # Find worst edge
    worst = max(edges, key=lambda e: e.whitening_ratio)

    edge_severities = {e.side: classify_severity(e.whitening_ratio) for e in edges}

    return WhiteningReport(
        path=path,
        image_shape=img.shape,
        edges=edges,
        overall_ratio=overall_ratio,
        severity=classify_severity(overall_ratio),
        worst_edge=worst.side,
        worst_ratio=worst.whitening_ratio,
        edge_severities=edge_severities,
    )


def detect_whitening_adaptive(img: np.ndarray, path: str = '',
                               strip_width: int = 30) -> WhiteningReport:
    """Adaptive version that compares edge brightness to card interior.

    Instead of an absolute L threshold, we compute the interior's mean
    lightness and look for edge pixels significantly brighter than the
    border region (but not the card art area).
    """
    h, w = img.shape[:2]
    sw = max(10, int(strip_width * min(w, h) / 1008))

    # Border region: the outer ring (where colored border lives)
    # Slightly inward from the very edge to get the "intended" border color
    inner_margin = sw * 2
    border_ring = np.vstack([
        img[sw:inner_margin, sw:-sw].reshape(-1, 3),       # top inner border
        img[-inner_margin:-sw, sw:-sw].reshape(-1, 3),     # bottom inner border
        img[sw:-sw, sw:inner_margin].reshape(-1, 3),       # left inner border
        img[sw:-sw, -inner_margin:-sw].reshape(-1, 3),     # right inner border
    ])

    border_lab = cv2.cvtColor(border_ring.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)
    border_L_mean = float(np.mean(border_lab[0, :, 0]))

    # Adaptive threshold: if border is dark, lower threshold detects whitening better
    # If border is already light (e.g., yellow card), raise threshold to avoid false positives
    adaptive_L = max(180, min(230, border_L_mean + 80))

    report = detect_whitening(img, path, strip_width, L_threshold=adaptive_L)
    return report


def print_report(report: WhiteningReport) -> None:
    """Pretty-print a whitening report."""
    basename = os.path.basename(report.path)
    print(f"\n{'='*60}")
    print(f"Card: {basename}  Shape: {report.image_shape[:2]}")
    print(f"Overall: {report.overall_ratio:.4f} ({report.severity})")
    print(f"Worst edge: {report.worst_edge} ({report.worst_ratio:.4f})")
    print(f"{'─'*60}")
    print(f"{'Edge':<8} {'Ratio':>8} {'White px':>10} {'Total':>8} {'MaxRun':>8} {'Clusters':>10} {'Severity':<10}")
    for e in report.edges:
        print(f"{e.side:<8} {e.whitening_ratio:>8.4f} {e.white_pixels:>10} {e.total_pixels:>8} {e.max_white_run:>8} {e.cluster_count:>10} {classify_severity(e.whitening_ratio):<10}")


def run_test():
    """Run whitening detection on all available test cards."""
    os.chdir('/home/godli/cardprice')

    test_sets = [
        ("Test Binder Segments (630x880)", sorted(glob.glob('data/test_binder_segments/card_*.png'))),
        ("Inbox Hires Page 1 (1008x1530)", sorted(glob.glob('data/inbox/page_20260228_174819_cards_hires/card_*.png'))),
        ("Inbox Hires Page 2 (1008x1530)", sorted(glob.glob('data/inbox/page_20260228_195512_cards_hires/card_*.png'))),
        ("Inbox Hires Page 3 (1008x1530)", sorted(glob.glob('data/inbox/page_20260228_202134_cards_hires/card_*.png'))),
    ]

    all_reports = []

    for set_name, paths in test_sets:
        if not paths:
            continue
        print(f"\n{'#'*60}")
        print(f"# {set_name} ({len(paths)} cards)")
        print(f"{'#'*60}")

        for p in paths:
            img = cv2.imread(p)
            if img is None:
                print(f"  SKIP: {p}")
                continue

            # Run both fixed and adaptive
            report_fixed = detect_whitening(img, p)
            report_adaptive = detect_whitening_adaptive(img, p)

            print_report(report_fixed)

            # If adaptive differs significantly, note it
            if abs(report_fixed.overall_ratio - report_adaptive.overall_ratio) > 0.005:
                print(f"  [Adaptive: overall={report_adaptive.overall_ratio:.4f} "
                      f"({report_adaptive.severity})]")

            all_reports.append({
                'path': p,
                'shape': list(img.shape),
                'fixed': {
                    'overall_ratio': report_fixed.overall_ratio,
                    'severity': report_fixed.severity,
                    'worst_edge': report_fixed.worst_edge,
                    'worst_ratio': report_fixed.worst_ratio,
                    'edge_severities': report_fixed.edge_severities,
                    'edges': [asdict(e) for e in report_fixed.edges],
                },
                'adaptive': {
                    'overall_ratio': report_adaptive.overall_ratio,
                    'severity': report_adaptive.severity,
                    'worst_edge': report_adaptive.worst_edge,
                    'worst_ratio': report_adaptive.worst_ratio,
                    'edge_severities': report_adaptive.edge_severities,
                },
            })

    # Also test on a few reference card images (known clean/mint)
    ref_paths = sorted(glob.glob('data/card_images/*.png'))[:6]
    if ref_paths:
        print(f"\n{'#'*60}")
        print(f"# Reference Card Images (known clean, {len(ref_paths)} cards)")
        print(f"{'#'*60}")

        for p in ref_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            report = detect_whitening(img, p)
            print_report(report)
            all_reports.append({
                'path': p,
                'shape': list(img.shape),
                'fixed': {
                    'overall_ratio': report.overall_ratio,
                    'severity': report.severity,
                    'worst_edge': report.worst_edge,
                    'worst_ratio': report.worst_ratio,
                    'edge_severities': report.edge_severities,
                    'edges': [asdict(e) for e in report.edges],
                },
                'is_reference': True,
            })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Path':<65} {'Overall':>8} {'Severity':<10} {'Worst Edge':<12} {'Worst':>8}")
    for r in all_reports:
        f = r['fixed']
        basename = os.path.basename(r['path'])
        parent = os.path.basename(os.path.dirname(r['path']))
        label = f"{parent}/{basename}"
        ref = " (REF)" if r.get('is_reference') else ""
        print(f"{label+ref:<65} {f['overall_ratio']:>8.4f} {f['severity']:<10} {f['worst_edge']:<12} {f['worst_ratio']:>8.4f}")

    # Save JSON
    out_path = 'data/eval/edge_whitening_results.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_reports, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    run_test()
