#!/usr/bin/env python3
"""FFT frequency analysis to detect foil/holo patterns on binder scan cards.

Theory: Holographic foil creates regular repeating shimmer patterns that should
show up as peaks in the FFT spectrum. Normal matte cards have more uniform
frequency distributions.

Ground truth for Dragon Frontiers page (page_20260305_094228_cards):
  card_00, card_02 = reverse holo (stamped)
  card_05, card_08 = holofoil
  rest (01, 03, 04, 06, 07) = normal
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

# === Config ===
CARD_DIR = Path("/home/godli/cardprice/data/inbox/page_20260305_094228_cards")
OUT_DIR = Path("/home/godli/cardprice/data/condition_training/stamps_analysis/fft")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH = {
    "card_00": "reverse_holo",
    "card_01": "normal",
    "card_02": "reverse_holo",
    "card_03": "normal",
    "card_04": "normal",
    "card_05": "holofoil",
    "card_06": "normal",
    "card_07": "normal",
    "card_08": "holofoil",
}

LABEL_COLORS = {
    "normal": "#2196F3",
    "reverse_holo": "#FF9800",
    "holofoil": "#E91E63",
}


def load_card(name: str) -> np.ndarray:
    path = CARD_DIR / f"{name}.png"
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load {path}")
    return img


def get_regions(img: np.ndarray) -> dict:
    """Extract artwork, border, text, and stamp regions from a card image.

    Card layout (approximate proportions):
      Top 5%: top border
      5-55%: artwork region
      55-95%: text/attack region
      Bottom 5%: bottom border
      Left/Right 5%: side borders

    Stamp region (reverse holo): typically in artwork area, bottom-right quadrant
    """
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    regions = {}

    # Artwork: center of top half (avoiding borders)
    art_y1, art_y2 = int(h * 0.08), int(h * 0.48)
    art_x1, art_x2 = int(w * 0.08), int(w * 0.92)
    regions["artwork"] = gray[art_y1:art_y2, art_x1:art_x2]

    # Border: top strip + left strip + right strip
    border_top = gray[0:int(h * 0.06), int(w * 0.1):int(w * 0.9)]
    border_left = gray[int(h * 0.1):int(h * 0.9), 0:int(w * 0.06)]
    border_right = gray[int(h * 0.1):int(h * 0.9), int(w * 0.94):w]
    # Combine borders into one image by stacking
    max_w = max(border_top.shape[1], border_left.shape[1], border_right.shape[1])
    def pad_to_width(arr, target_w):
        if arr.shape[1] < target_w:
            return np.pad(arr, ((0, 0), (0, target_w - arr.shape[1])), mode='reflect')
        return arr[:, :target_w]
    regions["border"] = np.vstack([
        pad_to_width(border_top, max_w),
        pad_to_width(border_left, max_w),
        pad_to_width(border_right, max_w),
    ])

    # Text region: bottom half
    text_y1, text_y2 = int(h * 0.52), int(h * 0.92)
    text_x1, text_x2 = int(w * 0.08), int(w * 0.92)
    regions["text"] = gray[text_y1:text_y2, text_x1:text_x2]

    # Stamp region: for reverse holos, stamp is typically in bottom-right of artwork
    # Use a generous crop that should capture stamp if present
    stamp_y1, stamp_y2 = int(h * 0.30), int(h * 0.50)
    stamp_x1, stamp_x2 = int(w * 0.15), int(w * 0.85)
    regions["stamp"] = gray[stamp_y1:stamp_y2, stamp_x1:stamp_x2]

    # Full card (no border)
    regions["full"] = gray[int(h * 0.05):int(h * 0.95), int(w * 0.05):int(w * 0.95)]

    return regions


def compute_fft_features(region: np.ndarray) -> dict:
    """Compute FFT-based features from a grayscale region."""
    h, w = region.shape

    # Apply window function to reduce spectral leakage
    win_y = np.hanning(h)
    win_x = np.hanning(w)
    window = np.outer(win_y, win_x)
    windowed = region.astype(np.float64) * window

    # 2D FFT
    fft = np.fft.fft2(windowed)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    # Log magnitude for visualization
    log_magnitude = np.log1p(magnitude)

    # Power spectrum
    power = magnitude ** 2
    total_power = power.sum()

    if total_power == 0:
        return {
            "high_freq_ratio": 0,
            "mid_freq_ratio": 0,
            "low_freq_ratio": 0,
            "peak_magnitude": 0,
            "spectral_entropy": 0,
            "radial_profile": np.zeros(50),
            "log_magnitude": log_magnitude,
            "magnitude": magnitude,
            "directional_variance": 0,
            "high_freq_peaks": 0,
        }

    # Radial frequency analysis
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cy, cx)

    # Radial profile: average power at each radius
    n_bins = min(50, max_r)
    bin_edges = np.linspace(0, max_r, n_bins + 1)
    radial_profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if mask.any():
            radial_profile[i] = power[mask].mean()

    # Normalize radial profile
    if radial_profile.sum() > 0:
        radial_profile_norm = radial_profile / radial_profile.sum()
    else:
        radial_profile_norm = radial_profile

    # Frequency band ratios
    low_mask = r < max_r * 0.2
    mid_mask = (r >= max_r * 0.2) & (r < max_r * 0.6)
    high_mask = r >= max_r * 0.6

    low_freq_ratio = power[low_mask].sum() / total_power if total_power > 0 else 0
    mid_freq_ratio = power[mid_mask].sum() / total_power if total_power > 0 else 0
    high_freq_ratio = power[high_mask].sum() / total_power if total_power > 0 else 0

    # Peak magnitude (excluding DC component)
    mag_no_dc = magnitude.copy()
    mag_no_dc[cy - 2:cy + 3, cx - 2:cx + 3] = 0  # Zero out DC neighborhood
    peak_magnitude = mag_no_dc.max()

    # Count significant high-frequency peaks
    high_freq_mag = mag_no_dc.copy()
    high_freq_mag[r < max_r * 0.4] = 0
    threshold = np.percentile(high_freq_mag[high_freq_mag > 0], 99) if (high_freq_mag > 0).any() else 0
    high_freq_peaks = (high_freq_mag > threshold).sum()

    # Spectral entropy (measure of frequency spread)
    power_norm = power / total_power
    power_norm = power_norm[power_norm > 0]
    spectral_entropy = -np.sum(power_norm * np.log2(power_norm))

    # Directional variance: check if energy is concentrated in specific directions
    # Compute angular distribution
    angles = np.arctan2(Y - cy, X - cx)
    n_angle_bins = 36
    angle_bins = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
    angular_power = np.zeros(n_angle_bins)
    for i in range(n_angle_bins):
        mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1]) & (r > max_r * 0.1)
        if mask.any():
            angular_power[i] = power[mask].mean()
    if angular_power.sum() > 0:
        angular_power_norm = angular_power / angular_power.sum()
        directional_variance = np.var(angular_power_norm)
    else:
        directional_variance = 0

    return {
        "high_freq_ratio": high_freq_ratio,
        "mid_freq_ratio": mid_freq_ratio,
        "low_freq_ratio": low_freq_ratio,
        "peak_magnitude": peak_magnitude,
        "spectral_entropy": spectral_entropy,
        "radial_profile": radial_profile_norm,
        "log_magnitude": log_magnitude,
        "magnitude": magnitude,
        "directional_variance": directional_variance,
        "high_freq_peaks": high_freq_peaks,
    }


def compute_texture_features(region: np.ndarray) -> dict:
    """Additional texture-based features that complement FFT."""
    # Local variance (texture roughness)
    local_mean = cv2.blur(region.astype(np.float64), (11, 11))
    local_var = cv2.blur((region.astype(np.float64) - local_mean) ** 2, (11, 11))

    # Gradient magnitude (edge density)
    gx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Laplacian variance (focus/detail measure)
    lap = cv2.Laplacian(region, cv2.CV_64F)

    return {
        "local_var_mean": local_var.mean(),
        "local_var_std": local_var.std(),
        "gradient_mean": grad_mag.mean(),
        "gradient_std": grad_mag.std(),
        "laplacian_var": lap.var(),
        "intensity_mean": region.mean(),
        "intensity_std": region.std(),
    }


def plot_fft_spectra_comparison(all_features: dict, region_name: str):
    """Plot FFT magnitude spectra side by side for 3 classes."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(f"FFT Magnitude Spectra — {region_name.upper()} Region", fontsize=16, fontweight='bold')

    # Group cards by class
    groups = {"normal": [], "reverse_holo": [], "holofoil": []}
    for card_name, feats in all_features.items():
        label = GROUND_TRUTH[card_name]
        groups[label].append((card_name, feats))

    for col_idx, (label, cards) in enumerate(groups.items()):
        color = LABEL_COLORS[label]

        # Row 0: FFT magnitude spectrum (show first card of each class)
        card_name, feats = cards[0]
        log_mag = feats[region_name]["log_magnitude"]
        ax = axes[0, col_idx]
        im = ax.imshow(log_mag, cmap='magma', aspect='auto')
        ax.set_title(f"{label}\n({card_name})", color=color, fontweight='bold')
        ax.set_xlabel("Frequency X")
        ax.set_ylabel("Frequency Y")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Row 1: Radial frequency profile (all cards of this class)
        ax = axes[1, col_idx]
        for card_name, feats in cards:
            profile = feats[region_name]["radial_profile"]
            ax.plot(profile, label=card_name, alpha=0.7)
        ax.set_title(f"Radial Frequency Profile — {label}", fontweight='bold')
        ax.set_xlabel("Frequency bin (low → high)")
        ax.set_ylabel("Normalized power")
        ax.legend(fontsize=8)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Row 2: Bar chart of key features (all cards)
        ax = axes[2, col_idx]
        feature_names = ["high_freq_ratio", "mid_freq_ratio", "spectral_entropy"]
        x = np.arange(len(feature_names))
        width = 0.8 / len(cards)
        for i, (card_name, feats) in enumerate(cards):
            vals = [feats[region_name][f] for f in feature_names]
            # Normalize entropy to [0,1] range for display
            vals[2] = vals[2] / 25.0  # typical entropy range
            ax.bar(x + i * width, vals, width, label=card_name, alpha=0.7)
        ax.set_title(f"Key Features — {label}", fontweight='bold')
        ax.set_xticks(x + width * len(cards) / 2)
        ax.set_xticklabels(["High freq\nratio", "Mid freq\nratio", "Spectral\nentropy/25"])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"fft_spectra_{region_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fft_spectra_{region_name}.png")


def plot_class_separation(all_features: dict):
    """Plot feature distributions to see if classes are separable."""
    regions = ["artwork", "border", "text", "stamp", "full"]
    fft_features = ["high_freq_ratio", "mid_freq_ratio", "spectral_entropy",
                     "directional_variance", "high_freq_peaks"]
    tex_features = ["local_var_mean", "gradient_mean", "laplacian_var", "intensity_std"]

    # Collect data
    data = {label: {r: {f: [] for f in fft_features + tex_features} for r in regions}
            for label in ["normal", "reverse_holo", "holofoil"]}

    for card_name, feats in all_features.items():
        if card_name.startswith("_"):
            continue
        label = GROUND_TRUTH[card_name]
        for r in regions:
            for f in fft_features:
                data[label][r][f].append(feats[r][f])

    # Plot: one subplot per region, showing feature distributions
    fig, axes = plt.subplots(len(regions), len(fft_features), figsize=(24, 20))
    fig.suptitle("FFT Feature Distributions by Region and Class", fontsize=16, fontweight='bold')

    for row, region in enumerate(regions):
        for col, feat in enumerate(fft_features):
            ax = axes[row, col]
            for label in ["normal", "reverse_holo", "holofoil"]:
                vals = data[label][region][feat]
                color = LABEL_COLORS[label]
                ax.scatter([label[:6]] * len(vals), vals, c=color, s=60,
                          alpha=0.7, edgecolors='black', linewidth=0.5, label=label)
            if row == 0:
                ax.set_title(feat.replace("_", " ").title(), fontweight='bold', fontsize=9)
            if col == 0:
                ax.set_ylabel(region.upper(), fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=7)
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_separation_fft.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: class_separation_fft.png")


def plot_stamp_analysis(all_features: dict):
    """Specific analysis of stamp region for reverse holo detection."""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 9, figure=fig, hspace=0.4, wspace=0.4)
    fig.suptitle("Stamp Region FFT Analysis\n(Reverse holo stamps should show periodic text patterns)",
                 fontsize=14, fontweight='bold')

    card_names = sorted(all_features.keys())

    for idx, card_name in enumerate(card_names):
        feats = all_features[card_name]
        label = GROUND_TRUTH[card_name]
        color = LABEL_COLORS[label]

        # Row 0: Stamp region image
        ax = fig.add_subplot(gs[0, idx])
        # We need the actual stamp crop, store it
        stamp_img = feats["_stamp_img"]
        ax.imshow(stamp_img, cmap='gray')
        ax.set_title(f"{card_name}\n{label}", fontsize=8, color=color, fontweight='bold')
        ax.axis('off')

        # Row 1: FFT magnitude
        ax = fig.add_subplot(gs[1, idx])
        log_mag = feats["stamp"]["log_magnitude"]
        ax.imshow(log_mag, cmap='magma', aspect='auto')
        ax.set_title("FFT", fontsize=8)
        ax.axis('off')

        # Row 2: Radial profile
        ax = fig.add_subplot(gs[2, idx])
        profile = feats["stamp"]["radial_profile"]
        ax.plot(profile, color=color, linewidth=1.5)
        ax.set_title(f"HF={feats['stamp']['high_freq_ratio']:.4f}", fontsize=7)
        ax.set_yscale('log')
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    plt.savefig(OUT_DIR / "stamp_fft_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: stamp_fft_analysis.png")


def plot_best_features_scatter(all_features: dict):
    """Find and plot the most discriminative 2D feature pairs."""
    # Collect all scalar features for all cards
    regions = ["artwork", "border", "stamp", "full"]
    scalar_feats = ["high_freq_ratio", "mid_freq_ratio", "spectral_entropy",
                    "directional_variance", "high_freq_peaks", "peak_magnitude"]

    # Build feature matrix
    card_names = sorted(all_features.keys())
    labels = [GROUND_TRUTH[c] for c in card_names]

    feature_vecs = {}
    for region in regions:
        for feat in scalar_feats:
            key = f"{region}_{feat}"
            vals = [all_features[c][region][feat] for c in card_names]
            feature_vecs[key] = np.array(vals)

    # Also add texture features
    tex_feats = ["local_var_mean", "gradient_mean", "laplacian_var", "intensity_std"]
    for region in regions:
        for feat in tex_feats:
            key = f"{region}_tex_{feat}"
            vals = [all_features[c].get(f"{region}_tex", {}).get(feat, 0) for c in card_names]
            feature_vecs[key] = np.array(vals)

    # Find best separating features using simple class separation metric
    def separation_score(vals, labels):
        """Higher = better separation between classes."""
        classes = set(labels)
        if len(classes) < 2:
            return 0
        means = {}
        for c in classes:
            c_vals = [v for v, l in zip(vals, labels) if l == c]
            if len(c_vals) == 0:
                return 0
            means[c] = np.mean(c_vals)

        # Between-class variance / within-class variance
        overall_mean = np.mean(vals)
        between = sum(len([l for l in labels if l == c]) * (means[c] - overall_mean) ** 2 for c in classes)
        within = sum(sum((v - means[l]) ** 2 for v, l in zip(vals, labels) if l == c) for c in classes)
        if within == 0:
            return float('inf')
        return between / (within + 1e-10)

    scores = {k: separation_score(v, labels) for k, v in feature_vecs.items()}
    top_features = sorted(scores, key=scores.get, reverse=True)[:8]

    print("\n=== Top 8 Most Discriminative Features ===")
    for f in top_features:
        print(f"  {f}: separation={scores[f]:.4f}")
        for label in ["normal", "reverse_holo", "holofoil"]:
            vals = [v for v, l in zip(feature_vecs[f], labels) if l == label]
            if vals:
                print(f"    {label}: mean={np.mean(vals):.6f}, std={np.std(vals):.6f}")

    # Plot top 4 pairs
    n_plots = min(4, len(top_features) // 2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Best Separating Feature Pairs (FFT + Texture)", fontsize=14, fontweight='bold')

    for plot_idx in range(n_plots):
        ax = axes[plot_idx // 2, plot_idx % 2]
        f1 = top_features[plot_idx * 2]
        f2 = top_features[plot_idx * 2 + 1]

        for label in ["normal", "reverse_holo", "holofoil"]:
            mask = [l == label for l in labels]
            x = feature_vecs[f1][mask]
            y = feature_vecs[f2][mask]
            ax.scatter(x, y, c=LABEL_COLORS[label], s=100, label=label,
                      edgecolors='black', linewidth=0.5, alpha=0.8)
            # Label each point
            for i, (xi, yi) in enumerate(zip(x, y)):
                cn = [c for c, l in zip(card_names, labels) if l == label][i]
                ax.annotate(cn[-2:], (xi, yi), fontsize=7, ha='center', va='bottom')

        ax.set_xlabel(f1.replace("_", " "), fontsize=9)
        ax.set_ylabel(f2.replace("_", " "), fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "best_feature_pairs.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: best_feature_pairs.png")


def main():
    print("=" * 60)
    print("FFT Foil/Holo Analysis — Dragon Frontiers Cards")
    print("=" * 60)

    all_features = {}

    for card_name in sorted(GROUND_TRUTH.keys()):
        label = GROUND_TRUTH[card_name]
        print(f"\n--- {card_name} ({label}) ---")

        img = load_card(card_name)
        regions = get_regions(img)

        card_feats = {}
        for region_name, region_img in regions.items():
            fft_feats = compute_fft_features(region_img)
            card_feats[region_name] = fft_feats

            tex_feats = compute_texture_features(region_img)
            card_feats[region_name + "_tex"] = tex_feats

            print(f"  {region_name:10s}: HF_ratio={fft_feats['high_freq_ratio']:.5f}, "
                  f"MF_ratio={fft_feats['mid_freq_ratio']:.5f}, "
                  f"entropy={fft_feats['spectral_entropy']:.2f}, "
                  f"dir_var={fft_feats['directional_variance']:.6f}, "
                  f"HF_peaks={fft_feats['high_freq_peaks']}")

        # Store stamp image for visualization
        card_feats["_stamp_img"] = regions["stamp"]

        all_features[card_name] = card_feats

    # === Summary table ===
    print("\n" + "=" * 80)
    print("SUMMARY: Key Features by Class")
    print("=" * 80)

    for region_name in ["artwork", "border", "stamp", "full"]:
        print(f"\n--- {region_name.upper()} Region ---")
        print(f"{'Card':<10} {'Label':<15} {'HF_ratio':>10} {'MF_ratio':>10} {'Entropy':>10} {'DirVar':>12} {'HF_peaks':>10} {'GradMean':>10} {'LaplacVar':>10}")
        print("-" * 100)

        for card_name in sorted(all_features.keys()):
            label = GROUND_TRUTH[card_name]
            ff = all_features[card_name][region_name]
            tf = all_features[card_name][region_name + "_tex"]
            print(f"{card_name:<10} {label:<15} {ff['high_freq_ratio']:>10.5f} "
                  f"{ff['mid_freq_ratio']:>10.5f} {ff['spectral_entropy']:>10.2f} "
                  f"{ff['directional_variance']:>12.6f} {ff['high_freq_peaks']:>10} "
                  f"{tf['gradient_mean']:>10.2f} {tf['laplacian_var']:>10.1f}")

        # Class averages
        print()
        for label in ["normal", "reverse_holo", "holofoil"]:
            cards = [c for c in all_features if GROUND_TRUTH[c] == label]
            hf = np.mean([all_features[c][region_name]["high_freq_ratio"] for c in cards])
            mf = np.mean([all_features[c][region_name]["mid_freq_ratio"] for c in cards])
            ent = np.mean([all_features[c][region_name]["spectral_entropy"] for c in cards])
            dv = np.mean([all_features[c][region_name]["directional_variance"] for c in cards])
            hp = np.mean([all_features[c][region_name]["high_freq_peaks"] for c in cards])
            gm = np.mean([all_features[c][region_name + "_tex"]["gradient_mean"] for c in cards])
            lv = np.mean([all_features[c][region_name + "_tex"]["laplacian_var"] for c in cards])
            print(f"{'AVG':<10} {label:<15} {hf:>10.5f} {mf:>10.5f} {ent:>10.2f} "
                  f"{dv:>12.6f} {hp:>10.0f} {gm:>10.2f} {lv:>10.1f}")

    # === Generate plots ===
    print("\n\nGenerating plots...")

    for region_name in ["artwork", "border", "stamp", "full"]:
        plot_fft_spectra_comparison(all_features, region_name)

    plot_class_separation(all_features)
    plot_stamp_analysis(all_features)
    plot_best_features_scatter(all_features)

    # === Verdict ===
    print("\n" + "=" * 60)
    print("ANALYSIS VERDICT")
    print("=" * 60)

    # Check if any feature cleanly separates classes
    for region_name in ["artwork", "border", "stamp", "full"]:
        for feat in ["high_freq_ratio", "mid_freq_ratio", "spectral_entropy", "directional_variance"]:
            normal_vals = [all_features[c][region_name][feat]
                          for c in all_features if GROUND_TRUTH[c] == "normal"]
            rholo_vals = [all_features[c][region_name][feat]
                         for c in all_features if GROUND_TRUTH[c] == "reverse_holo"]
            holo_vals = [all_features[c][region_name][feat]
                        for c in all_features if GROUND_TRUTH[c] == "holofoil"]

            # Check for clean separation (no overlap between ranges)
            n_range = (min(normal_vals), max(normal_vals))
            r_range = (min(rholo_vals), max(rholo_vals))
            h_range = (min(holo_vals), max(holo_vals))

            # Normal vs holo separation
            sep_n_h = min(h_range) > max(n_range) or max(h_range) < min(n_range)
            sep_n_r = min(r_range) > max(n_range) or max(r_range) < min(n_range)
            sep_r_h = min(h_range) > max(r_range) or max(h_range) < min(r_range)

            if sep_n_h or sep_n_r or sep_r_h:
                seps = []
                if sep_n_h: seps.append("normal-vs-holo")
                if sep_n_r: seps.append("normal-vs-reverse")
                if sep_r_h: seps.append("reverse-vs-holo")
                print(f"  CLEAN SEPARATION: {region_name}.{feat} separates {', '.join(seps)}")
                print(f"    normal:  [{n_range[0]:.6f}, {n_range[1]:.6f}]")
                print(f"    r_holo:  [{r_range[0]:.6f}, {r_range[1]:.6f}]")
                print(f"    holo:    [{h_range[0]:.6f}, {h_range[1]:.6f}]")

    # Also check texture features
    for region_name in ["artwork", "border", "stamp", "full"]:
        for feat in ["local_var_mean", "gradient_mean", "laplacian_var", "intensity_std"]:
            normal_vals = [all_features[c][region_name + "_tex"][feat]
                          for c in all_features if GROUND_TRUTH[c] == "normal"]
            rholo_vals = [all_features[c][region_name + "_tex"][feat]
                         for c in all_features if GROUND_TRUTH[c] == "reverse_holo"]
            holo_vals = [all_features[c][region_name + "_tex"][feat]
                        for c in all_features if GROUND_TRUTH[c] == "holofoil"]

            n_range = (min(normal_vals), max(normal_vals))
            r_range = (min(rholo_vals), max(rholo_vals))
            h_range = (min(holo_vals), max(holo_vals))

            sep_n_h = min(h_range) > max(n_range) or max(h_range) < min(n_range)
            sep_n_r = min(r_range) > max(n_range) or max(r_range) < min(n_range)
            sep_r_h = min(h_range) > max(r_range) or max(h_range) < min(r_range)

            if sep_n_h or sep_n_r or sep_r_h:
                seps = []
                if sep_n_h: seps.append("normal-vs-holo")
                if sep_n_r: seps.append("normal-vs-reverse")
                if sep_r_h: seps.append("reverse-vs-holo")
                print(f"  CLEAN SEPARATION: {region_name}_tex.{feat} separates {', '.join(seps)}")
                print(f"    normal:  [{n_range[0]:.6f}, {n_range[1]:.6f}]")
                print(f"    r_holo:  [{r_range[0]:.6f}, {r_range[1]:.6f}]")
                print(f"    holo:    [{h_range[0]:.6f}, {h_range[1]:.6f}]")

    print("\nDone! All plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
