#!/usr/bin/env python3
"""
Stamp detection via color histogram analysis of the stamp region.

Theory: EX-era stamps are metallic gold/silver. Even through a binder sleeve,
the stamp shifts the color distribution — higher brightness, specific hue range
(gold ~30-50 in HSV), different saturation patterns.

For each card:
  - Crop stamp region (x=[0.55,0.92], y=[0.40,0.68])
  - Crop control region (top-left artwork, x=[0.08,0.45], y=[0.15,0.43])
  - Compare HSV distributions between stamp and control regions
  - Compute gold-pixel and metallic-pixel ratios
  - Build threshold classifier and report separability

Output: histogram plots + accuracy report in
  data/condition_training/stamps_analysis/color/
"""

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/home/godli/cardprice")
GT_PATH = BASE / "data/condition_training/stamps_real/binder_ground_truth.jsonl"
INBOX = BASE / "data/inbox"
OUT_DIR = BASE / "data/condition_training/stamps_analysis/color"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Region cropping ──────────────────────────────────────────────────

def crop_stamp_region(img: np.ndarray) -> np.ndarray:
    """Stamp region: bottom-right of artwork (x=[0.55,0.92], y=[0.40,0.68])."""
    h, w = img.shape[:2]
    return img[int(h * 0.40):int(h * 0.68), int(w * 0.55):int(w * 0.92)]


def crop_control_region(img: np.ndarray) -> np.ndarray:
    """Control region: top-left of artwork (x=[0.08,0.45], y=[0.15,0.43])."""
    h, w = img.shape[:2]
    return img[int(h * 0.15):int(h * 0.43), int(w * 0.08):int(w * 0.45)]


# ── Feature extraction ───────────────────────────────────────────────

def hsv_histograms(region_bgr: np.ndarray):
    """Return H, S, V channel histograms (normalized)."""
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()
    # Normalize to probability distributions
    h_hist /= (h_hist.sum() + 1e-10)
    s_hist /= (s_hist.sum() + 1e-10)
    v_hist /= (v_hist.sum() + 1e-10)
    return h_hist, s_hist, v_hist


def compute_color_features(img: np.ndarray) -> dict:
    """Extract color-based features from stamp vs control regions."""
    stamp = crop_stamp_region(img)
    control = crop_control_region(img)
    stamp_hsv = cv2.cvtColor(stamp, cv2.COLOR_BGR2HSV)
    control_hsv = cv2.cvtColor(control, cv2.COLOR_BGR2HSV)

    sh, ss, sv = stamp_hsv[:, :, 0], stamp_hsv[:, :, 1], stamp_hsv[:, :, 2]
    ch, cs, cv_ = control_hsv[:, :, 0], control_hsv[:, :, 1], control_hsv[:, :, 2]

    n_stamp = stamp_hsv.shape[0] * stamp_hsv.shape[1]
    n_ctrl = control_hsv.shape[0] * control_hsv.shape[1]

    # Gold pixel ratio: H in [20,50] (OpenCV 0-180 = 0-360), S > 50, V > 100
    # Note: OpenCV hue is 0-180, so [20,50] corresponds to ~40-100 degrees
    gold_stamp = np.sum((sh >= 10) & (sh <= 25) & (ss > 50) & (sv > 100))
    gold_control = np.sum((ch >= 10) & (ch <= 25) & (cs > 50) & (cv_ > 100))

    # Broader gold: H in [10,30] to catch warm yellows through binder sleeve
    gold_broad_stamp = np.sum((sh >= 8) & (sh <= 35) & (ss > 40) & (sv > 80))
    gold_broad_control = np.sum((ch >= 8) & (ch <= 35) & (cs > 40) & (cv_ > 80))

    # Metallic ratio: low saturation + high brightness (silver/white metallic sheen)
    metallic_stamp = np.sum((ss < 30) & (sv > 150))
    metallic_control = np.sum((cs < 30) & (cv_ > 150))

    # High-brightness low-sat (catches both gold and silver metallic)
    bright_stamp = np.sum(sv > 180)
    bright_control = np.sum(cv_ > 180)

    # Saturation statistics
    sat_mean_stamp = float(np.mean(ss))
    sat_mean_control = float(np.mean(cs))
    sat_std_stamp = float(np.std(ss))
    sat_std_control = float(np.std(cs))

    # Value (brightness) statistics
    val_mean_stamp = float(np.mean(sv))
    val_mean_control = float(np.mean(cv_))
    val_std_stamp = float(np.std(sv))
    val_std_control = float(np.std(cv_))

    # Hue statistics
    hue_mean_stamp = float(np.mean(sh))
    hue_mean_control = float(np.mean(ch))
    hue_std_stamp = float(np.std(sh))

    # Histogram comparison (chi-squared distance)
    sh_hist, ss_hist, sv_hist = hsv_histograms(stamp)
    ch_hist, cs_hist, cv_hist = hsv_histograms(control)
    hue_chi2 = float(cv2.compareHist(
        sh_hist.astype(np.float32), ch_hist.astype(np.float32),
        cv2.HISTCMP_CHISQR))
    sat_chi2 = float(cv2.compareHist(
        ss_hist.astype(np.float32), cs_hist.astype(np.float32),
        cv2.HISTCMP_CHISQR))
    val_chi2 = float(cv2.compareHist(
        sv_hist.astype(np.float32), cv_hist.astype(np.float32),
        cv2.HISTCMP_CHISQR))

    # Bhattacharyya distance (0=identical, 1=completely different)
    hue_bhat = float(cv2.compareHist(
        sh_hist.astype(np.float32), ch_hist.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA))
    sat_bhat = float(cv2.compareHist(
        ss_hist.astype(np.float32), cs_hist.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA))
    val_bhat = float(cv2.compareHist(
        sv_hist.astype(np.float32), cv_hist.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA))

    return {
        "gold_ratio_stamp": gold_stamp / n_stamp,
        "gold_ratio_control": gold_control / n_ctrl,
        "gold_ratio_diff": gold_stamp / n_stamp - gold_control / n_ctrl,
        "gold_broad_ratio_stamp": gold_broad_stamp / n_stamp,
        "gold_broad_ratio_control": gold_broad_control / n_ctrl,
        "gold_broad_diff": gold_broad_stamp / n_stamp - gold_broad_control / n_ctrl,
        "metallic_ratio_stamp": metallic_stamp / n_stamp,
        "metallic_ratio_control": metallic_control / n_ctrl,
        "metallic_diff": metallic_stamp / n_stamp - metallic_control / n_ctrl,
        "bright_ratio_stamp": bright_stamp / n_stamp,
        "bright_ratio_control": bright_control / n_ctrl,
        "bright_diff": bright_stamp / n_stamp - bright_control / n_ctrl,
        "sat_mean_stamp": sat_mean_stamp,
        "sat_mean_control": sat_mean_control,
        "sat_diff": sat_mean_stamp - sat_mean_control,
        "sat_std_stamp": sat_std_stamp,
        "sat_std_control": sat_std_control,
        "val_mean_stamp": val_mean_stamp,
        "val_mean_control": val_mean_control,
        "val_diff": val_mean_stamp - val_mean_control,
        "val_std_stamp": val_std_stamp,
        "val_std_control": val_std_control,
        "hue_mean_stamp": hue_mean_stamp,
        "hue_mean_control": hue_mean_control,
        "hue_std_stamp": hue_std_stamp,
        "hue_chi2": hue_chi2,
        "sat_chi2": sat_chi2,
        "val_chi2": val_chi2,
        "hue_bhat": hue_bhat,
        "sat_bhat": sat_bhat,
        "val_bhat": val_bhat,
        # Raw histograms for plotting
        "_stamp_h_hist": sh_hist,
        "_stamp_s_hist": ss_hist,
        "_stamp_v_hist": sv_hist,
        "_control_h_hist": ch_hist,
        "_control_s_hist": cs_hist,
        "_control_v_hist": cv_hist,
    }


# ── Data loading ─────────────────────────────────────────────────────

def load_ground_truth():
    """Load binder ground truth, dedup by image (last entry wins)."""
    entries = {}
    with open(GT_PATH) as f:
        for line in f:
            entry = json.loads(line.strip())
            entries[entry["image"]] = entry  # last wins for dupes
    results = []
    for entry in entries.values():
        img_path = INBOX / entry["image"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARN: missing {img_path}")
            continue
        results.append({
            "img": img,
            "stamped": entry["stamped"],
            "name": entry.get("card_name", "?"),
            "variant": entry.get("variant", "?"),
            "image": entry["image"],
        })
    return results


# ── Threshold classifier ─────────────────────────────────────────────

def find_best_threshold(values, labels):
    """Find threshold that maximizes accuracy. Returns (threshold, direction, acc)."""
    sorted_vals = sorted(set(values))
    best_acc, best_t, best_dir = 0, 0, ">"
    for t in sorted_vals:
        # stamped > t
        tp = sum(1 for v, l in zip(values, labels) if v > t and l)
        tn = sum(1 for v, l in zip(values, labels) if v <= t and not l)
        acc = (tp + tn) / len(labels)
        if acc > best_acc:
            best_acc, best_t, best_dir = acc, t, ">"
        # stamped < t
        tp2 = sum(1 for v, l in zip(values, labels) if v < t and l)
        tn2 = sum(1 for v, l in zip(values, labels) if v >= t and not l)
        acc2 = (tp2 + tn2) / len(labels)
        if acc2 > best_acc:
            best_acc, best_t, best_dir = acc2, t, "<"
    return best_t, best_dir, best_acc


# ── Visualization ────────────────────────────────────────────────────

def plot_hue_histograms(cards, features_list):
    """Plot average hue histograms: stamped vs clean, stamp region vs control."""
    stamped_stamp_h = []
    stamped_ctrl_h = []
    clean_stamp_h = []
    clean_ctrl_h = []

    for card, feat in zip(cards, features_list):
        if card["stamped"]:
            stamped_stamp_h.append(feat["_stamp_h_hist"])
            stamped_ctrl_h.append(feat["_control_h_hist"])
        else:
            clean_stamp_h.append(feat["_stamp_h_hist"])
            clean_ctrl_h.append(feat["_control_h_hist"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Hue Histograms: Stamp Region vs Control Region", fontsize=14)
    hue_range = np.arange(180)

    def plot_avg(ax, hists, title, color):
        if not hists:
            ax.set_title(f"{title} (no data)")
            return
        avg = np.mean(hists, axis=0)
        std = np.std(hists, axis=0)
        ax.fill_between(hue_range, avg - std, avg + std, alpha=0.3, color=color)
        ax.plot(hue_range, avg, color=color, linewidth=1.5)
        for h in hists:
            ax.plot(hue_range, h, color=color, alpha=0.15, linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("Hue (0-180)")
        ax.set_ylabel("Density")
        # Mark gold range
        ax.axvspan(10, 25, alpha=0.1, color="gold", label="Gold hue range")
        ax.legend(fontsize=8)

    plot_avg(axes[0, 0], stamped_stamp_h, f"STAMPED - Stamp Region (n={len(stamped_stamp_h)})", "red")
    plot_avg(axes[0, 1], stamped_ctrl_h, f"STAMPED - Control Region (n={len(stamped_ctrl_h)})", "darkred")
    plot_avg(axes[1, 0], clean_stamp_h, f"CLEAN - Stamp Region (n={len(clean_stamp_h)})", "green")
    plot_avg(axes[1, 1], clean_ctrl_h, f"CLEAN - Control Region (n={len(clean_ctrl_h)})", "darkgreen")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "hue_histograms.png", dpi=150)
    plt.close()
    print(f"  Saved hue_histograms.png")


def plot_saturation_histograms(cards, features_list):
    """Plot saturation distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Saturation Distribution: Stamp Region", fontsize=14)
    sat_range = np.arange(256)

    stamped_s, clean_s = [], []
    for card, feat in zip(cards, features_list):
        if card["stamped"]:
            stamped_s.append(feat["_stamp_s_hist"])
        else:
            clean_s.append(feat["_stamp_s_hist"])

    if stamped_s:
        avg = np.mean(stamped_s, axis=0)
        axes[0].plot(sat_range, avg, "r-", linewidth=1.5, label="Stamped avg")
        for h in stamped_s:
            axes[0].plot(sat_range, h, "r-", alpha=0.15, linewidth=0.5)
    if clean_s:
        avg = np.mean(clean_s, axis=0)
        axes[0].plot(sat_range, avg, "g-", linewidth=1.5, label="Clean avg")
        for h in clean_s:
            axes[0].plot(sat_range, h, "g-", alpha=0.15, linewidth=0.5)
    axes[0].set_title("Stamp Region Saturation")
    axes[0].set_xlabel("Saturation")
    axes[0].legend()

    # Control region
    stamped_sc, clean_sc = [], []
    for card, feat in zip(cards, features_list):
        if card["stamped"]:
            stamped_sc.append(feat["_control_s_hist"])
        else:
            clean_sc.append(feat["_control_s_hist"])

    if stamped_sc:
        avg = np.mean(stamped_sc, axis=0)
        axes[1].plot(sat_range, avg, "r-", linewidth=1.5, label="Stamped avg")
    if clean_sc:
        avg = np.mean(clean_sc, axis=0)
        axes[1].plot(sat_range, avg, "g-", linewidth=1.5, label="Clean avg")
    axes[1].set_title("Control Region Saturation")
    axes[1].set_xlabel("Saturation")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "saturation_histograms.png", dpi=150)
    plt.close()
    print(f"  Saved saturation_histograms.png")


def plot_value_histograms(cards, features_list):
    """Plot brightness (V channel) distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Brightness (Value) Distribution", fontsize=14)
    v_range = np.arange(256)

    stamped_v, clean_v = [], []
    stamped_vc, clean_vc = [], []
    for card, feat in zip(cards, features_list):
        if card["stamped"]:
            stamped_v.append(feat["_stamp_v_hist"])
            stamped_vc.append(feat["_control_v_hist"])
        else:
            clean_v.append(feat["_stamp_v_hist"])
            clean_vc.append(feat["_control_v_hist"])

    for data, ax, title in [
        ((stamped_v, clean_v), axes[0], "Stamp Region Brightness"),
        ((stamped_vc, clean_vc), axes[1], "Control Region Brightness"),
    ]:
        s_data, c_data = data
        if s_data:
            avg = np.mean(s_data, axis=0)
            ax.plot(v_range, avg, "r-", linewidth=1.5, label="Stamped avg")
            for h in s_data:
                ax.plot(v_range, h, "r-", alpha=0.15, linewidth=0.5)
        if c_data:
            avg = np.mean(c_data, axis=0)
            ax.plot(v_range, avg, "g-", linewidth=1.5, label="Clean avg")
            for h in c_data:
                ax.plot(v_range, h, "g-", alpha=0.15, linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("Value (Brightness)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "brightness_histograms.png", dpi=150)
    plt.close()
    print(f"  Saved brightness_histograms.png")


def plot_feature_scatter(cards, features_list, feat_names_pairs):
    """Scatter plot of feature pairs, colored by label."""
    n_pairs = len(feat_names_pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]
    fig.suptitle("Feature Scatter: Stamped (red) vs Clean (green)", fontsize=14)

    for ax, (fx, fy) in zip(axes, feat_names_pairs):
        for card, feat in zip(cards, features_list):
            color = "red" if card["stamped"] else "green"
            marker = "o" if card["stamped"] else "s"
            ax.scatter(feat[fx], feat[fy], c=color, marker=marker, s=60,
                       edgecolors="black", linewidth=0.5, alpha=0.8)
            ax.annotate(card["name"][:8], (feat[fx], feat[fy]),
                        fontsize=6, alpha=0.7)
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.set_title(f"{fx} vs {fy}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_scatter.png", dpi=150)
    plt.close()
    print(f"  Saved feature_scatter.png")


def plot_feature_comparison(cards, features_list):
    """Bar chart comparing all scalar features between stamped and clean."""
    scalar_keys = [k for k in features_list[0] if not k.startswith("_")]
    stamped_feats = [f for c, f in zip(cards, features_list) if c["stamped"]]
    clean_feats = [f for c, f in zip(cards, features_list) if not c["stamped"]]

    fig, axes = plt.subplots(6, 5, figsize=(22, 24))
    axes = axes.flatten()
    fig.suptitle("Per-Feature Distributions: Stamped (red) vs Clean (green)", fontsize=14)

    for idx, key in enumerate(scalar_keys):
        if idx >= len(axes):
            break
        ax = axes[idx]
        s_vals = [f[key] for f in stamped_feats]
        c_vals = [f[key] for f in clean_feats]

        # Strip plot
        ax.scatter([0] * len(s_vals), s_vals, c="red", alpha=0.7, s=40, label="Stamped")
        ax.scatter([1] * len(c_vals), c_vals, c="green", alpha=0.7, s=40, label="Clean")
        # Mean markers
        if s_vals:
            ax.hlines(np.mean(s_vals), -0.3, 0.3, colors="darkred", linewidth=2)
        if c_vals:
            ax.hlines(np.mean(c_vals), 0.7, 1.3, colors="darkgreen", linewidth=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Stamp", "Clean"])
        ax.set_title(key, fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    # Hide unused
    for idx in range(len(scalar_keys), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved feature_comparison.png")


def plot_stamp_vs_control_crops(cards):
    """Visual crop comparison: stamp region and control region side by side."""
    n = len(cards)
    fig, axes = plt.subplots(n, 2, figsize=(8, 2.5 * n))
    fig.suptitle("Stamp Region (left) vs Control Region (right)", fontsize=14, y=1.0)

    for i, card in enumerate(cards):
        stamp = crop_stamp_region(card["img"])
        ctrl = crop_control_region(card["img"])
        label = "STAMPED" if card["stamped"] else "CLEAN"
        color = "red" if card["stamped"] else "green"

        axes[i, 0].imshow(cv2.cvtColor(stamp, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"{card['name']} [{label}] - Stamp", fontsize=9, color=color)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cv2.cvtColor(ctrl, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"{card['name']} [{label}] - Control", fontsize=9, color=color)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "crop_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved crop_comparison.png")


# ── Main analysis ────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Stamp Color Histogram Analysis")
    print("=" * 70)

    # Load data
    cards = load_ground_truth()
    n_stamped = sum(1 for c in cards if c["stamped"])
    n_clean = sum(1 for c in cards if not c["stamped"])
    print(f"\n  Loaded {len(cards)} cards: {n_stamped} stamped, {n_clean} clean")
    for c in cards:
        print(f"    {'[S]' if c['stamped'] else '[C]'} {c['name']:20s} {c['variant']:15s} {c['image']}")

    # Extract features
    print("\n  Extracting color features...")
    features_list = []
    for card in cards:
        feat = compute_color_features(card["img"])
        features_list.append(feat)

    # ── Separability analysis ──
    scalar_keys = [k for k in features_list[0] if not k.startswith("_")]
    labels = [c["stamped"] for c in cards]

    print(f"\n{'=' * 70}")
    print(f"  Feature Separability (|mean_s - mean_c| / (std_s + std_c))")
    print(f"{'=' * 70}")
    print(f"  {'Feature':<30s} {'Stamped':>12s} {'Clean':>12s} {'Sep':>8s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8}")

    separations = {}
    for key in scalar_keys:
        s_vals = np.array([f[key] for c, f in zip(cards, features_list) if c["stamped"]])
        c_vals = np.array([f[key] for c, f in zip(cards, features_list) if not c["stamped"]])
        s_mean, s_std = np.mean(s_vals), np.std(s_vals)
        c_mean, c_std = np.mean(c_vals), np.std(c_vals)
        sep = abs(s_mean - c_mean) / (s_std + c_std + 1e-10)
        separations[key] = sep
        print(f"  {key:<30s} {s_mean:>8.4f}+{s_std:<4.3f} {c_mean:>8.4f}+{c_std:<4.3f} {sep:>7.3f}")

    ranked = sorted(separations.items(), key=lambda x: -x[1])
    print(f"\n  Top 10 features by separability:")
    for i, (k, v) in enumerate(ranked[:10]):
        print(f"    {i+1:2d}. {k:<30s} {v:.3f}")

    # ── Threshold classifier ──
    print(f"\n{'=' * 70}")
    print(f"  Threshold Classifier (single feature)")
    print(f"{'=' * 70}")

    results = []
    for key in scalar_keys:
        vals = [f[key] for f in features_list]
        t, d, acc = find_best_threshold(vals, labels)
        results.append((acc, key, t, d))

    results.sort(key=lambda x: -x[0])
    for acc, key, t, d in results:
        print(f"  {acc:5.1%}  {key:<30s} {d} {t:.6f}")

    # Show best single-feature misclassifications
    best_acc, best_key, best_t, best_d = results[0]
    print(f"\n  BEST single feature: {best_key} {best_d} {best_t:.6f} -> {best_acc:.1%}")
    print(f"\n  Misclassified cards:")
    for card, feat in zip(cards, features_list):
        val = feat[best_key]
        if best_d == ">":
            pred = val > best_t
        else:
            pred = val < best_t
        status = "OK" if pred == card["stamped"] else "WRONG"
        if status == "WRONG":
            print(f"    {card['name']:20s} {best_key}={val:.6f} "
                  f"pred={'stamped' if pred else 'clean'} "
                  f"actual={'stamped' if card['stamped'] else 'clean'}")

    # ── Two-feature combos ──
    print(f"\n{'=' * 70}")
    print(f"  Two-Feature Combo Classifier (OR logic)")
    print(f"{'=' * 70}")

    top_keys = [k for _, k, _, _ in results[:10]]
    combo_results = []
    for i, k1 in enumerate(top_keys):
        v1 = [f[k1] for f in features_list]
        t1, d1, _ = find_best_threshold(v1, labels)
        for k2 in top_keys[i+1:]:
            v2 = [f[k2] for f in features_list]
            t2, d2, _ = find_best_threshold(v2, labels)
            # AND logic
            correct_and = 0
            for v1i, v2i, lab in zip(v1, v2, labels):
                p1 = (v1i > t1) if d1 == ">" else (v1i < t1)
                p2 = (v2i > t2) if d2 == ">" else (v2i < t2)
                pred = p1 and p2
                if pred == lab:
                    correct_and += 1
            acc_and = correct_and / len(labels)
            # OR logic
            correct_or = 0
            for v1i, v2i, lab in zip(v1, v2, labels):
                p1 = (v1i > t1) if d1 == ">" else (v1i < t1)
                p2 = (v2i > t2) if d2 == ">" else (v2i < t2)
                pred = p1 or p2
                if pred == lab:
                    correct_or += 1
            acc_or = correct_or / len(labels)
            combo_results.append((max(acc_and, acc_or),
                                  "AND" if acc_and >= acc_or else "OR",
                                  k1, t1, d1, k2, t2, d2))

    combo_results.sort(key=lambda x: -x[0])
    for acc, logic, k1, t1, d1, k2, t2, d2 in combo_results[:10]:
        print(f"  {acc:5.1%}  {k1} {d1} {t1:.4f} {logic} {k2} {d2} {t2:.4f}")

    # ── Per-card detail table ──
    print(f"\n{'=' * 70}")
    print(f"  Per-Card Feature Values (top features)")
    print(f"{'=' * 70}")
    top3 = [k for _, k, _, _ in results[:3]]
    header = f"  {'Card':20s} {'Label':8s}"
    for k in top3:
        header += f" {k[:15]:>15s}"
    print(header)
    print(f"  {'-' * (20 + 8 + 15 * len(top3) + len(top3))}")
    for card, feat in zip(cards, features_list):
        row = f"  {card['name']:20s} {'STAMP' if card['stamped'] else 'CLEAN':8s}"
        for k in top3:
            row += f" {feat[k]:>15.6f}"
        print(row)

    # ── Plots ──
    print(f"\n  Generating plots...")
    plot_stamp_vs_control_crops(cards)
    plot_hue_histograms(cards, features_list)
    plot_saturation_histograms(cards, features_list)
    plot_value_histograms(cards, features_list)
    plot_feature_comparison(cards, features_list)

    # Scatter: best 2 features
    if len(results) >= 2:
        plot_feature_scatter(cards, features_list, [
            (results[0][1], results[1][1]),
            ("gold_ratio_stamp", "metallic_ratio_stamp"),
            ("gold_broad_diff", "metallic_diff"),
        ])

    # Save features JSON
    out_json = OUT_DIR / "color_features.json"
    json_data = []
    for card, feat in zip(cards, features_list):
        entry = {k: v for k, v in feat.items() if not k.startswith("_")}
        entry["card_name"] = card["name"]
        entry["stamped"] = card["stamped"]
        entry["variant"] = card["variant"]
        json_data.append(entry)
    with open(out_json, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved color_features.json")

    print(f"\n  All output in: {OUT_DIR}")


if __name__ == "__main__":
    main()
