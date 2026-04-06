"""Pokemon type detection via DINOv2 embeddings of the type symbol ROI.

Instead of analyzing the card background color (which is unreliable on binder
scans due to warm color cast, sleeve reflections, etc.), this module crops the
small type symbol icon from the top-right corner of the card and classifies it
using DINOv2 nearest-neighbor matching against a reference bank.

Pipeline:
  1. Crop the type symbol ROI from the top-right area of the card segment.
  2. Extract a 768-dim DINOv2 CLS embedding of the cropped symbol.
  3. Compare against a pre-built reference bank (per-type prototype embeddings).
  4. Return the type with highest cosine similarity.

Reference bank construction:
  - For each of the 11 Pokemon types, select 10 reference card images.
  - Crop the type symbol ROI from each reference image.
  - Extract DINOv2 embeddings and store as per-type prototypes.
  - At query time, compare query embedding to all prototypes; the type
    whose prototype has the highest average similarity wins.

Advantages over color-based detection:
  - The type symbol itself is invariant to card era, lighting, white balance.
  - DINOv2 captures shape features (flame, water drop, leaf, etc.) not just color.
  - Works even on cards with metallic/silver EX frames where background is neutral.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Type symbol ROI in normalized coordinates (fraction of card width/height).
# The type symbol is the small energy icon next to the HP in the top-right.
# On a 1008x1530 card segment:
#   x: ~880-980 -> 0.87 - 0.97
#   y: ~15-85   -> 0.01 - 0.056
# We use a slightly generous crop to account for slight alignment variation.
_ROI_X1 = 0.85
_ROI_X2 = 0.99
_ROI_Y1 = 0.005
_ROI_Y2 = 0.065

# For reference card images (smaller, ~245x342), the symbol is in a similar
# relative position but we may need to adjust slightly.
_REF_ROI_X1 = 0.83
_REF_ROI_X2 = 0.98
_REF_ROI_Y1 = 0.005
_REF_ROI_Y2 = 0.075

# All Pokemon energy types
ALL_TYPES = [
    "Grass", "Fire", "Water", "Lightning", "Psychic",
    "Fighting", "Darkness", "Metal", "Dragon", "Fairy", "Colorless",
]

# Path to the pre-built reference bank
_BANK_PATH = Path("data/type_symbol_bank.pkl")

# Card names JSON for type lookup
_CARD_NAMES_JSON = Path(__file__).resolve().parent.parent.parent / "data" / "card_names.json"

# Reference image directory
_REF_IMAGE_DIR = Path("data/card_images")

# ---------------------------------------------------------------------------
# DINOv2 model (reuse from dino_matcher if already loaded)
# ---------------------------------------------------------------------------

_model: Optional[torch.nn.Module] = None
_device: Optional[torch.device] = None

# ImageNet normalization
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _load_model() -> tuple:
    """Load DINOv2 ViT-B/14 (reuses dino_matcher's cached model if available)."""
    global _model, _device
    if _model is not None:
        return _model, _device

    # Try to reuse the already-loaded model from dino_matcher
    try:
        from cardprice.ml.dino_matcher import _model as dm_model, _device as dm_device
        if dm_model is not None:
            _model = dm_model
            _device = dm_device
            logger.info("Reusing DINOv2 model from dino_matcher.")
            return _model, _device
    except ImportError:
        pass

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading DINOv2 ViT-B/14 on %s ...", _device)
    _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    _model.to(_device)
    _model.eval()
    logger.info("DINOv2 model loaded.")
    return _model, _device


# ---------------------------------------------------------------------------
# Symbol ROI extraction
# ---------------------------------------------------------------------------

def _crop_symbol_roi(
    img: np.ndarray,
    *,
    is_reference: bool = False,
) -> np.ndarray:
    """Crop the type symbol from the top-right corner of a card image.

    Parameters
    ----------
    img : np.ndarray
        BGR image of the card.
    is_reference : bool
        If True, use the reference image ROI coordinates (slightly different
        aspect ratio than binder scan segments).

    Returns
    -------
    np.ndarray
        Cropped BGR image of the symbol region.
    """
    h, w = img.shape[:2]
    if is_reference:
        x1, x2 = int(w * _REF_ROI_X1), int(w * _REF_ROI_X2)
        y1, y2 = int(h * _REF_ROI_Y1), int(h * _REF_ROI_Y2)
    else:
        x1, x2 = int(w * _ROI_X1), int(w * _ROI_X2)
        y1, y2 = int(h * _ROI_Y1), int(h * _ROI_Y2)

    crop = img[y1:y2, x1:x2]

    # Ensure the crop is at least 10x10 pixels
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        logger.warning(
            "Symbol ROI too small (%dx%d) from %dx%d image",
            crop.shape[1], crop.shape[0], w, h,
        )
        # Fallback: use a larger region
        x1, x2 = int(w * 0.80), int(w * 0.99)
        y1, y2 = int(h * 0.00), int(h * 0.10)
        crop = img[y1:y2, x1:x2]

    return crop


def crop_symbol_roi(image_path, *, is_reference: bool = False) -> np.ndarray:
    """Crop the type symbol ROI from an image file."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return _crop_symbol_roi(img, is_reference=is_reference)


# ---------------------------------------------------------------------------
# Embedding extraction for symbol ROI
# ---------------------------------------------------------------------------

def _extract_roi_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """Extract a 768-dim L2-normalized DINOv2 embedding from a BGR crop.

    Returns np.ndarray of shape (768,), float32, L2-normalized.
    """
    model, device = _load_model()

    # Convert BGR -> RGB -> PIL
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    tensor = _transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(tensor)

    vec = embedding.cpu().numpy().astype(np.float32).squeeze()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _extract_roi_embedding_batch(crops_bgr: list) -> list:
    """Extract embeddings for a batch of BGR crops."""
    if not crops_bgr:
        return []

    model, device = _load_model()

    tensors = []
    for crop in crops_bgr:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensors.append(_transform(pil_img))

    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        embeddings = model(batch)

    vecs = embeddings.cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs /= norms

    return [vecs[i] for i in range(len(vecs))]


# ---------------------------------------------------------------------------
# Reference bank: build from reference card images
# ---------------------------------------------------------------------------

def _load_card_types() -> dict:
    """Load card_id -> type mapping from JSON fallback.

    Returns dict mapping card_id (e.g. 'base1-4/normal') -> type string.
    """
    if not _CARD_NAMES_JSON.exists():
        raise FileNotFoundError(f"Card names JSON not found: {_CARD_NAMES_JSON}")

    with open(_CARD_NAMES_JSON) as f:
        data = json.load(f)

    card_types = {}
    for entry in data:
        if len(entry) > 4 and entry[4]:
            card_id = entry[0]  # e.g. 'base1-4/normal'
            types = entry[4]     # e.g. ['Fire']
            if types:
                card_types[card_id] = types[0]  # primary type

    return card_types


def _find_ref_image(card_id: str) -> Optional[Path]:
    """Find the reference image file for a card_id.

    card_id format: 'base1-4/normal' -> data/card_images/base1/base1-4_normal.png
    """
    parts = card_id.split("/")
    if len(parts) != 2:
        return None
    base_id, variant = parts
    set_id = base_id.rsplit("-", 1)[0]
    filename = f"{base_id}_{variant}.png"
    path = _REF_IMAGE_DIR / set_id / filename
    if path.exists():
        return path
    return None


def build_reference_bank(
    cards_per_type: int = 15,
    bank_path: Optional[str] = None,
    *,
    preferred_sets: Optional[list] = None,
) -> dict:
    """Build a reference bank of type symbol DINOv2 embeddings.

    For each type, selects `cards_per_type` reference cards, crops the type
    symbol ROI, extracts DINOv2 embeddings, and stores them.

    Parameters
    ----------
    cards_per_type : int
        Number of reference cards per type.
    bank_path : str, optional
        Where to save the bank pickle. Defaults to _BANK_PATH.
    preferred_sets : list, optional
        Prefer cards from these sets (modern sets with clear symbols).

    Returns
    -------
    dict
        Bank data: {type_name: np.ndarray of shape (N, 768)}.
    """
    if bank_path is None:
        bank_path = str(_BANK_PATH)

    card_types = _load_card_types()

    # Group card_ids by type
    by_type: dict[str, list[str]] = {t: [] for t in ALL_TYPES}
    for card_id, card_type in card_types.items():
        if card_type in by_type:
            by_type[card_type].append(card_id)

    # Preferred sets: modern sets with high-quality, standardized symbol placement
    if preferred_sets is None:
        preferred_sets = [
            "sv8", "sv7", "sv6", "sv5", "sv4", "sv3", "sv2", "sv1",
            "swsh12", "swsh11", "swsh10", "swsh9", "swsh8", "swsh7",
            "sm12", "sm11", "sm10", "sm9", "sm8",
            "dp1", "dp2", "dp3", "dp4", "dp5", "dp6", "dp7",
            "bw1", "bw2", "bw3", "bw4", "bw5",
        ]
    preferred_set = set(preferred_sets)

    bank = {}
    total_built = 0

    for type_name in ALL_TYPES:
        candidates = by_type[type_name]
        if not candidates:
            logger.warning("No cards found for type %s", type_name)
            continue

        # Sort: preferred sets first, then alphabetically for reproducibility
        def sort_key(cid):
            set_id = cid.split("-")[0].split("/")[0]
            return (0 if set_id in preferred_set else 1, cid)

        candidates.sort(key=sort_key)

        # Collect crops that have valid reference images
        crops = []
        selected = []
        for cid in candidates:
            if len(selected) >= cards_per_type:
                break
            ref_path = _find_ref_image(cid)
            if ref_path is None:
                continue
            try:
                crop = crop_symbol_roi(ref_path, is_reference=True)
                if crop.size > 0:
                    crops.append(crop)
                    selected.append(cid)
            except Exception as e:
                logger.debug("Failed to crop %s: %s", cid, e)
                continue

        if not crops:
            logger.warning("No valid reference crops for type %s", type_name)
            continue

        # Batch extract embeddings
        embeddings = _extract_roi_embedding_batch(crops)
        bank[type_name] = np.stack(embeddings)  # (N, 768)
        total_built += len(embeddings)

        logger.info(
            "Type %s: %d reference symbols from %s",
            type_name, len(embeddings),
            ", ".join(cid.split("/")[0] for cid in selected[:5]) + ("..." if len(selected) > 5 else ""),
        )

    # Save
    os.makedirs(os.path.dirname(bank_path) or ".", exist_ok=True)
    with open(bank_path, "wb") as f:
        pickle.dump(bank, f)

    logger.info("Saved type symbol bank (%d types, %d total embeddings) to %s",
                len(bank), total_built, bank_path)

    return bank


# ---------------------------------------------------------------------------
# Query-time classification
# ---------------------------------------------------------------------------

_bank: Optional[dict] = None


def _load_bank(bank_path: Optional[str] = None) -> dict:
    """Load the pre-built reference bank."""
    global _bank
    if _bank is not None:
        return _bank

    if bank_path is None:
        bank_path = str(_BANK_PATH)

    if not os.path.exists(bank_path):
        raise FileNotFoundError(
            f"Type symbol bank not found: {bank_path}. "
            "Run build_reference_bank() first."
        )

    with open(bank_path, "rb") as f:
        _bank = pickle.load(f)

    logger.info(
        "Loaded type symbol bank: %s",
        ", ".join(f"{t}={v.shape[0]}" for t, v in _bank.items()),
    )
    return _bank


def classify_type_symbol(
    image_path,
    *,
    top_n: int = 3,
    bank_path: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """Classify the Pokemon type from the type symbol in a card image.

    Parameters
    ----------
    image_path : str or Path
        Path to the card segment image (1008x1530 or similar).
    top_n : int
        Number of top predictions to return.
    bank_path : str, optional
        Path to the reference bank pickle.

    Returns
    -------
    list of (type_name, confidence)
        Sorted by confidence descending. Confidence is cosine similarity
        averaged over reference prototypes for that type.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not decode image: {image_path}")

    return classify_type_symbol_from_array(img, top_n=top_n, bank_path=bank_path,
                                           label=image_path.name)


def classify_type_symbol_from_array(
    img: np.ndarray,
    *,
    top_n: int = 3,
    bank_path: Optional[str] = None,
    label: str = "<array>",
) -> List[Tuple[str, float]]:
    """Classify type from an already-loaded BGR image array.

    Parameters
    ----------
    img : np.ndarray
        BGR image of the card.
    top_n : int
        Number of predictions.
    bank_path : str, optional
        Path to reference bank.
    label : str
        Label for logging.

    Returns
    -------
    list of (type_name, confidence)
    """
    bank = _load_bank(bank_path)

    # Crop symbol ROI
    crop = _crop_symbol_roi(img, is_reference=False)

    # Extract embedding
    query_emb = _extract_roi_embedding(crop)  # (768,)

    # Compare against each type's reference embeddings
    scores = {}
    for type_name, ref_embs in bank.items():
        # Cosine similarity (dot product on L2-normed vectors)
        sims = ref_embs @ query_emb  # (N,)
        # Use max similarity (nearest prototype) for better discrimination
        # than average, since some reference cards may have variant symbol styles
        scores[type_name] = float(np.max(sims))

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Normalize scores to [0, 1] range for confidence
    # (cosine sim is already in [-1, 1] but typically 0.3-0.9 for these)
    results = [(name, score) for name, score in ranked[:top_n]]

    logger.debug(
        "Type symbol classification for %s: %s",
        label,
        ", ".join(f"{n} ({c:.3f})" for n, c in results),
    )

    return results


# ---------------------------------------------------------------------------
# Ground truth for eval
# ---------------------------------------------------------------------------

# Reuse the ground truth from color_detector, mapping to hires paths
_EVAL_GROUND_TRUTH_HIRES = {
    # Page 1 (EX delta species + EX era) - hires segments
    "page_20260228_174819_cards_hires/card_00.png": "Lightning",  # Dragonair delta
    "page_20260228_174819_cards_hires/card_01.png": "Colorless",  # Skitty
    "page_20260228_174819_cards_hires/card_02.png": "Psychic",    # Trapinch delta
    "page_20260228_174819_cards_hires/card_03.png": "Psychic",    # Vibrava delta
    "page_20260228_174819_cards_hires/card_04.png": "Metal",      # Delcatty ex
    "page_20260228_174819_cards_hires/card_05.png": "Colorless",  # Wigglytuff ex
    "page_20260228_174819_cards_hires/card_06.png": "Metal",      # Swampert ex
    "page_20260228_174819_cards_hires/card_07.png": "Metal",      # Jirachi ex
    "page_20260228_174819_cards_hires/card_08.png": "Metal",      # Flygon ex delta
    # Page 2 (e-series)
    "page_20260228_195512_cards_hires/card_00.png": "Psychic",    # Natu
    "page_20260228_195512_cards_hires/card_01.png": "Psychic",    # Xatu
    "page_20260228_195512_cards_hires/card_02.png": "Psychic",    # Mr. Mime
    "page_20260228_195512_cards_hires/card_03.png": "Psychic",    # Natu
    "page_20260228_195512_cards_hires/card_04.png": "Psychic",    # Xatu
    "page_20260228_195512_cards_hires/card_05.png": "Colorless",  # Rattata
    "page_20260228_195512_cards_hires/card_06.png": "Colorless",  # Rattata
    "page_20260228_195512_cards_hires/card_07.png": "Colorless",  # Raticate
    "page_20260228_195512_cards_hires/card_08.png": "Colorless",  # Ditto
    # Page 3 (mixed eras)
    # card_00 is empty slot - skip
    "page_20260228_202134_cards_hires/card_01.png": "Colorless",  # Latios delta
    "page_20260228_202134_cards_hires/card_02.png": "Colorless",  # Latias ex
    "page_20260228_202134_cards_hires/card_03.png": "Grass",      # Venusaur
    "page_20260228_202134_cards_hires/card_04.png": "Colorless",  # Flygon
    "page_20260228_202134_cards_hires/card_05.png": "Lightning",  # Raikou
    "page_20260228_202134_cards_hires/card_06.png": "Water",      # Kingdra
    "page_20260228_202134_cards_hires/card_07.png": "Water",      # Suicune
    "page_20260228_202134_cards_hires/card_08.png": "Colorless",  # Staraptor
}


def run_eval(data_root: str = "data/inbox", bank_path: Optional[str] = None) -> dict:
    """Run type symbol classification against eval ground truth.

    Returns dict with accuracy, correct count, total count, and per-card results.
    """
    root = Path(data_root)
    results = []
    correct = 0
    total = 0

    for rel_path, expected_type in _EVAL_GROUND_TRUTH_HIRES.items():
        full_path = root / rel_path
        if not full_path.exists():
            results.append({
                "path": rel_path,
                "expected": expected_type,
                "predicted": "MISSING",
                "correct": False,
            })
            continue

        total += 1
        try:
            preds = classify_type_symbol(full_path, top_n=3, bank_path=bank_path)
            predicted = preds[0][0]
            conf = preds[0][1]
            is_correct = predicted == expected_type

            # Also check if expected is in top-3
            top3_types = [p[0] for p in preds]
            in_top3 = expected_type in top3_types

            if is_correct:
                correct += 1

            results.append({
                "path": rel_path,
                "expected": expected_type,
                "predicted": predicted,
                "confidence": conf,
                "correct": is_correct,
                "in_top3": in_top3,
                "top3": preds,
            })
        except Exception as e:
            results.append({
                "path": rel_path,
                "expected": expected_type,
                "predicted": f"ERROR: {e}",
                "correct": False,
            })

    accuracy = correct / total if total > 0 else 0.0
    top3_correct = sum(1 for r in results if r.get("in_top3", False))
    top3_accuracy = top3_correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "correct": correct,
        "top3_correct": top3_correct,
        "total": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = sys.argv[1:]

    if not args or args[0] == "--eval":
        # Run eval
        print("Running type symbol DINOv2 eval...")

        # Build bank if needed
        if not _BANK_PATH.exists():
            print("Building reference bank first...")
            build_reference_bank()

        print("=" * 90)
        eval_result = run_eval()

        for r in eval_result["results"]:
            marker = "OK" if r["correct"] else ("TOP3" if r.get("in_top3") else "MISS")
            conf = f"{r.get('confidence', 0):.3f}" if "confidence" in r else "?"
            top3 = ""
            if "top3" in r:
                top3 = " | " + ", ".join(f"{t} {s:.3f}" for t, s in r["top3"])
            print(
                f"[{marker:4s}] {r['path']:55s} "
                f"exp={r['expected']:12s} "
                f"got={r['predicted']:12s} "
                f"sim={conf:>6s}{top3}"
            )

        print("=" * 90)
        print(
            f"Top-1 accuracy: {eval_result['correct']}/{eval_result['total']} "
            f"= {eval_result['accuracy']:.1%}"
        )
        print(
            f"Top-3 accuracy: {eval_result['top3_correct']}/{eval_result['total']} "
            f"= {eval_result['top3_accuracy']:.1%}"
        )

    elif args[0] == "--build":
        n = int(args[1]) if len(args) > 1 else 15
        print(f"Building reference bank with {n} cards per type...")
        bank = build_reference_bank(cards_per_type=n)
        for t, embs in bank.items():
            print(f"  {t:15s}: {embs.shape[0]} embeddings")

    else:
        # Classify individual images
        if not _BANK_PATH.exists():
            print("Building reference bank first...")
            build_reference_bank()

        for p in args:
            try:
                preds = classify_type_symbol(p, top_n=5)
                top = preds[0]
                alts = ", ".join(f"{n} {c:.3f}" for n, c in preds[1:])
                print(
                    f"{Path(p).name:30s} -> {top[0]:12s} (sim={top[1]:.3f})"
                    + (f"   alts: {alts}" if alts else "")
                )
            except Exception as e:
                print(f"{Path(p).name:30s} -> ERROR: {e}")
