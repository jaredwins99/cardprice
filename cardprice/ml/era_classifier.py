"""Era classifier: predict Pokemon TCG era from DINOv2 embeddings.

Each Pokemon TCG era has distinctive visual features (borders, text layout,
HP position, card frame design). DINOv2 embeddings capture these visual
patterns, making a simple LogisticRegression classifier effective.

Eras:
    1  = Base Set / Jungle / Fossil / Base Set 2 / Rocket / Gym (base*, gym*)
    2  = Neo (neo*)
    3  = e-Card / Expedition / Aquapolis / Skyridge (ecard*)
    4  = EX (ex*, pop1-pop5)
    5  = Diamond & Pearl / Platinum (dp*, pl*, pop6-pop9)
    6  = HeartGold SoulSilver (hgss*, hsp*)
    7  = Black & White (bw*, bwp*)
    8  = XY (xy*, xyp*, g1*, dc1*, col1*, dv1*)
    9  = Sun & Moon (sm*, sma*, smp*, det1*)
    10 = Sword & Shield (swsh*, swshp*, cel25*, pgo*)
    11 = Scarlet & Violet (sv*, sve*, svp*)

Usage:
    # Train and save
    from cardprice.ml.era_classifier import train_era_classifier
    results = train_era_classifier()

    # Predict from image
    from cardprice.ml.era_classifier import predict_era, get_era_name
    era, conf = predict_era("data/inbox/page_xxx/card_00.png")
    print(f"Era {era} ({get_era_name(era)}), confidence {conf:.2f}")

    # Predict from pre-computed embedding
    from cardprice.ml.era_classifier import predict_era_from_embedding
    era, conf = predict_era_from_embedding(embedding_768d)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REF_EMBEDDINGS_PATH = _PROJECT_ROOT / "data" / "ref_embeddings.pkl"
_CARD_NAMES_JSON = _PROJECT_ROOT / "data" / "card_names.json"
_MODEL_PATH = _PROJECT_ROOT / "data" / "models" / "era_classifier.pkl"

ERA_NAMES = {
    1: "Base Set (1999-2000)",
    2: "Neo (2000-2002)",
    3: "e-Card (2002-2003)",
    4: "EX (2003-2007)",
    5: "Diamond & Pearl (2007-2009)",
    6: "HeartGold SoulSilver (2010-2011)",
    7: "Black & White (2011-2013)",
    8: "XY (2013-2017)",
    9: "Sun & Moon (2017-2019)",
    10: "Sword & Shield (2020-2023)",
    11: "Scarlet & Violet (2023-present)",
}

# Map set prefix -> era
# Covers all 171 set IDs in the dataset
_SET_ERA_MAP: dict[str, int] = {}

def _init_set_era_map():
    """Build prefix -> era mapping."""
    if _SET_ERA_MAP:
        return

    # Era 1: Base / Jungle / Fossil / Rocket / Gym / promos
    for prefix in ["base1", "base2", "base3", "base4", "base5", "base6",
                    "basep", "gym1", "gym2", "bp"]:
        _SET_ERA_MAP[prefix] = 1

    # Era 2: Neo
    for prefix in ["neo1", "neo2", "neo3", "neo4", "si1"]:
        _SET_ERA_MAP[prefix] = 2

    # Era 3: e-Card
    for prefix in ["ecard1", "ecard2", "ecard3"]:
        _SET_ERA_MAP[prefix] = 3

    # Era 4: EX
    for i in range(1, 17):
        _SET_ERA_MAP[f"ex{i}"] = 4
    for prefix in ["pop1", "pop2", "pop3", "pop4", "pop5",
                    "np", "tk1a", "tk1b", "tk2a", "tk2b"]:
        _SET_ERA_MAP[prefix] = 4

    # Era 5: DP / Platinum
    for i in range(1, 8):
        _SET_ERA_MAP[f"dp{i}"] = 5
    _SET_ERA_MAP["dpp"] = 5
    for i in range(1, 5):
        _SET_ERA_MAP[f"pl{i}"] = 5
    for prefix in ["pop6", "pop7", "pop8", "pop9"]:
        _SET_ERA_MAP[prefix] = 5

    # Era 6: HGSS
    for i in range(1, 5):
        _SET_ERA_MAP[f"hgss{i}"] = 6
    _SET_ERA_MAP["hsp"] = 6
    _SET_ERA_MAP["col1"] = 6

    # Era 7: BW
    for i in range(1, 12):
        _SET_ERA_MAP[f"bw{i}"] = 7
    _SET_ERA_MAP["bwp"] = 7

    # Era 8: XY
    _SET_ERA_MAP["xy0"] = 8
    for i in range(1, 13):
        _SET_ERA_MAP[f"xy{i}"] = 8
    for prefix in ["xyp", "g1", "dc1", "dv1"]:
        _SET_ERA_MAP[prefix] = 8

    # Era 9: SM
    for i in range(1, 13):
        _SET_ERA_MAP[f"sm{i}"] = 9
    for prefix in ["sm35", "sm75", "sm115", "sma", "smp", "det1"]:
        _SET_ERA_MAP[prefix] = 9

    # Era 10: SWSH
    for i in range(1, 13):
        _SET_ERA_MAP[f"swsh{i}"] = 10
    for prefix in ["swsh35", "swsh45", "swsh45sv",
                    "swsh9tg", "swsh10tg", "swsh11tg", "swsh12tg",
                    "swsh12pt5", "swsh12pt5gg",
                    "swshp", "cel25", "cel25c", "pgo", "fut20"]:
        _SET_ERA_MAP[prefix] = 10

    # Era 11: SV
    for i in range(1, 11):
        _SET_ERA_MAP[f"sv{i}"] = 11
    for prefix in ["sv3pt5", "sv4pt5", "sv6pt5", "sv8pt5",
                    "sve", "svp", "rsv10pt5", "zsv10pt5"]:
        _SET_ERA_MAP[prefix] = 11

    # McDonald's promos — map to approximate eras
    _SET_ERA_MAP["mcd11"] = 7   # BW era
    _SET_ERA_MAP["mcd12"] = 7   # BW era
    _SET_ERA_MAP["mcd14"] = 8   # XY era
    _SET_ERA_MAP["mcd15"] = 8   # XY era
    _SET_ERA_MAP["mcd16"] = 8   # XY era
    _SET_ERA_MAP["mcd17"] = 9   # SM era
    _SET_ERA_MAP["mcd18"] = 9   # SM era
    _SET_ERA_MAP["mcd19"] = 9   # SM era
    _SET_ERA_MAP["mcd21"] = 10  # SWSH era
    _SET_ERA_MAP["mcd22"] = 10  # SWSH era

    # Misc
    _SET_ERA_MAP["me1"] = 4     # mythical era (EX layout)
    _SET_ERA_MAP["me2"] = 4
    _SET_ERA_MAP["me2pt5"] = 4
    _SET_ERA_MAP["ru1"] = 11    # recent


def set_id_to_era(set_id: str) -> Optional[int]:
    """Map a set_id (e.g. 'ex4', 'swsh10tg') to an era number (1-11)."""
    _init_set_era_map()
    return _SET_ERA_MAP.get(set_id)


def get_era_name(era: int) -> str:
    """Return human-readable era name."""
    return ERA_NAMES.get(era, f"Unknown era {era}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_training_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load embeddings and era labels from ref_embeddings.pkl + card_names.json.

    Returns
    -------
    X : np.ndarray, shape (N, 768)
        DINOv2 embeddings.
    y : np.ndarray, shape (N,)
        Era labels (1-11).
    card_ids : list[str]
        Corresponding card IDs.
    """
    _init_set_era_map()

    # Load embeddings
    with open(_REF_EMBEDDINGS_PATH, "rb") as f:
        embeddings: dict[str, np.ndarray] = pickle.load(f)

    # Build card_id -> set_id mapping from card_names.json
    with open(_CARD_NAMES_JSON) as f:
        card_names_list = json.load(f)

    card_to_set: dict[str, str] = {}
    for entry in card_names_list:
        card_id = entry[0]  # e.g. "xy4-91/normal"
        set_id = entry[2]   # e.g. "xy4"
        card_to_set[card_id] = set_id

    # Build X, y arrays
    X_list = []
    y_list = []
    card_ids = []
    skipped = 0

    for card_id, emb in embeddings.items():
        set_id = card_to_set.get(card_id)
        if set_id is None:
            # Try extracting set_id from card_id directly
            # card_id format: "base1-100/normal" -> set = "base1"
            parts = card_id.split("-")
            if len(parts) >= 2:
                set_id = parts[0]
            else:
                skipped += 1
                continue

        era = set_id_to_era(set_id)
        if era is None:
            skipped += 1
            continue

        X_list.append(emb)
        y_list.append(era)
        card_ids.append(card_id)

    if skipped > 0:
        logger.info("Skipped %d cards with unmapped set IDs", skipped)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    logger.info("Loaded %d cards across %d eras", len(X), len(set(y)))
    return X, y, card_ids


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_era_classifier(save: bool = True) -> dict:
    """Train a LogisticRegression classifier on DINOv2 embeddings -> era.

    Parameters
    ----------
    save : bool
        If True, save the trained model to data/models/era_classifier.pkl.

    Returns
    -------
    dict
        Training results including accuracy, cross-validation scores, and
        per-class metrics.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler

    X, y, card_ids = _load_training_data()

    # Standardize features (LogReg benefits from this)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train LogisticRegression with multinomial
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        random_state=42,
    )

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="accuracy")
    logger.info("5-fold CV accuracy: %.3f +/- %.3f", cv_scores.mean(), cv_scores.std())

    # Fit on full data
    clf.fit(X_scaled, y)

    # Full-data accuracy (not test, but sanity check)
    train_acc = clf.score(X_scaled, y)
    y_pred = clf.predict(X_scaled)
    report = classification_report(y, y_pred, output_dict=True,
                                   target_names=[get_era_name(e) for e in sorted(set(y))])

    results = {
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist(),
        "train_accuracy": float(train_acc),
        "n_samples": len(X),
        "n_eras": len(set(y)),
        "era_counts": {int(e): int(c) for e, c in
                       zip(*np.unique(y, return_counts=True))},
        "report": report,
    }

    if save and cv_scores.mean() > 0.70:
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            "classifier": clf,
            "scaler": scaler,
            "era_names": ERA_NAMES,
        }
        with open(_MODEL_PATH, "wb") as f:
            pickle.dump(model_data, f)
        logger.info("Saved era classifier to %s", _MODEL_PATH)
        results["saved"] = True
    else:
        if cv_scores.mean() <= 0.70:
            logger.warning("CV accuracy %.3f <= 0.70, NOT saving model", cv_scores.mean())
        results["saved"] = False

    return results


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_cached_model: Optional[dict] = None


def _load_model() -> dict:
    """Load the trained era classifier from disk."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Era classifier model not found at {_MODEL_PATH}. "
            "Run train_era_classifier() first."
        )

    with open(_MODEL_PATH, "rb") as f:
        _cached_model = pickle.load(f)
    logger.info("Loaded era classifier from %s", _MODEL_PATH)
    return _cached_model


def predict_era_from_embedding(embedding: np.ndarray) -> tuple[int, float]:
    """Predict era from a pre-computed 768-dim DINOv2 embedding.

    Parameters
    ----------
    embedding : np.ndarray
        768-dim L2-normalized DINOv2 CLS embedding.

    Returns
    -------
    era : int
        Predicted era (1-11).
    confidence : float
        Prediction confidence (max softmax probability).
    """
    model_data = _load_model()
    clf = model_data["classifier"]
    scaler = model_data["scaler"]

    # Reshape and scale
    X = embedding.reshape(1, -1).astype(np.float32)
    X_scaled = scaler.transform(X)

    # Predict with probabilities
    era = int(clf.predict(X_scaled)[0])
    proba = clf.predict_proba(X_scaled)[0]
    confidence = float(proba.max())

    return era, confidence


def predict_era(image_path: str) -> tuple[int, float]:
    """Predict era from a card image path.

    Loads DINOv2, extracts embedding, then classifies.

    Parameters
    ----------
    image_path : str
        Path to the card image.

    Returns
    -------
    era : int
        Predicted era (1-11).
    confidence : float
        Prediction confidence (max softmax probability).
    """
    from cardprice.ml.dino_matcher import extract_embedding

    embedding = extract_embedding(image_path)
    return predict_era_from_embedding(embedding)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Train the era classifier and print results."""
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 70)
    print("ERA CLASSIFIER: Training on DINOv2 reference embeddings")
    print("=" * 70)

    results = train_era_classifier(save=True)

    print(f"\n5-fold CV accuracy: {results['cv_mean']:.3f} +/- {results['cv_std']:.3f}")
    print(f"Train accuracy:     {results['train_accuracy']:.3f}")
    print(f"Samples:            {results['n_samples']}")
    print(f"Eras:               {results['n_eras']}")
    print(f"Model saved:        {results['saved']}")

    print("\nSamples per era:")
    for era, count in sorted(results["era_counts"].items()):
        print(f"  Era {era:2d} ({get_era_name(era):40s}): {count:5d}")

    print("\nPer-class metrics:")
    for era_name, metrics in results["report"].items():
        if isinstance(metrics, dict) and "precision" in metrics:
            print(f"  {era_name:45s}  P={metrics['precision']:.3f}  "
                  f"R={metrics['recall']:.3f}  F1={metrics['f1-score']:.3f}  "
                  f"n={metrics['support']}")

    # Test on eval images if available
    import glob
    eval_images = sorted(glob.glob("data/inbox/page_*_cards/card_*.png"))
    if eval_images and results["saved"]:
        print(f"\n{'=' * 70}")
        print(f"Testing on {len(eval_images)} eval card images")
        print(f"{'=' * 70}")
        for img in eval_images:
            try:
                era, conf = predict_era(img)
                print(f"  {img}: era={era} ({get_era_name(era)}) conf={conf:.2f}")
            except Exception as e:
                print(f"  {img}: ERROR - {e}")

    return 0 if results["saved"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
