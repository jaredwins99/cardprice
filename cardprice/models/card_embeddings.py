"""Card feature encoding for price prediction.

Provides a CardFeatureEncoder with a sklearn-style fit/transform API that
converts raw card attributes into dense numeric feature vectors, plus
convenience functions to build feature matrices from the database and
train a GradientBoostingRegressor price predictor.

Uses: numpy, scikit-learn, pandas.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import text

from cardprice.models.price_predictor import RARITY_ORDER, _parse_card_number

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POKEMON_TYPES: list[str] = [
    "fire",
    "water",
    "grass",
    "electric",
    "psychic",
    "fighting",
    "dark",
    "steel",
    "fairy",
    "dragon",
    "colorless",
]

_RARITY_TO_ORDINAL: dict[str, int] = {r: i + 1 for i, r in enumerate(RARITY_ORDER)}

_SUPERTYPE_LIST: list[str] = ["Pokémon", "Trainer", "Energy"]

# ---------------------------------------------------------------------------
# CardFeatureEncoder
# ---------------------------------------------------------------------------


class CardFeatureEncoder:
    """Encode raw card attributes into a dense numeric feature vector.

    Feature layout (per card):
        [0]              rarity ordinal           (1 dim)
        [1:4]            supertype one-hot         (3 dims: Pokémon, Trainer, Energy)
        [4:15]           pokemon types multi-hot    (11 dims)
        [15]             generation                (1 dim)
        [16]             hp_normalized             (1 dim)
        [17]             set_age_log               (1 dim)
        [18]             card_position             (1 dim)
        [19]             set_series encoded        (1 dim)
        [20]             variant encoded           (1 dim)
        ---
        Total: 21 dimensions (when fitted)

    Follows sklearn fit/transform convention.
    """

    def __init__(self) -> None:
        self._series_encoder = LabelEncoder()
        self._variant_encoder = LabelEncoder()
        self._is_fitted: bool = False
        self._max_hp: float = 1.0
        self._feature_names: list[str] = []

    # -- public API ----------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> CardFeatureEncoder:
        """Fit label encoders on the training data.

        Expected columns: set_series, variant, rarity, supertype,
        types (list[str] | None), generation, hp, release_date,
        card_number, total_cards.
        """
        series_vals = df["set_series"].fillna("Unknown").astype(str).values
        variant_vals = df["variant"].fillna("normal").astype(str).values

        self._series_encoder.fit(np.append(series_vals, ["Unknown"]))
        self._variant_encoder.fit(np.append(variant_vals, ["normal"]))

        hp_col = pd.to_numeric(df["hp"], errors="coerce")
        self._max_hp = float(hp_col.max()) if hp_col.notna().any() else 1.0
        if self._max_hp == 0:
            self._max_hp = 1.0

        self._feature_names = self._build_feature_names()
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a DataFrame of card rows into a feature matrix.

        Returns shape (n_cards, n_features).
        """
        if not self._is_fitted:
            raise RuntimeError("CardFeatureEncoder has not been fitted. Call fit() first.")

        n = len(df)
        features = np.zeros((n, self.n_features), dtype=np.float32)
        today = date.today()

        for i, (_, row) in enumerate(df.iterrows()):
            features[i] = self._encode_row(row, today)

        return features

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    @property
    def n_features(self) -> int:
        """Number of output feature dimensions."""
        return 21

    @property
    def feature_names(self) -> list[str]:
        """Human-readable names for each feature dimension."""
        if not self._feature_names:
            self._feature_names = self._build_feature_names()
        return self._feature_names

    # -- internals -----------------------------------------------------------

    def _build_feature_names(self) -> list[str]:
        names = ["rarity_ordinal"]
        names += [f"supertype_{s}" for s in _SUPERTYPE_LIST]
        names += [f"type_{t}" for t in POKEMON_TYPES]
        names += ["generation", "hp_normalized", "set_age_log", "card_position"]
        names += ["set_series_enc", "variant_enc"]
        return names

    def _encode_row(self, row: Any, today: date) -> np.ndarray:
        """Encode a single row into a 1-D feature vector."""
        vec = np.zeros(self.n_features, dtype=np.float32)
        offset = 0

        # 1. Rarity ordinal (1 dim)
        rarity = row.get("rarity") if hasattr(row, "get") else getattr(row, "rarity", None)
        vec[offset] = _RARITY_TO_ORDINAL.get(rarity, 0) if rarity else 0
        offset += 1

        # 2. Supertype one-hot (3 dims)
        supertype = row.get("supertype") if hasattr(row, "get") else getattr(row, "supertype", None)
        for j, st in enumerate(_SUPERTYPE_LIST):
            if supertype == st:
                vec[offset + j] = 1.0
        offset += len(_SUPERTYPE_LIST)

        # 3. Pokemon types multi-hot (11 dims)
        types_raw = row.get("types") if hasattr(row, "get") else getattr(row, "types", None)
        if types_raw is not None:
            if isinstance(types_raw, str):
                type_list = [t.strip().lower() for t in types_raw.split(",")]
            elif isinstance(types_raw, (list, tuple)):
                type_list = [str(t).strip().lower() for t in types_raw]
            else:
                type_list = []
            for j, pt in enumerate(POKEMON_TYPES):
                if pt in type_list:
                    vec[offset + j] = 1.0
        offset += len(POKEMON_TYPES)

        # 4. Generation (1 dim)
        generation = row.get("generation") if hasattr(row, "get") else getattr(row, "generation", None)
        vec[offset] = float(generation) if generation is not None else 0.0
        offset += 1

        # 5. HP normalized (1 dim)
        hp = row.get("hp") if hasattr(row, "get") else getattr(row, "hp", None)
        hp_val = _safe_float(hp)
        vec[offset] = hp_val / self._max_hp if hp_val is not None else 0.0
        offset += 1

        # 6. Set age in log(days + 1) (1 dim)
        release_date = row.get("release_date") if hasattr(row, "get") else getattr(row, "release_date", None)
        if release_date is not None:
            if isinstance(release_date, str):
                try:
                    from datetime import datetime
                    release_date = datetime.strptime(release_date, "%Y-%m-%d").date()
                except ValueError:
                    release_date = None
            if release_date is not None:
                age_days = max((today - release_date).days, 0)
                vec[offset] = float(np.log1p(age_days))
        offset += 1

        # 7. Card position = card_number / total_cards (1 dim)
        card_number = row.get("card_number") if hasattr(row, "get") else getattr(row, "card_number", None)
        total_cards = row.get("total_cards") if hasattr(row, "get") else getattr(row, "total_cards", None)
        card_num = _parse_card_number(card_number)
        total = _safe_float(total_cards)
        if card_num is not None and total is not None and total > 0:
            vec[offset] = card_num / total
        offset += 1

        # 8. Set series label-encoded (1 dim)
        series = row.get("set_series") if hasattr(row, "get") else getattr(row, "set_series", None)
        series_str = str(series) if series is not None else "Unknown"
        try:
            vec[offset] = float(self._series_encoder.transform([series_str])[0])
        except ValueError:
            vec[offset] = float(self._series_encoder.transform(["Unknown"])[0])
        offset += 1

        # 9. Variant label-encoded (1 dim)
        variant = row.get("variant") if hasattr(row, "get") else getattr(row, "variant", None)
        variant_str = str(variant) if variant is not None else "normal"
        try:
            vec[offset] = float(self._variant_encoder.transform([variant_str])[0])
        except ValueError:
            vec[offset] = float(self._variant_encoder.transform(["normal"])[0])
        offset += 1

        return vec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_field(row: Any, name: str, default: Any = None) -> Any:
    """Get a field from a row that may be a dict or an object."""
    if hasattr(row, "get"):
        return row.get(name, default)
    return getattr(row, name, default)


def _safe_float(val: Any) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

_FEATURE_MATRIX_SQL = text("""
    SELECT
        dc.card_id,
        ds.series        AS set_series,
        ds.release_date,
        ds.total_cards,
        dc.rarity,
        dc.supertype,
        dc.hp,
        dc.card_number,
        dc.variant,
        dp.types,
        dp.generation
    FROM dim_cards dc
    JOIN dim_sets ds    ON ds.set_id = dc.set_id
    LEFT JOIN dim_pokemon dp ON dp.pokemon_id = dc.pokemon_id
""")


def build_feature_matrix(session: Any) -> tuple[np.ndarray, list[str]]:
    """Query dim_cards + dim_sets + dim_pokemon and build encoded feature vectors.

    Parameters
    ----------
    session : SQLAlchemy Session

    Returns
    -------
    (feature_matrix, card_ids)
        feature_matrix : np.ndarray of shape (n_cards, n_features)
        card_ids       : list of card_id strings matching each row
    """
    rows = session.execute(_FEATURE_MATRIX_SQL).fetchall()
    if not rows:
        raise ValueError("No cards found in dim_cards. Run the card catalog loader first.")

    columns = [
        "card_id", "set_series", "release_date", "total_cards",
        "rarity", "supertype", "hp", "card_number", "variant",
        "types", "generation",
    ]
    df = pd.DataFrame(rows, columns=columns)

    logger.info("Building feature matrix for %d cards", len(df))

    card_ids = df["card_id"].tolist()

    encoder = CardFeatureEncoder()
    X = encoder.fit_transform(df)

    logger.info(
        "Feature matrix shape: %s, feature names: %s",
        X.shape, encoder.feature_names,
    )

    return X, card_ids


# ---------------------------------------------------------------------------
# train_price_predictor
# ---------------------------------------------------------------------------

_TRAINING_SQL = text("""
    WITH latest_prices AS (
        SELECT DISTINCT ON (card_id)
            card_id,
            market_price
        FROM fact_market_prices
        WHERE card_id IS NOT NULL
          AND market_price IS NOT NULL
          AND market_price > 0
        ORDER BY card_id, price_date DESC
    )
    SELECT
        dc.card_id,
        ds.series        AS set_series,
        ds.release_date,
        ds.total_cards,
        dc.rarity,
        dc.supertype,
        dc.hp,
        dc.card_number,
        dc.variant,
        dp.types,
        dp.generation,
        lp.market_price
    FROM latest_prices lp
    JOIN dim_cards dc      ON dc.card_id = lp.card_id
    JOIN dim_sets ds       ON ds.set_id  = dc.set_id
    LEFT JOIN dim_pokemon dp ON dp.pokemon_id = dc.pokemon_id
""")


def train_price_predictor(
    session: Any,
    *,
    test_size: float = 0.2,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a GradientBoostingRegressor on log(market_price).

    Queries the database for cards with prices, encodes features using
    CardFeatureEncoder, splits into train/test, fits a GBR, and reports
    R-squared, MAE, and feature importance.

    Parameters
    ----------
    session : SQLAlchemy Session
    test_size : fraction of data for testing
    n_estimators, max_depth, learning_rate : GBR hyperparameters
    random_state : for reproducibility

    Returns
    -------
    dict with keys: model, encoder, metrics, feature_importance
    """
    from cardprice.db.session import SessionLocal

    own_session = False
    if session is None:
        session = SessionLocal()
        own_session = True

    try:
        rows = session.execute(_TRAINING_SQL).fetchall()
        if not rows:
            raise ValueError(
                "No training data found. Ensure fact_market_prices has linked card_ids."
            )

        columns = [
            "card_id", "set_series", "release_date", "total_cards",
            "rarity", "supertype", "hp", "card_number", "variant",
            "types", "generation", "market_price",
        ]
        df = pd.DataFrame(rows, columns=columns)
        df["market_price"] = pd.to_numeric(df["market_price"], errors="coerce")
        df = df[df["market_price"] > 0].copy()

        logger.info(
            "Training data: %d cards, median price $%.2f",
            len(df), df["market_price"].median(),
        )

        # Encode features
        encoder = CardFeatureEncoder()
        X = encoder.fit_transform(df)
        y = np.log(df["market_price"].values.astype(np.float64))

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
        )

        # Fit model
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            subsample=0.8,
            loss="huber",
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)

        # MAE in dollar space (approximate via median transform)
        mae_dollar_test = float(np.median(
            np.abs(np.exp(y_test) - np.exp(y_pred_test))
        ))

        metrics = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "r2_train": round(r2_train, 4),
            "r2_test": round(r2_test, 4),
            "mae_log_train": round(mae_train, 4),
            "mae_log_test": round(mae_test, 4),
            "mae_dollar_test": round(mae_dollar_test, 2),
            "price_median": round(float(df["market_price"].median()), 2),
            "price_mean": round(float(df["market_price"].mean()), 2),
        }

        # Feature importance
        importances = model.feature_importances_
        feature_importance = sorted(
            zip(encoder.feature_names, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        # Report
        print("=" * 60)
        print("Card Embeddings Price Predictor — Results")
        print("=" * 60)
        print(f"  Training samples:  {metrics['n_train']}")
        print(f"  Test samples:      {metrics['n_test']}")
        print(f"  Median price:      ${metrics['price_median']:.2f}")
        print()
        print(f"  R² (train):        {metrics['r2_train']:.4f}")
        print(f"  R² (test):         {metrics['r2_test']:.4f}")
        print(f"  MAE log (train):   {metrics['mae_log_train']:.4f}")
        print(f"  MAE log (test):    {metrics['mae_log_test']:.4f}")
        print(f"  MAE dollar (test): ${metrics['mae_dollar_test']:.2f}")
        print()
        print("  Feature Importance (top 10):")
        for feat, imp in feature_importance[:10]:
            print(f"    {feat:30s}  {imp:.4f}")
        print("=" * 60)

        return {
            "model": model,
            "encoder": encoder,
            "metrics": metrics,
            "feature_importance": feature_importance,
        }

    finally:
        if own_session:
            session.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from cardprice.db.session import SessionLocal

    sess = SessionLocal()
    try:
        train_price_predictor(sess)
    finally:
        sess.close()
