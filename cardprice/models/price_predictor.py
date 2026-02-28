"""Baseline price prediction model for Pokemon cards.

Uses card features (set, rarity, type, HP, variant, etc.) and a
GradientBoostingRegressor to predict current market price.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rarity ordinal encoding (ascending value)
# ---------------------------------------------------------------------------

RARITY_ORDER: list[str] = [
    "Common",
    "Uncommon",
    "Rare",
    "Rare Holo",
    "Rare Holo EX",
    "Rare Holo GX",
    "Rare Holo V",
    "Rare VMAX",
    "Rare VSTAR",
    "Rare Holo VMAX",
    "Rare Ultra",
    "Rare Rainbow",
    "Rare Secret",
    "Rare Shiny",
    "Rare Shining",
    "Amazing Rare",
    "LEGEND",
    "Promo",
    # ex-era and newer
    "Double Rare",
    "Ultra Rare",
    "Illustration Rare",
    "Special Illustration Rare",
    "Hyper Rare",
    "ACE SPEC Rare",
    "Shiny Rare",
    "Shiny Ultra Rare",
]

_RARITY_TO_ORDINAL: dict[str, int] = {r: i + 1 for i, r in enumerate(RARITY_ORDER)}

# Feature column definitions
CATEGORICAL_FEATURES = ["set_series", "supertype", "variant"]
ORDINAL_FEATURES = ["rarity_ordinal"]  # already numeric after encoding
NUMERIC_FEATURES = [
    "set_age_days",
    "hp",
    "card_position",
    "has_pokemon",
    "rarity_ordinal",
]

# ---------------------------------------------------------------------------
# 1. build_training_data
# ---------------------------------------------------------------------------

_TRAINING_DATA_SQL = text("""
    WITH latest_prices AS (
        SELECT DISTINCT ON (card_id)
            card_id,
            market_price,
            price_date
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
        dc.rarity,
        dc.supertype,
        dc.hp,
        dc.card_number,
        ds.total_cards,
        dc.variant,
        lp.market_price,
        lp.price_date
    FROM latest_prices lp
    JOIN dim_cards dc ON dc.card_id = lp.card_id
    JOIN dim_sets ds  ON ds.set_id  = dc.set_id
""")


def _parse_card_number(card_number: str | None) -> float | None:
    """Extract numeric portion from card_number (e.g. '4' from '4', '123a' -> 123, 'SV042' -> 42)."""
    if card_number is None:
        return None
    m = re.search(r"(\d+)", str(card_number))
    return float(m.group(1)) if m else None


def build_training_data(session: Session) -> pd.DataFrame:
    """Query the database and build a feature DataFrame for training.

    Returns a DataFrame with feature columns and a 'market_price' target.
    """
    rows = session.execute(_TRAINING_DATA_SQL).fetchall()
    if not rows:
        raise ValueError("No training data found. Ensure fact_market_prices has linked card_ids.")

    df = pd.DataFrame(rows, columns=[
        "card_id", "set_series", "release_date", "rarity", "supertype",
        "hp", "card_number", "total_cards", "variant", "market_price", "price_date",
    ])

    logger.info("Raw training rows: %d", len(df))

    # --- Feature engineering ---

    # set_age_days: days since set release
    today = date.today()
    df["set_age_days"] = df["release_date"].apply(
        lambda d: (today - d).days if d is not None else None
    )

    # rarity_ordinal
    df["rarity_ordinal"] = df["rarity"].map(_RARITY_TO_ORDINAL).fillna(0).astype(int)

    # card_position: card_number / total_cards
    df["card_number_numeric"] = df["card_number"].apply(_parse_card_number)
    df["card_position"] = np.where(
        (df["total_cards"].notna()) & (df["total_cards"] > 0),
        df["card_number_numeric"] / df["total_cards"],
        np.nan,
    )

    # has_pokemon: binary
    df["has_pokemon"] = (df["supertype"] == "Pokémon").astype(int)

    # Fill missing categoricals
    df["set_series"] = df["set_series"].fillna("Unknown")
    df["supertype"] = df["supertype"].fillna("Unknown")
    df["variant"] = df["variant"].fillna("normal")

    # Convert market_price to float
    df["market_price"] = df["market_price"].astype(float)

    # Drop rows with zero or negative price (shouldn't exist but be safe)
    df = df[df["market_price"] > 0].copy()

    logger.info(
        "Training data ready: %d rows, %.1f%% have HP, median price $%.2f",
        len(df),
        df["hp"].notna().mean() * 100,
        df["market_price"].median(),
    )

    return df


# ---------------------------------------------------------------------------
# 2. train_model
# ---------------------------------------------------------------------------

def _build_preprocessor() -> ColumnTransformer:
    """Build the sklearn ColumnTransformer for the feature set."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=10, sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def train_model(
    df: pd.DataFrame,
    *,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train a GradientBoostingRegressor on the feature DataFrame.

    Parameters
    ----------
    df : DataFrame from build_training_data()
    n_estimators, max_depth, learning_rate : GBR hyperparameters
    random_state : for reproducibility

    Returns
    -------
    (model_pipeline, metrics_dict)
        model_pipeline: fitted sklearn Pipeline (preprocessor + regressor)
        metrics_dict: R², MAE, RMSE, feature importance ranking
    """
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df["market_price"].values

    # Log-transform the target for better distribution (prices are right-skewed)
    y_log = np.log1p(y)

    preprocessor = _build_preprocessor()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            subsample=0.8,
            loss="huber",  # robust to outliers
        )),
    ])

    # 5-fold cross-validation
    logger.info("Running 5-fold cross-validation...")
    cv_results = cross_validate(
        pipeline, X, y_log,
        cv=5,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
        return_train_score=True,
    )

    metrics = {
        "cv_r2_mean": float(np.mean(cv_results["test_r2"])),
        "cv_r2_std": float(np.std(cv_results["test_r2"])),
        "cv_mae_log_mean": float(-np.mean(cv_results["test_neg_mean_absolute_error"])),
        "cv_rmse_log_mean": float(-np.mean(cv_results["test_neg_root_mean_squared_error"])),
        "train_r2_mean": float(np.mean(cv_results["train_r2"])),
        "n_samples": len(y),
        "price_median": float(np.median(y)),
        "price_mean": float(np.mean(y)),
        "price_std": float(np.std(y)),
    }

    logger.info(
        "CV R²: %.3f (±%.3f), MAE(log): %.3f, RMSE(log): %.3f",
        metrics["cv_r2_mean"], metrics["cv_r2_std"],
        metrics["cv_mae_log_mean"], metrics["cv_rmse_log_mean"],
    )

    # Fit on full data
    pipeline.fit(X, y_log)

    # Extract feature names after preprocessing
    feature_names = _get_feature_names(pipeline)
    importances = pipeline.named_steps["regressor"].feature_importances_

    importance_ranking = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    metrics["feature_importance"] = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in importance_ranking
    ]

    logger.info("Top 5 features: %s", importance_ranking[:5])

    return pipeline, metrics


def _get_feature_names(pipeline: Pipeline) -> list[str]:
    """Extract feature names from the fitted pipeline's preprocessor."""
    preprocessor = pipeline.named_steps["preprocessor"]
    names: list[str] = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            names.extend(columns)
        elif name == "cat":
            encoder = transformer.named_steps["encoder"]
            names.extend(encoder.get_feature_names_out(columns).tolist())

    return names


# ---------------------------------------------------------------------------
# 3. predict_price
# ---------------------------------------------------------------------------

_CARD_FEATURES_SQL = text("""
    SELECT
        dc.card_id,
        ds.series        AS set_series,
        ds.release_date,
        dc.rarity,
        dc.supertype,
        dc.hp,
        dc.card_number,
        ds.total_cards,
        dc.variant
    FROM dim_cards dc
    JOIN dim_sets ds ON ds.set_id = dc.set_id
    WHERE dc.card_id = :card_id
""")


def predict_price(
    card_id: str,
    model: Pipeline,
    session: Session,
) -> dict[str, Any]:
    """Predict the market price for a specific card.

    Parameters
    ----------
    card_id : dim_cards.card_id (e.g. "base1-4/holofoil")
    model   : fitted Pipeline from train_model()
    session : open SQLAlchemy Session

    Returns
    -------
    dict with predicted_price, confidence_interval, features_used
    """
    row = session.execute(_CARD_FEATURES_SQL, {"card_id": card_id}).fetchone()
    if row is None:
        raise ValueError(f"Card not found: {card_id}")

    today = date.today()
    release_date = row.release_date
    set_age_days = (today - release_date).days if release_date else None
    rarity_ordinal = _RARITY_TO_ORDINAL.get(row.rarity, 0)
    card_number_numeric = _parse_card_number(row.card_number)
    total_cards = row.total_cards
    card_position = (
        card_number_numeric / total_cards
        if card_number_numeric and total_cards and total_cards > 0
        else None
    )
    has_pokemon = 1 if row.supertype == "Pokémon" else 0

    features = {
        "set_age_days": set_age_days,
        "hp": row.hp,
        "card_position": card_position,
        "has_pokemon": has_pokemon,
        "rarity_ordinal": rarity_ordinal,
        "set_series": row.set_series or "Unknown",
        "supertype": row.supertype or "Unknown",
        "variant": row.variant or "normal",
    }

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = pd.DataFrame([features])[feature_cols]

    # Predict in log space, then invert
    y_log_pred = model.predict(X)[0]
    predicted_price = float(np.expm1(y_log_pred))

    # Estimate confidence interval using the regressor's staged_predict
    # (approximate: use training residual std as proxy)
    regressor = model.named_steps["regressor"]
    # Use the quantile-based approach: predict at different stages
    # For a simpler baseline, use ±1 std of the training loss
    train_loss_std = float(regressor.train_score_[-1]) ** 0.5 if hasattr(regressor, "train_score_") else 0.3
    # Rough confidence interval in log space
    log_std = max(train_loss_std, 0.15)  # floor at 15% relative
    ci_low = float(np.expm1(y_log_pred - 1.96 * log_std))
    ci_high = float(np.expm1(y_log_pred + 1.96 * log_std))

    return {
        "card_id": card_id,
        "predicted_price": round(max(predicted_price, 0.01), 2),
        "confidence_interval": {
            "low": round(max(ci_low, 0.01), 2),
            "high": round(ci_high, 2),
        },
        "features_used": features,
    }


# ---------------------------------------------------------------------------
# 4. analyze_price_drivers
# ---------------------------------------------------------------------------

def analyze_price_drivers(
    model: Pipeline,
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """Extract and rank feature importance from the trained model.

    Parameters
    ----------
    model : fitted Pipeline from train_model()
    top_n : number of top features to return

    Returns
    -------
    List of (feature_name, importance) tuples, sorted descending.
    """
    feature_names = _get_feature_names(model)
    importances = model.named_steps["regressor"].feature_importances_

    ranking = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranking[:top_n]


# ---------------------------------------------------------------------------
# CLI / standalone runner
# ---------------------------------------------------------------------------

def run_baseline(session: Session | None = None) -> dict[str, Any]:
    """End-to-end: build data, train, report metrics.

    Convenience function for quick evaluation from the CLI or notebooks.
    """
    from cardprice.db.session import SessionLocal

    if session is None:
        session = SessionLocal()
        own_session = True
    else:
        own_session = False

    try:
        print("Building training data...")
        df = build_training_data(session)
        print(f"  {len(df)} cards with price data")
        print(f"  Median price: ${df['market_price'].median():.2f}")
        print(f"  Mean price:   ${df['market_price'].mean():.2f}")
        print(f"  Rarity distribution:")
        for rarity, count in df["rarity"].value_counts().head(10).items():
            print(f"    {rarity}: {count}")
        print()

        print("Training GradientBoosting model...")
        model, metrics = train_model(df)
        print(f"  CV R²:         {metrics['cv_r2_mean']:.3f} (±{metrics['cv_r2_std']:.3f})")
        print(f"  CV MAE (log):  {metrics['cv_mae_log_mean']:.3f}")
        print(f"  CV RMSE (log): {metrics['cv_rmse_log_mean']:.3f}")
        print(f"  Train R²:      {metrics['train_r2_mean']:.3f}")
        print()

        print("Top price drivers:")
        drivers = analyze_price_drivers(model)
        for feat, imp in drivers[:10]:
            print(f"  {feat:40s}  {imp:.4f}")
        print()

        # Sample predictions
        sample_ids = df.sample(min(5, len(df)), random_state=42)["card_id"].tolist()
        print("Sample predictions:")
        for cid in sample_ids:
            try:
                pred = predict_price(cid, model, session)
                actual = float(df.loc[df["card_id"] == cid, "market_price"].iloc[0])
                print(
                    f"  {cid:40s}  predicted=${pred['predicted_price']:>8.2f}  "
                    f"actual=${actual:>8.2f}  "
                    f"CI=[${pred['confidence_interval']['low']:.2f}, "
                    f"${pred['confidence_interval']['high']:.2f}]"
                )
            except Exception as e:
                print(f"  {cid}: error - {e}")

        return {"df": df, "model": model, "metrics": metrics}

    finally:
        if own_session:
            session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_baseline()
