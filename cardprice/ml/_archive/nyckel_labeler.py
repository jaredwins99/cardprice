"""Nyckel API integration for pseudo-labeling card condition.

Calls the Nyckel pretrained condition classifier and maps its labels to
our 5 TCG conditions (NM / LP / MP / HP / DMG).

SETUP REQUIRED:
  1. Sign up at https://www.nyckel.com/signup (Google/GitHub SSO)
  2. Go to Settings > API Credentials to create a client_id + client_secret
  3. Set environment variables:
       export NYCKEL_CLIENT_ID="your_client_id"
       export NYCKEL_CLIENT_SECRET="your_client_secret"
  4. Optionally set NYCKEL_FUNCTION_ID to override the default classifier.

PRETRAINED CLASSIFIER NOTES (researched 2026-03-16):
  - Nyckel has NO pretrained Pokemon/trading card condition classifier.
  - The closest pretrained option is "beanie-baby-condition" (20 labels
    including Mint, Near Mint, Damaged, Excellent, Fair, Good, etc.).
  - For best results, create a CUSTOM classifier on Nyckel's platform
    trained on Pokemon card images with your desired labels, then set
    NYCKEL_FUNCTION_ID to that function's ID.
  - Rate limit: 25 RPS. Free tier has limited invocations (exact quota
    is behind their login wall; expect ~1000 free invocations/month).

API DOCS: https://www.nyckel.com/docs
"""

import base64
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# --- Configuration ---

# Default to the beanie-baby-condition pretrained classifier (closest
# available pretrained model). Override with a custom function ID for
# a Pokemon card condition classifier you've trained on Nyckel.
DEFAULT_FUNCTION_ID = "beanie-baby-condition"

NYCKEL_TOKEN_URL = "https://www.nyckel.com/connect/token"
NYCKEL_API_BASE = "https://www.nyckel.com/v1/functions"

# --- Label mapping: Nyckel labels → TCG condition ---
# The beanie-baby-condition classifier has ~20 labels. We map them to
# our 5 TCG grades. Labels are matched case-insensitively.
# If you use a custom classifier with different labels, update this map.

NYCKEL_TO_TCG = {
    # Near Mint
    "mint": "NM",
    "near mint": "NM",
    "gem mint": "NM",
    "never played with": "NM",
    "new": "NM",
    "pristine": "NM",
    "sealed": "NM",
    # Lightly Played
    "excellent": "LP",
    "very good": "LP",
    "like new": "LP",
    "lightly played": "LP",
    "light play": "LP",
    # Moderately Played
    "good": "MP",
    "displayed": "MP",
    "moderate": "MP",
    "moderately played": "MP",
    "played": "MP",
    # Heavily Played
    "fair": "HP",
    "heavily played": "HP",
    "heavy play": "HP",
    "poor": "HP",
    "well loved": "HP",
    "common": "MP",  # ambiguous, default to MP
    # Damaged
    "damaged": "DMG",
    "defective": "DMG",
    "parts only": "DMG",
    # Direct TCG labels (if user trains a custom classifier with these)
    "nm": "NM",
    "lp": "LP",
    "mp": "MP",
    "hp": "HP",
    "dmg": "DMG",
    "near mint (nm)": "NM",
    "lightly played (lp)": "LP",
    "moderately played (mp)": "MP",
    "heavily played (hp)": "HP",
    "damaged (dmg)": "DMG",
    # PSA-style numeric grades
    "10": "NM",
    "9": "NM",
    "8": "LP",
    "7": "LP",
    "6": "MP",
    "5": "MP",
    "4": "HP",
    "3": "HP",
    "2": "DMG",
    "1": "DMG",
    # Defect-type labels (e.g. if classifier identifies specific damage)
    "corner wear": "LP",
    "edge wear": "LP",
    "surface scratches": "MP",
    "creased": "HP",
    "torn": "DMG",
    "water damage": "DMG",
    "stained": "HP",
}

# Confidence thresholds for the TCG grade mapping
# If Nyckel confidence is below this, we mark the prediction as low-confidence
MIN_CONFIDENCE = 0.30


class NyckelError(Exception):
    """Raised when the Nyckel API returns an error."""
    pass


class NyckelAuthError(NyckelError):
    """Raised when authentication fails (missing or invalid credentials)."""
    pass


# --- Token cache ---
_token_cache: dict = {"access_token": None, "expires_at": 0.0}


def _get_credentials() -> tuple[str, str]:
    """Read Nyckel credentials from environment variables."""
    client_id = os.environ.get("NYCKEL_CLIENT_ID", "")
    client_secret = os.environ.get("NYCKEL_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise NyckelAuthError(
            "NYCKEL_CLIENT_ID and NYCKEL_CLIENT_SECRET environment variables "
            "must be set. Sign up at https://www.nyckel.com/signup and create "
            "API credentials under Settings > API Credentials."
        )
    return client_id, client_secret


def _get_access_token() -> str:
    """Get a valid Bearer token, refreshing if expired.

    Nyckel tokens last 3600 seconds. We cache and refresh with a 60s buffer.
    """
    now = time.time()
    if _token_cache["access_token"] and now < _token_cache["expires_at"]:
        return _token_cache["access_token"]

    client_id, client_secret = _get_credentials()

    resp = requests.post(
        NYCKEL_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=15,
    )
    if resp.status_code != 200:
        raise NyckelAuthError(
            f"Failed to obtain Nyckel access token: {resp.status_code} {resp.text}"
        )

    data = resp.json()
    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = now + data.get("expires_in", 3600) - 60
    return data["access_token"]


def _map_nyckel_label(label_name: str) -> str:
    """Map a Nyckel label to a TCG condition grade.

    Falls back to "MP" if the label is unknown (safe middle-ground).
    """
    normalized = label_name.strip().lower()
    tcg = NYCKEL_TO_TCG.get(normalized)
    if tcg:
        return tcg

    # Partial match fallback: check if any key is contained in the label
    for key, grade in NYCKEL_TO_TCG.items():
        if key in normalized:
            return grade

    logger.warning("Unknown Nyckel label '%s', defaulting to MP", label_name)
    return "MP"


def predict_condition(
    image_path: str,
    function_id: Optional[str] = None,
    return_all_labels: bool = False,
) -> dict:
    """Predict card condition using the Nyckel API.

    Parameters
    ----------
    image_path : str
        Path to the card image file (JPEG/PNG).
    function_id : str, optional
        Nyckel function ID to invoke. Defaults to NYCKEL_FUNCTION_ID env var,
        then falls back to DEFAULT_FUNCTION_ID ("beanie-baby-condition").
    return_all_labels : bool
        If True, include all label confidences in the response.

    Returns
    -------
    dict with keys:
        predicted_label : str
            TCG condition grade: "NM", "LP", "MP", "HP", or "DMG"
        confidence : float
            Nyckel's confidence in the top prediction (0-1).
        nyckel_raw : dict
            Raw Nyckel API response for debugging.
        nyckel_label : str
            The original Nyckel label name before mapping.
        low_confidence : bool
            True if confidence < MIN_CONFIDENCE threshold.
        all_labels : list[dict], optional
            All label confidences (only if return_all_labels=True).

    Raises
    ------
    NyckelAuthError
        If credentials are missing or invalid.
    NyckelError
        If the API call fails.
    FileNotFoundError
        If image_path doesn't exist.
    """
    image_path = str(Path(image_path).resolve())
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    func_id = (
        function_id
        or os.environ.get("NYCKEL_FUNCTION_ID")
        or DEFAULT_FUNCTION_ID
    )

    token = _get_access_token()
    invoke_url = f"{NYCKEL_API_BASE}/{func_id}/invoke"

    # Read image and encode as data URI
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime = mime_types.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"

    # Build request params
    params = {}
    if return_all_labels:
        params["labelCount"] = 20  # get all label confidences

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {"data": data_uri}

    try:
        resp = requests.post(
            invoke_url,
            json=body,
            headers=headers,
            params=params,
            timeout=30,
        )
    except requests.RequestException as e:
        raise NyckelError(f"Nyckel API request failed: {e}") from e

    if resp.status_code == 401:
        # Token expired mid-request; clear cache and retry once
        _token_cache["access_token"] = None
        _token_cache["expires_at"] = 0.0
        token = _get_access_token()
        headers["Authorization"] = f"Bearer {token}"
        try:
            resp = requests.post(
                invoke_url,
                json=body,
                headers=headers,
                params=params,
                timeout=30,
            )
        except requests.RequestException as e:
            raise NyckelError(f"Nyckel API retry failed: {e}") from e

    if resp.status_code == 429:
        raise NyckelError(
            "Nyckel rate limit exceeded (25 RPS). Retry after a short delay."
        )

    if resp.status_code != 200:
        raise NyckelError(
            f"Nyckel API error {resp.status_code}: {resp.text}"
        )

    raw = resp.json()

    # Parse response
    nyckel_label = raw.get("labelName", "unknown")
    confidence = raw.get("confidence", 0.0)
    tcg_label = _map_nyckel_label(nyckel_label)

    result = {
        "predicted_label": tcg_label,
        "confidence": round(confidence, 4),
        "nyckel_raw": raw,
        "nyckel_label": nyckel_label,
        "low_confidence": confidence < MIN_CONFIDENCE,
    }

    # If all labels requested, map them all
    if return_all_labels and "labelConfidences" in raw:
        result["all_labels"] = [
            {
                "nyckel_label": lc["labelName"],
                "tcg_label": _map_nyckel_label(lc["labelName"]),
                "confidence": round(lc["confidence"], 4),
            }
            for lc in raw["labelConfidences"]
        ]

    return result


def predict_condition_batch(
    image_paths: list[str],
    function_id: Optional[str] = None,
    delay: float = 0.05,
) -> list[dict]:
    """Predict condition for multiple card images.

    Processes sequentially with a small delay to respect rate limits.

    Parameters
    ----------
    image_paths : list[str]
        Paths to card images.
    function_id : str, optional
        Override Nyckel function ID.
    delay : float
        Seconds between API calls (default 0.05 = 20 RPS, under 25 limit).

    Returns
    -------
    list[dict]
        One prediction dict per image (same format as predict_condition).
        On error, returns dict with "error" key instead.
    """
    results = []
    for i, path in enumerate(image_paths):
        try:
            result = predict_condition(path, function_id=function_id)
            results.append(result)
        except Exception as e:
            logger.error("Failed to predict %s: %s", path, e)
            results.append({
                "predicted_label": None,
                "confidence": 0.0,
                "error": str(e),
                "image_path": path,
            })

        if i < len(image_paths) - 1 and delay > 0:
            time.sleep(delay)

    return results
