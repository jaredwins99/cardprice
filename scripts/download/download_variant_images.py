#!/usr/bin/env python3
"""Download variant card images from pokemontcg.io API for sets ex1, ex2, ex3.

For each card with tcgplayer.prices keys beyond 'normal', download the large image
and save as data/card_images/{set_id}/{card_id}_{variant_key}.png

Only downloads if the file doesn't already exist.
"""

import os
import time
import requests

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "card_images")
API_URL = "https://api.pokemontcg.io/v2/cards"

# Map tcgplayer price keys to filename suffixes
VARIANT_KEY_MAP = {
    "reverseHolofoil": "reverse_holofoil",
    "1stEditionHolofoil": "1st_edition_holofoil",
    "1stEditionNormal": "1st_edition_normal",
    "holofoil": "holofoil",
    "unlimitedHolofoil": "unlimited_holofoil",
    # 'normal' is already downloaded as _normal.png
}

SETS = ["ex1", "ex2", "ex3"]


def download_set(set_id: str) -> None:
    out_dir = os.path.join(BASE_DIR, set_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Fetching cards for set {set_id} ===")
    resp = requests.get(API_URL, params={"q": f"set.id:{set_id}", "pageSize": 250})
    resp.raise_for_status()
    cards = resp.json()["data"]
    print(f"  Got {len(cards)} cards")

    downloads = 0
    skipped = 0

    for card in cards:
        card_id = card["id"]
        image_url = card.get("images", {}).get("large")
        if not image_url:
            continue

        tcg_prices = card.get("tcgplayer", {}).get("prices", {})

        for price_key, file_suffix in VARIANT_KEY_MAP.items():
            if price_key not in tcg_prices:
                continue
            if price_key == "normal":
                continue

            out_path = os.path.join(out_dir, f"{card_id}_{file_suffix}.png")
            if os.path.exists(out_path):
                skipped += 1
                continue

            print(f"  Downloading {card_id} [{file_suffix}] ...")
            try:
                img_resp = requests.get(image_url, timeout=30)
                img_resp.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(img_resp.content)
                downloads += 1
            except Exception as e:
                print(f"    ERROR: {e}")

    print(f"  Set {set_id}: {downloads} downloaded, {skipped} already existed")


def main():
    for i, set_id in enumerate(SETS):
        if i > 0:
            print("  (rate limit pause 1s)")
            time.sleep(1)
        download_set(set_id)

    print("\nDone!")


if __name__ == "__main__":
    main()
