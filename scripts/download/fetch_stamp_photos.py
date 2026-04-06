#!/usr/bin/env python3
"""Fetch real stamped Pokemon card images from TCGPlayer seller photos."""

import json
import base64
import os
import sys
import time
import urllib.request

OUT_DIR = "/home/godli/cardprice/data/condition_training/stamps_real"
SOURCES_FILE = os.path.join(OUT_DIR, "sources.jsonl")

# Modern SV-era product IDs (more likely to have seller photos)
PRODUCTS = [
    ("sv9-18/normal", 623445, "Meowscarada", "sv9"),
    ("sv9-3/normal", 623430, "Butterfree", "sv9"),
    ("sv9-100/normal", 623527, "Lokix", "sv9"),
    ("sv9-32/normal", 623459, "Articuno", "sv9"),
    ("sv9-41/normal", 623468, "Wailord", "sv9"),
    ("sv9-116/normal", 623543, "N's Reshiram", "sv9"),
    ("sv9-117/normal", 623544, "Hop's Snorlax", "sv9"),
    ("sv9-21/normal", 623448, "Magmortar", "sv9"),
    ("sv9-29/normal", 623456, "Volcarona", "sv9"),
    ("sv9-34/normal", 623461, "Octillery", "sv9"),
    ("sv9-37/normal", 623464, "Ludicolo", "sv9"),
    ("sv9-102/normal", 623529, "Escavalier", "sv9"),
    ("sv9-105/normal", 623532, "N's Klinklang", "sv9"),
    ("sv9-107/normal", 623534, "Magearna", "sv9"),
    ("sv9-110/normal", 623537, "Copperajah", "sv9"),
    ("sv9-126/normal", 623553, "Cinccino", "sv9"),
    ("sv9-128/normal", 623555, "Noivern", "sv9"),
    ("sv9-132/normal", 623559, "Greedent", "sv9"),
    ("sv9-13/normal", 623440, "Accelgor", "sv9"),
    ("sv9-15/normal", 623442, "Virizion", "sv9"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.tcgplayer.com/",
    "Origin": "https://www.tcgplayer.com",
}

POST_BODY = json.dumps({
    "filters": {"term": {}, "range": {}, "exclude": {}},
    "from": 0,
    "size": 50,
    "context": {"shippingCountry": "US", "cart": {}}
}).encode()


def make_image_url(uuid_str):
    """Convert a TCGPlayer image UUID to a static URL."""
    payload = json.dumps({
        "key": f"{uuid_str}.jpg",
        "edits": {
            "resize": {"height": 800},
            "jpeg": {"quality": 90}
        }
    })
    encoded = base64.b64encode(payload.encode()).decode()
    return f"https://static.tcgplayer.com/{encoded}"


def fetch_listings(pid):
    """Fetch listings for a product ID via POST. Returns parsed JSON or None."""
    url = f"https://mp-search-api.tcgplayer.com/v1/product/{pid}/listings?mpfev=2952"
    req = urllib.request.Request(url, data=POST_BODY, headers=HEADERS, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR fetching listings for {pid}: {e}")
        return None


def download_image(url, filepath):
    """Download an image to filepath. Returns True on success."""
    req = urllib.request.Request(url, headers={
        "User-Agent": HEADERS["User-Agent"],
        "Referer": "https://www.tcgplayer.com/",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            if len(data) < 1000:
                print(f"    Skipping tiny image ({len(data)} bytes)")
                return False
            with open(filepath, "wb") as f:
                f.write(data)
            print(f"    Saved {filepath} ({len(data)} bytes)")
            return True
    except Exception as e:
        print(f"    ERROR downloading: {e}")
        return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total_downloaded = 0

    with open(SOURCES_FILE, "a") as sources_f:
        for card_id, pid, name, set_id in PRODUCTS:
            print(f"\n[{card_id}] {name} (pid={pid})")

            data = fetch_listings(pid)
            if not data:
                time.sleep(3)
                continue

            outer_results = data.get("results", [])
            if not outer_results:
                print(f"  No outer results")
                time.sleep(3)
                continue

            listings = outer_results[0].get("results", [])
            total = outer_results[0].get("totalResults", 0)
            print(f"  {total} total listings, fetched {len(listings)}")

            found_photos = 0
            for listing in listings:
                custom = listing.get("customData", {})
                images = custom.get("images", [])
                if not images:
                    continue

                printing = listing.get("printing", "")
                condition = listing.get("condition", "")
                seller = listing.get("sellerName", "unknown")
                price = listing.get("price", 0)
                printing_str = str(printing).lower()
                is_reverse = "reverse" in printing_str

                print(f"  Custom listing: printing='{printing}', cond='{condition}', "
                      f"seller='{seller}', price={price}, {len(images)} images")
                found_photos += 1

                for i, uuid_str in enumerate(images):
                    img_url = make_image_url(uuid_str)
                    safe_name = name.replace(" ", "_").replace("'", "").lower()
                    filename = f"{set_id}_{safe_name}_{pid}_{uuid_str[:8]}.jpg"
                    filepath = os.path.join(OUT_DIR, filename)

                    if os.path.exists(filepath):
                        print(f"    Already exists: {filename}")
                        continue

                    if download_image(img_url, filepath):
                        total_downloaded += 1
                        record = {
                            "file": filename,
                            "card_id": card_id,
                            "tcg_product_id": pid,
                            "name": name,
                            "set_id": set_id,
                            "printing": str(printing),
                            "condition": str(condition),
                            "seller": seller,
                            "price": price,
                            "uuid": uuid_str,
                            "source": "tcgplayer_seller_photo",
                            "is_reverse_holo": is_reverse,
                        }
                        sources_f.write(json.dumps(record) + "\n")
                        sources_f.flush()

                    time.sleep(1)

            if found_photos == 0:
                print(f"  No custom photo listings found")

            time.sleep(3)  # respectful delay between API calls

    print(f"\n=== Done. Downloaded {total_downloaded} images to {OUT_DIR} ===")


if __name__ == "__main__":
    main()
