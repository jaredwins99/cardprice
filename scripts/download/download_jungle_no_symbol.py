#!/usr/bin/env python3
"""Download Jungle no-symbol error and normal holo card images for training data."""

import json
import os
import time
import urllib.request

OUT_DIR = "/home/godli/cardprice/data/condition_training/ground_truth_variants/no_symbol"
os.makedirs(OUT_DIR, exist_ok=True)

# Sports Card Investor no-symbol error images (all 16 holos)
SCI_ERROR_URLS = {
    "01_clefable":    ("https://images.production.sportscardinvestor.com/2290_6655_8956_1_No_Symbol", "Clefable", 1),
    "02_electrode":   ("https://images.production.sportscardinvestor.com/2290_5151_8956_2_No_Symbol", "Electrode", 2),
    "03_flareon":     ("https://images.production.sportscardinvestor.com/2290_6474_8956_3_No_Symbol", "Flareon", 3),
    "04_jolteon":     ("https://images.production.sportscardinvestor.com/2290_6541_8956_4_No_Symbol", "Jolteon", 4),
    "05_kangaskhan":  ("https://images.production.sportscardinvestor.com/2290_6653_8956_05_64_No_Symbol", "Kangaskhan", 5),
    "06_mr_mime":     ("https://images.production.sportscardinvestor.com/2290_6656_8956_06_64_No_Symbol", "Mr. Mime", 6),
    "07_nidoqueen":   ("https://images.production.sportscardinvestor.com/2290_6657_8956_07_64_No_Symbol", "Nidoqueen", 7),
    "08_pidgeot":     ("https://images.production.sportscardinvestor.com/2290_7303_8956_08_64_No_Symbol", "Pidgeot", 8),
    "09_pinsir":      ("https://images.production.sportscardinvestor.com/2290_6658_8956_09_64_No_Symbol", "Pinsir", 9),
    "10_scyther":     ("https://images.production.sportscardinvestor.com/2290_6654_8956_10_64_No_Symbol", "Scyther", 10),
    "11_snorlax":     ("https://images.production.sportscardinvestor.com/2290_5500_8956_11_64_No_Symbol", "Snorlax", 11),
    "12_vaporeon":    ("https://images.production.sportscardinvestor.com/2290_6468_8956_12_64_No_Symbol", "Vaporeon", 12),
    "13_venomoth":    ("https://images.production.sportscardinvestor.com/2290_6659_8956_13_64_No_Symbol", "Venomoth", 13),
    "14_victreebel":  ("https://images.production.sportscardinvestor.com/2290_6660_8956_14_64_No_Symbol", "Victreebel", 14),
    "15_vileplume":   ("https://images.production.sportscardinvestor.com/2290_6395_10_15_No_Symbol", "Vileplume", 15),
    "16_wigglytuff":  ("https://images.production.sportscardinvestor.com/2290_6661_8956_16_64_No_Symbol", "Wigglytuff", 16),
}

# Collector's Cache error images (URL-encoded, different scans)
CC_ERROR_URLS = {
    "cc_01_clefable":  ("https://cc-client-assets.nyc3.cdn.digitaloceanspaces.com/photo/collectorscache/file/754423/clefable%20error.jpg", "Clefable", 1),
    "cc_02_electrode": ("https://cc-client-assets.nyc3.cdn.digitaloceanspaces.com/photo/collectorscache/file/6078bf10909d11e68cff5fd11fd2fbce/error%20electrode.png", "Electrode", 2),
    "cc_03_flareon":   ("https://cc-client-assets.nyc3.cdn.digitaloceanspaces.com/photo/collectorscache/file/754425/Flareon%20error.jpg", "Flareon", 3),
    "cc_07_nidoqueen": ("https://cc-client-assets.nyc3.cdn.digitaloceanspaces.com/photo/collectorscache/file/754433/Nidoqueen%20error.jpg", "Nidoqueen", 7),
    "cc_13_venomoth":  ("https://cc-client-assets.nyc3.cdn.digitaloceanspaces.com/photo/collectorscache/file/754445/Venomoth%20error.jpg", "Venomoth", 13),
}

# Normal Jungle holos from pokellector (official card scans WITH the jungle symbol)
NORMAL_URLS = {}
for num, name in [(1,"Clefable"),(2,"Electrode"),(3,"Flareon"),(4,"Jolteon"),(5,"Kangaskhan"),
                   (6,"Mr. Mime"),(7,"Nidoqueen"),(8,"Pidgeot"),(9,"Pinsir"),(10,"Scyther"),
                   (11,"Snorlax"),(12,"Vaporeon"),(13,"Venomoth"),(14,"Victreebel"),(15,"Vileplume"),(16,"Wigglytuff")]:
    key = f"{num:02d}_{name.lower().replace(' ', '_').replace('.', '')}"
    url_name = "Mr-Mime" if name == "Mr. Mime" else name.replace(" ", "").replace(".", "")
    NORMAL_URLS[key] = (f"https://den-cards.pokellector.com/120/{url_name}.JU.{num}.png", name, num)


def download(url, filepath):
    """Download a file with a User-Agent header."""
    if os.path.exists(filepath):
        print(f"  SKIP (exists): {os.path.basename(filepath)}")
        return True
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            content_type = resp.headers.get("Content-Type", "")
            if len(data) < 500:
                print(f"  SKIP (too small {len(data)}b): {url}")
                return False
            with open(filepath, "wb") as f:
                f.write(data)
            print(f"  OK ({len(data):,}b, {content_type}): {os.path.basename(filepath)}")
            return True
    except Exception as e:
        print(f"  FAIL: {url} -> {e}")
        return False


labels = []

# Download SCI no-symbol error images
print("=== Downloading no-symbol error images (SportsCardInvestor) ===")
for key, (url, name, num) in sorted(SCI_ERROR_URLS.items()):
    filepath = os.path.join(OUT_DIR, f"no_symbol_sci_{key}.jpg")
    if download(url, filepath):
        labels.append({
            "file": os.path.basename(filepath),
            "variant": "no_symbol_error",
            "card_name": name,
            "card_number": f"{num}/64",
            "set_id": "base2",
            "source": "sportscardinvestor",
        })
    time.sleep(2)

# Download CC no-symbol error images (already downloaded some)
print("\n=== Downloading no-symbol error images (Collector's Cache) ===")
for key, (url, name, num) in sorted(CC_ERROR_URLS.items()):
    ext = "png" if ".png" in url else "jpg"
    filepath = os.path.join(OUT_DIR, f"no_symbol_{key}.{ext}")
    if download(url, filepath):
        labels.append({
            "file": os.path.basename(filepath),
            "variant": "no_symbol_error",
            "card_name": name,
            "card_number": f"{num}/64",
            "set_id": "base2",
            "source": "collectorscache",
        })
    time.sleep(2)

# Download normal Jungle holos from pokellector (most already downloaded)
print("\n=== Downloading normal Jungle holos (pokellector) ===")
for key, (url, name, num) in sorted(NORMAL_URLS.items()):
    filepath = os.path.join(OUT_DIR, f"normal_pokellector_{key}.png")
    if download(url, filepath):
        labels.append({
            "file": os.path.basename(filepath),
            "variant": "normal",
            "card_name": name,
            "card_number": f"{num}/64",
            "set_id": "base2",
            "source": "pokellector",
        })
    time.sleep(2)

# Save labels
labels_path = os.path.join(OUT_DIR, "labels.json")
with open(labels_path, "w") as f:
    json.dump(labels, f, indent=2)
print(f"\nSaved {len(labels)} labels to {labels_path}")
print(f"  No-symbol error: {sum(1 for l in labels if l['variant'] == 'no_symbol_error')}")
print(f"  Normal: {sum(1 for l in labels if l['variant'] == 'normal')}")
