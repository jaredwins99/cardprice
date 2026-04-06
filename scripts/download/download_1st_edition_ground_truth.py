#!/usr/bin/env python3
"""
Download 1st Edition and Unlimited Pokemon card images for ground truth comparison.

Sources:
1. pokemontcg.io API - Unlimited edition hi-res digital scans
2. archive.org CBZ - Mix of 1st edition and unlimited physical scans
3. pkmncards.com - High quality card scans (various editions)

Target cards: Iconic holos and commons from Base Set, Jungle, Fossil,
Team Rocket, Gym Heroes/Challenge, Neo Genesis/Discovery/Revelation/Destiny
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

BASE_DIR = Path("/home/godli/cardprice/data/condition_training/ground_truth_variants")
FIRST_ED_DIR = BASE_DIR / "1st_edition"
UNLIMITED_DIR = BASE_DIR / "unlimited"
LABELS_PATH = FIRST_ED_DIR / "labels.jsonl"
CBZ_CACHE = Path("/tmp/pokemon_cbz")

FIRST_ED_DIR.mkdir(parents=True, exist_ok=True)
UNLIMITED_DIR.mkdir(parents=True, exist_ok=True)
CBZ_CACHE.mkdir(parents=True, exist_ok=True)

# Target cards - iconic cards that commonly appear in both editions
TARGET_CARDS = {
    "base1": {
        "1": "Alakazam",
        "2": "Blastoise",
        "4": "Charizard",
        "5": "Clefairy",
        "6": "Gyarados",
        "7": "Hitmonchan",
        "8": "Machamp",
        "9": "Magneton",
        "10": "Mewtwo",
        "11": "Nidoking",
        "12": "Ninetales",
        "13": "Poliwrath",
        "15": "Venusaur",
        "16": "Zapdos",
        "25": "Pikachu",  # Red cheeks pikachu
        "46": "Charmander",
        "58": "Pikachu",  # Yellow cheeks variant
        "63": "Squirtle",
        "69": "Weedle",
    },
    "base2": {  # Jungle
        "1": "Clefable",
        "2": "Electrode",
        "3": "Flareon",
        "4": "Jolteon",
        "5": "Kangaskhan",
        "6": "Mr. Mime",
        "7": "Nidoqueen",
        "8": "Pidgeot",
        "9": "Pinsir",
        "10": "Scyther",
        "11": "Snorlax",
        "12": "Vaporeon",
        "13": "Venomoth",
        "14": "Victreebel",
        "15": "Vileplume",
        "16": "Wigglytuff",
    },
    "base3": {  # Fossil
        "1": "Aerodactyl",
        "2": "Articuno",
        "3": "Ditto",
        "4": "Dragonite",
        "5": "Gengar",
        "6": "Haunter",
        "7": "Hitmonlee",
        "8": "Hypno",
        "9": "Kabutops",
        "10": "Lapras",
        "11": "Magneton",
        "12": "Moltres",
        "13": "Muk",
        "14": "Raichu",
        "15": "Zapdos",
    },
    "base5": {  # Team Rocket
        "1": "Dark Alakazam",
        "2": "Dark Arbok",
        "3": "Dark Blastoise",
        "4": "Dark Charizard",
        "5": "Dark Dragonite",
        "6": "Dark Dugtrio",
        "7": "Dark Golbat",
        "8": "Dark Gyarados",
        "9": "Dark Hypno",
        "10": "Dark Machamp",
        "11": "Dark Magneton",
        "12": "Dark Slowbro",
        "13": "Dark Vileplume",
        "14": "Dark Weezing",
        "15": "Here Comes Team Rocket!",
    },
    "neo1": {  # Neo Genesis
        "1": "Ampharos",
        "2": "Azumarill",
        "3": "Bellossom",
        "4": "Feraligatr",
        "5": "Heracross",
        "6": "Jumpluff",
        "7": "Kingdra",
        "8": "Lugia",
        "9": "Meganium",
        "10": "Meganium",
        "11": "Pichu",
        "17": "Typhlosion",
    },
}

RATE_LIMIT = 2.0  # seconds between requests


def download_file(url: str, dest: Path, rate_limit: float = RATE_LIMIT) -> bool:
    """Download a file with rate limiting and error handling."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0 (Pokemon Card Research)")
        resp = urllib.request.urlopen(req, timeout=30)
        data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        print(f"  [ok] {dest.name} ({len(data)} bytes)")
        time.sleep(rate_limit)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"  [FAIL] {dest.name}: {e}")
        return False


def write_label(labels_file, filename: str, variant: str, set_id: str, card_name: str,
                card_number: str, source_url: str):
    """Append a label entry to the JSONL file."""
    entry = {
        "image": filename,
        "variant": variant,
        "set_id": set_id,
        "card_name": card_name,
        "card_number": card_number,
        "source_url": source_url,
    }
    labels_file.write(json.dumps(entry) + "\n")


def download_unlimited_from_api():
    """Download unlimited edition cards from pokemontcg.io API."""
    print("\n=== Downloading Unlimited Edition from pokemontcg.io ===")

    labels_path = UNLIMITED_DIR / "labels.jsonl"
    existing_labels = set()
    if labels_path.exists():
        with open(labels_path) as f:
            for line in f:
                entry = json.loads(line)
                existing_labels.add(entry["image"])

    with open(labels_path, "a") as lf:
        for set_id, cards in TARGET_CARDS.items():
            print(f"\nSet: {set_id}")
            for card_num, card_name in cards.items():
                filename = f"{set_id}_{card_num}_{card_name.replace(' ', '_').replace('.', '').replace('!', '')}.png"
                dest = UNLIMITED_DIR / filename

                if filename in existing_labels:
                    print(f"  [skip] {filename} already labeled")
                    continue

                url = f"https://images.pokemontcg.io/{set_id}/{card_num}_hires.png"
                if download_file(url, dest):
                    write_label(lf, filename, "unlimited", set_id, card_name,
                                card_num, url)


def download_archive_cbz(set_name: str, set_id: str):
    """Download and extract a CBZ from archive.org."""
    cbz_filename = f"{set_name}.cbz"
    cbz_path = CBZ_CACHE / cbz_filename
    extract_dir = CBZ_CACHE / f"{set_name}_extracted"

    if not cbz_path.exists():
        url = f"https://archive.org/download/Pokemon-Trading-Card-Game-Card-Scans/{urllib.request.quote(set_name)}.cbz"
        print(f"  Downloading {set_name} CBZ...")
        if not download_file(url, cbz_path, rate_limit=0):
            return None
    else:
        print(f"  CBZ exists: {cbz_path}")

    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(cbz_path) as z:
                z.extractall(extract_dir)
            print(f"  Extracted to {extract_dir}")
        except zipfile.BadZipFile:
            print(f"  [FAIL] Bad zip file: {cbz_path}")
            return None

    return extract_dir


def process_archive_images():
    """Download archive.org CBZ files and process them."""
    print("\n=== Processing archive.org CBZ files ===")

    set_map = {
        "Base Set": "base1",
        "Jungle": "base2",
        "Fossil": "base3",
        "Team Rocket": "base5",
        "Gym Heroes": "gym1",
        "Gym Challenge": "gym2",
        "Neo Genesis": "neo1",
        "Neo Discovery": "neo2",
        "Neo Revelation": "neo3",
        "Neo Destiny": "neo4",
    }

    for set_name, set_id in set_map.items():
        print(f"\n--- {set_name} ({set_id}) ---")
        extract_dir = download_archive_cbz(set_name, set_id)
        if extract_dir is None:
            continue

        # Find all JPGs in the extracted directory
        jpg_files = []
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    jpg_files.append(os.path.join(root, f))

        print(f"  Found {len(jpg_files)} images")

        # Copy to our directory with proper names
        # These need manual classification of 1st ed vs unlimited
        archive_dir = BASE_DIR / "archive_raw" / set_id
        archive_dir.mkdir(parents=True, exist_ok=True)

        for src_path in sorted(jpg_files):
            fname = os.path.basename(src_path)
            dest = archive_dir / fname
            if not dest.exists():
                import shutil
                shutil.copy2(src_path, dest)

        print(f"  Copied to {archive_dir}")


def download_pkmncards():
    """Download card scans from pkmncards.com (these show edition variants)."""
    print("\n=== Downloading from pkmncards.com ===")

    # pkmncards URL pattern: name-set-code-number.jpg
    # These are generally unlimited but sometimes 1st edition
    pkmncards_map = {
        # Base Set
        ("base1", "4", "Charizard"): "charizard-base-set-bs-4",
        ("base1", "2", "Blastoise"): "blastoise-base-set-bs-2",
        ("base1", "15", "Venusaur"): "venusaur-base-set-bs-15",
        ("base1", "1", "Alakazam"): "alakazam-base-set-bs-1",
        ("base1", "10", "Mewtwo"): "mewtwo-base-set-bs-10",
        ("base1", "16", "Zapdos"): "zapdos-base-set-bs-16",
        ("base1", "6", "Gyarados"): "gyarados-base-set-bs-6",
        ("base1", "12", "Ninetales"): "ninetales-base-set-bs-12",
        ("base1", "7", "Hitmonchan"): "hitmonchan-base-set-bs-7",
        ("base1", "8", "Machamp"): "machamp-base-set-bs-8",
        # Jungle
        ("base2", "10", "Scyther"): "scyther-jungle-ju-10",
        ("base2", "11", "Snorlax"): "snorlax-jungle-ju-11",
        ("base2", "3", "Flareon"): "flareon-jungle-ju-3",
        ("base2", "4", "Jolteon"): "jolteon-jungle-ju-4",
        ("base2", "12", "Vaporeon"): "vaporeon-jungle-ju-12",
        ("base2", "9", "Pinsir"): "pinsir-jungle-ju-9",
        # Fossil
        ("base3", "2", "Articuno"): "articuno-fossil-fo-2",
        ("base3", "4", "Dragonite"): "dragonite-fossil-fo-4",
        ("base3", "5", "Gengar"): "gengar-fossil-fo-5",
        ("base3", "10", "Lapras"): "lapras-fossil-fo-10",
        ("base3", "12", "Moltres"): "moltres-fossil-fo-12",
        # Team Rocket
        ("base5", "4", "Dark Charizard"): "dark-charizard-team-rocket-tr-4",
        ("base5", "3", "Dark Blastoise"): "dark-blastoise-team-rocket-tr-3",
        ("base5", "5", "Dark Dragonite"): "dark-dragonite-team-rocket-tr-5",
        ("base5", "8", "Dark Gyarados"): "dark-gyarados-team-rocket-tr-8",
        # Neo Genesis
        ("neo1", "8", "Lugia"): "lugia-neo-genesis-n1-9",
        ("neo1", "17", "Typhlosion"): "typhlosion-neo-genesis-n1-17",
        ("neo1", "4", "Feraligatr"): "feraligatr-neo-genesis-n1-4",
    }

    pkmncards_dir = BASE_DIR / "pkmncards_raw"
    pkmncards_dir.mkdir(parents=True, exist_ok=True)

    labels_path = pkmncards_dir / "labels.jsonl"
    existing = set()
    if labels_path.exists():
        with open(labels_path) as f:
            for line in f:
                existing.add(json.loads(line)["image"])

    with open(labels_path, "a") as lf:
        for (set_id, num, name), slug in pkmncards_map.items():
            filename = f"{slug}.jpg"
            if filename in existing:
                print(f"  [skip] {filename}")
                continue

            url = f"https://pkmncards.com/wp-content/uploads/{slug}.jpg"
            dest = pkmncards_dir / filename
            if download_file(url, dest):
                write_label(lf, filename, "unknown", set_id, name, num, url)


def download_1st_edition_from_known_urls():
    """Download known 1st edition card images from various sources."""
    print("\n=== Downloading known 1st Edition images ===")

    # CGC has some 1st edition card images
    cgc_images = [
        {
            "url": "https://s3.amazonaws.com/ccg-corporate-production/news-images/Beedrill (1) TB20220606111816771.png",
            "name": "Beedrill",
            "set_id": "base2",
            "number": "17",
        },
        {
            "url": "https://s3.amazonaws.com/ccg-corporate-production/news-images/Pikachu_BaseSet TB20220603143058377.png",
            "name": "Pikachu",
            "set_id": "base1",
            "number": "58",
        },
        {
            "url": "https://s3.amazonaws.com/ccg-corporate-production/news-images/DarkTyphlosion_NeoDestiny TB20220603143032444.png",
            "name": "Dark Typhlosion",
            "set_id": "neo4",
            "number": "10",
        },
    ]

    labels_path = FIRST_ED_DIR / "labels.jsonl"
    existing = set()
    if labels_path.exists():
        with open(labels_path) as f:
            for line in f:
                existing.add(json.loads(line)["image"])

    with open(labels_path, "a") as lf:
        for img_info in cgc_images:
            safe_name = img_info["name"].replace(" ", "_").replace(".", "").replace("'", "")
            filename = f"cgc_{img_info['set_id']}_{img_info['number']}_{safe_name}.png"
            if filename in existing:
                print(f"  [skip] {filename}")
                continue

            dest = FIRST_ED_DIR / filename
            if download_file(img_info["url"], dest):
                write_label(lf, filename, "1st_edition", img_info["set_id"],
                            img_info["name"], img_info["number"], img_info["url"])


def main():
    print("=== 1st Edition Ground Truth Image Collector ===")
    print(f"1st Edition dir: {FIRST_ED_DIR}")
    print(f"Unlimited dir: {UNLIMITED_DIR}")

    if "--unlimited" in sys.argv or "--all" in sys.argv:
        download_unlimited_from_api()

    if "--archive" in sys.argv or "--all" in sys.argv:
        process_archive_images()

    if "--pkmncards" in sys.argv or "--all" in sys.argv:
        download_pkmncards()

    if "--1st-known" in sys.argv or "--all" in sys.argv:
        download_1st_edition_from_known_urls()

    if len(sys.argv) == 1:
        print("\nUsage: python download_1st_edition_ground_truth.py [--unlimited] [--archive] [--pkmncards] [--1st-known] [--all]")
        print("  --unlimited   Download unlimited edition from pokemontcg.io")
        print("  --archive     Download & extract archive.org CBZ files")
        print("  --pkmncards   Download scans from pkmncards.com")
        print("  --1st-known   Download known 1st edition images")
        print("  --all         All of the above")


if __name__ == "__main__":
    main()
