#!/usr/bin/env python3
"""Download Shadowless vs Unlimited Base Set card images for ground truth training."""

import json
import os
import time
import urllib.request
import urllib.error
import ssl

OUT_DIR = "/home/godli/cardprice/data/condition_training/ground_truth_variants/shadowless"
LABELS_FILE = os.path.join(OUT_DIR, "labels.jsonl")
DELAY = 2.0  # seconds between requests

# SSL context that doesn't verify (some card sites have cert issues)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Image sources organized by variant type
IMAGES = [
    # === SHADOWLESS INDIVIDUAL CARDS (from PokeCardHQ) ===
    {
        "url": "https://cdn.pokecardhq.com/wp-content/uploads/2024/08/Pokemon_TCG_Shadowless_Blastoise-1024x576.jpg?strip=all",
        "filename": "shadowless_blastoise_pokecardhq.jpg",
        "variant": "shadowless",
        "pokemon": "Blastoise",
        "source": "pokecardhq.com",
        "notes": "Individual shadowless Blastoise"
    },
    {
        "url": "https://cdn.pokecardhq.com/wp-content/uploads/2024/08/Pokemon_TCG_Shadowless_Gyarados-2.jpg?strip=all",
        "filename": "shadowless_gyarados_pokecardhq.jpg",
        "variant": "shadowless",
        "pokemon": "Gyarados",
        "source": "pokecardhq.com",
        "notes": "Individual shadowless Gyarados"
    },
    {
        "url": "https://cdn.pokecardhq.com/wp-content/uploads/2024/08/Pokemon_TCG_Shadowless_Pikachu.jpg?strip=all",
        "filename": "shadowless_pikachu_pokecardhq.jpg",
        "variant": "shadowless",
        "pokemon": "Pikachu",
        "source": "pokecardhq.com",
        "notes": "Individual shadowless Pikachu"
    },
    {
        "url": "https://cdn.pokecardhq.com/wp-content/uploads/2024/08/Pokemon_TCG_Shadowless_Alakazam.jpg?strip=all",
        "filename": "shadowless_alakazam_pokecardhq.jpg",
        "variant": "shadowless",
        "pokemon": "Alakazam",
        "source": "pokecardhq.com",
        "notes": "Individual shadowless Alakazam"
    },
    {
        "url": "https://cdn.pokecardhq.com/wp-content/uploads/2024/08/scoop-up-base-set-bs-78.jpg?strip=all",
        "filename": "shadowless_scoop_up_pokecardhq.jpg",
        "variant": "shadowless",
        "pokemon": "Scoop Up (Trainer)",
        "source": "pokecardhq.com",
        "notes": "Individual shadowless Scoop Up trainer"
    },

    # === UNLIMITED INDIVIDUAL CARDS (from PokeCardHQ) ===
    {
        "url": "https://cdn.pokecardhq.com/wp-content/uploads/2024/08/Pokemon_TCG_Gyarados_Base_Set_Unlimited.jpg?strip=all",
        "filename": "unlimited_gyarados_pokecardhq.jpg",
        "variant": "unlimited",
        "pokemon": "Gyarados",
        "source": "pokecardhq.com",
        "notes": "Individual unlimited Gyarados"
    },
    {
        "url": "https://cdn.pokecardhq.com/wp-content/uploads/2024/08/Pokemon_TCG_Pikachu_Base_Set_Unlimited.jpg?strip=all",
        "filename": "unlimited_pikachu_pokecardhq.jpg",
        "variant": "unlimited",
        "pokemon": "Pikachu",
        "source": "pokecardhq.com",
        "notes": "Individual unlimited Pikachu"
    },
    {
        "url": "https://cdn.pokecardhq.com/wp-content/uploads/2024/08/Pokemon_TCG_Alakazam_Base_Set_Unlimited-2.jpg?strip=all",
        "filename": "unlimited_alakazam_pokecardhq.jpg",
        "variant": "unlimited",
        "pokemon": "Alakazam",
        "source": "pokecardhq.com",
        "notes": "Individual unlimited Alakazam"
    },
    {
        "url": "https://cdn.pokecardhq.com/wp-content/uploads/2024/08/Pokemon_TCG_Scoop_Up_Base_Set_Unlimited.jpg?strip=all",
        "filename": "unlimited_scoop_up_pokecardhq.jpg",
        "variant": "unlimited",
        "pokemon": "Scoop Up (Trainer)",
        "source": "pokecardhq.com",
        "notes": "Individual unlimited Scoop Up trainer"
    },

    # === COMPARISON IMAGES (show both variants side-by-side) ===
    {
        "url": "https://cdn.pokecardhq.com/wp-content/uploads/2024/08/Pokemon_TCG_Shadowless_Charizards_1st_and_2nd_printings-1024x576.jpg?strip=all",
        "filename": "comparison_charizard_1st_2nd_pokecardhq.jpg",
        "variant": "comparison",
        "pokemon": "Charizard",
        "source": "pokecardhq.com",
        "notes": "Side-by-side 1st edition vs shadowless Charizard"
    },
    {
        "url": "https://cdn.pokecardhq.com/wp-content/uploads/2024/08/Pokemon-_TCG_Shadowless_Pokemon_copyright_text-1024x576.jpg?strip=all",
        "filename": "comparison_copyright_text_pokecardhq.jpg",
        "variant": "comparison",
        "pokemon": "Copyright detail",
        "source": "pokecardhq.com",
        "notes": "Copyright text comparison shadowless vs unlimited"
    },
    {
        "url": "https://cdn.pokecardhq.com/wp-content/uploads/2024/08/Pokemon_TCG_Shadowless_Pokemon_cards.jpg?strip=all",
        "filename": "comparison_overview_pokecardhq.jpg",
        "variant": "comparison",
        "pokemon": "Multiple",
        "source": "pokecardhq.com",
        "notes": "Overview of shadowless cards"
    },

    # === COMPARISON IMAGES (from TheGamer) ===
    {
        "url": "https://static0.thegamerimages.com/wordpress/wp-content/uploads/2024/09/pokemon-tcg-what-is-a-shadowless-card-base-set-blastoise-comparison.jpg",
        "filename": "comparison_blastoise_thegamer.jpg",
        "variant": "comparison",
        "pokemon": "Blastoise",
        "source": "thegamer.com",
        "notes": "Side-by-side 1st edition vs unlimited Blastoise"
    },
    {
        "url": "https://static0.thegamerimages.com/wordpress/wp-content/uploads/2024/09/pokemon-tcg-what-is-a-shadowless-card-1st-edition-venusaur-blastoise-comparison.jpg",
        "filename": "comparison_venusaur_blastoise_thegamer.jpg",
        "variant": "comparison",
        "pokemon": "Venusaur/Blastoise",
        "source": "thegamer.com",
        "notes": "Side-by-side 1st edition Venusaur and Blastoise"
    },
    {
        "url": "https://static0.thegamerimages.com/wordpress/wp-content/uploads/2024/09/pokemon-tcg-what-is-a-shadowless-card-1st-edition-alakazam-comparison.jpg",
        "filename": "comparison_alakazam_charizard_thegamer.jpg",
        "variant": "comparison",
        "pokemon": "Alakazam/Charizard",
        "source": "thegamer.com",
        "notes": "Side-by-side 1st edition Alakazam and Charizard"
    },

    # === OTHER COMPARISON IMAGES ===
    {
        "url": "https://pokepatch.com/wp-content/uploads/2024/08/7d619-pokepatch-pokemon-tcg-base-set-shadowed-vs-shadowless.png?w=1568",
        "filename": "comparison_shadowed_vs_shadowless_pokepatch.png",
        "variant": "comparison",
        "pokemon": "Multiple",
        "source": "pokepatch.com",
        "notes": "Shadowed vs shadowless comparison"
    },
    {
        "url": "https://www.otia.com/site/assets/files/1763/pokemon-base-set-shadowless-vs-unlimited.1000x0.jpg.webp",
        "filename": "comparison_shadowless_vs_unlimited_otia.webp",
        "variant": "comparison",
        "pokemon": "Multiple",
        "source": "otia.com",
        "notes": "Shadowless vs unlimited comparison"
    },

    # === UNLIMITED BASE SET CARD SCANS (from pkmncards.com) ===
    # pkmncards.com uses unlimited scans by default
    {
        "url": "https://pkmncards.com/wp-content/uploads/charizard-base-set-bs-4.jpg",
        "filename": "unlimited_charizard_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Charizard",
        "source": "pkmncards.com",
        "notes": "Base Set Charizard #4 (unlimited)"
    },
    {
        "url": "https://pkmncards.com/wp-content/uploads/blastoise-base-set-bs-2.jpg",
        "filename": "unlimited_blastoise_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Blastoise",
        "source": "pkmncards.com",
        "notes": "Base Set Blastoise #2 (unlimited)"
    },
    {
        "url": "https://pkmncards.com/wp-content/uploads/venusaur-base-set-bs-15.jpg",
        "filename": "unlimited_venusaur_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Venusaur",
        "source": "pkmncards.com",
        "notes": "Base Set Venusaur #15 (unlimited)"
    },
    {
        "url": "https://pkmncards.com/wp-content/uploads/alakazam-base-set-bs-1.jpg",
        "filename": "unlimited_alakazam_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Alakazam",
        "source": "pkmncards.com",
        "notes": "Base Set Alakazam #1 (unlimited)"
    },
    {
        "url": "https://pkmncards.com/wp-content/uploads/ninetales-base-set-bs-12.jpg",
        "filename": "unlimited_ninetales_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Ninetales",
        "source": "pkmncards.com",
        "notes": "Base Set Ninetales #12 (unlimited)"
    },
    {
        "url": "https://pkmncards.com/wp-content/uploads/mewtwo-base-set-bs-10.jpg",
        "filename": "unlimited_mewtwo_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Mewtwo",
        "source": "pkmncards.com",
        "notes": "Base Set Mewtwo #10 (unlimited)"
    },
    {
        "url": "https://pkmncards.com/wp-content/uploads/hitmonchan-base-set-bs-7.jpg",
        "filename": "unlimited_hitmonchan_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Hitmonchan",
        "source": "pkmncards.com",
        "notes": "Base Set Hitmonchan #7 (unlimited)"
    },
    {
        "url": "https://pkmncards.com/wp-content/uploads/raichu-base-set-bs-14.jpg",
        "filename": "unlimited_raichu_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Raichu",
        "source": "pkmncards.com",
        "notes": "Base Set Raichu #14 (unlimited)"
    },
    {
        "url": "https://pkmncards.com/wp-content/uploads/gyarados-base-set-bs-6.jpg",
        "filename": "unlimited_gyarados_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Gyarados",
        "source": "pkmncards.com",
        "notes": "Base Set Gyarados #6 (unlimited)"
    },
    {
        "url": "https://pkmncards.com/wp-content/uploads/chansey-base-set-bs-3.jpg",
        "filename": "unlimited_chansey_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Chansey",
        "source": "pkmncards.com",
        "notes": "Base Set Chansey #3 (unlimited)"
    },
    {
        "url": "https://pkmncards.com/wp-content/uploads/poliwrath-base-set-bs-13.jpg",
        "filename": "unlimited_poliwrath_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Poliwrath",
        "source": "pkmncards.com",
        "notes": "Base Set Poliwrath #13 (unlimited)"
    },
    {
        "url": "https://pkmncards.com/wp-content/uploads/zapdos-base-set-bs-16.jpg",
        "filename": "unlimited_zapdos_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Zapdos",
        "source": "pkmncards.com",
        "notes": "Base Set Zapdos #16 (unlimited)"
    },
    {
        "url": "https://pkmncards.com/wp-content/uploads/clefairy-base-set-bs-5.jpg",
        "filename": "unlimited_clefairy_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Clefairy",
        "source": "pkmncards.com",
        "notes": "Base Set Clefairy #5 (unlimited)"
    },
    {
        "url": "https://pkmncards.com/wp-content/uploads/nidoking-base-set-bs-11.jpg",
        "filename": "unlimited_nidoking_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Nidoking",
        "source": "pkmncards.com",
        "notes": "Base Set Nidoking #11 (unlimited)"
    },
    {
        "url": "https://pkmncards.com/wp-content/uploads/magneton-base-set-bs-9.jpg",
        "filename": "unlimited_magneton_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Magneton",
        "source": "pkmncards.com",
        "notes": "Base Set Magneton #9 (unlimited)"
    },
    {
        "url": "https://pkmncards.com/wp-content/uploads/machamp-base-set-bs-8.jpg",
        "filename": "unlimited_machamp_pkmncards.jpg",
        "variant": "unlimited",
        "pokemon": "Machamp",
        "source": "pkmncards.com",
        "notes": "Base Set Machamp #8 (unlimited, note: always 1st edition stamp)"
    },

    # === BULBAPEDIA CARD SCANS (unlimited scans) ===
    {
        "url": "https://archives.bulbagarden.net/media/upload/4/4e/CharizardBaseSet4.jpg",
        "filename": "unlimited_charizard_bulbapedia.jpg",
        "variant": "unlimited",
        "pokemon": "Charizard",
        "source": "bulbapedia.bulbagarden.net",
        "notes": "Base Set Charizard #4 full resolution (unlimited)"
    },
    {
        "url": "https://archives.bulbagarden.net/media/upload/7/76/PikachuBaseSet58.jpg",
        "filename": "unlimited_pikachu_bulbapedia.jpg",
        "variant": "unlimited",
        "pokemon": "Pikachu",
        "source": "bulbapedia.bulbagarden.net",
        "notes": "Base Set Pikachu #58 full resolution (unlimited)"
    },

    # === PRICECHARTING SHADOWLESS SCANS ===
    {
        "url": "https://storage.googleapis.com/images.pricecharting.com/0e3bbf4bbd5b02a86e496ede579a072a5fa51c34136cb85cb5d3222e4d11dc9b/240.jpg",
        "filename": "shadowless_charizard_pricecharting.jpg",
        "variant": "shadowless",
        "pokemon": "Charizard",
        "source": "pricecharting.com",
        "notes": "Shadowless Charizard #4 product image"
    },

    # === TRADINGCARDSETS SHADOWLESS SET IMAGES ===
    {
        "url": "https://tradingcardsets.com/cdn/shop/products/image_d81f3a37-3120-4bd0-921b-7b3b2849f8dd.png?v=1651918053",
        "filename": "shadowless_set_overview_tradingcardsets_1.png",
        "variant": "shadowless",
        "pokemon": "Multiple",
        "source": "tradingcardsets.com",
        "notes": "Shadowless complete set image 1"
    },
    {
        "url": "https://tradingcardsets.com/cdn/shop/products/image_a18dc7fe-02f9-43f4-984c-cfa4238657dd.jpg?v=1651918002",
        "filename": "shadowless_set_overview_tradingcardsets_2.jpg",
        "variant": "shadowless",
        "pokemon": "Multiple",
        "source": "tradingcardsets.com",
        "notes": "Shadowless complete set image 2"
    },
    {
        "url": "https://tradingcardsets.com/cdn/shop/products/image_9b075468-9982-442d-bc96-4dee2a3f5a17.jpg?v=1651918002",
        "filename": "shadowless_set_overview_tradingcardsets_3.jpg",
        "variant": "shadowless",
        "pokemon": "Multiple",
        "source": "tradingcardsets.com",
        "notes": "Shadowless complete set image 3"
    },
    {
        "url": "https://tradingcardsets.com/cdn/shop/products/image_b4b5e517-e1ac-4a59-9a2b-60d9ff04fc8e.jpg?v=1651918002",
        "filename": "shadowless_set_overview_tradingcardsets_4.jpg",
        "variant": "shadowless",
        "pokemon": "Multiple",
        "source": "tradingcardsets.com",
        "notes": "Shadowless complete set image 4"
    },
    {
        "url": "https://tradingcardsets.com/cdn/shop/products/image_79101de2-1613-404b-99c1-f2330d8118cd.jpg?v=1651918002",
        "filename": "shadowless_set_overview_tradingcardsets_5.jpg",
        "variant": "shadowless",
        "pokemon": "Multiple",
        "source": "tradingcardsets.com",
        "notes": "Shadowless complete set image 5"
    },
    {
        "url": "https://tradingcardsets.com/cdn/shop/products/image_c1e7f7af-4eeb-465a-ae29-195761f6e495.jpg?v=1651918002",
        "filename": "shadowless_set_overview_tradingcardsets_6.jpg",
        "variant": "shadowless",
        "pokemon": "Multiple",
        "source": "tradingcardsets.com",
        "notes": "Shadowless complete set image 6"
    },

    # === PROGAMINGCREW SHADOWLESS HEADER ===
    {
        "url": "https://progamingcrew.com/cdn/shop/articles/shadowless_1024x1024.jpg?v=1590035127",
        "filename": "shadowless_overview_progamingcrew.jpg",
        "variant": "shadowless",
        "pokemon": "Multiple",
        "source": "progamingcrew.com",
        "notes": "Shadowless cards overview image"
    },
]

# Additional PriceCharting shadowless card URLs to try
PRICECHARTING_SHADOWLESS = [
    ("blastoise-shadowless-2", "Blastoise"),
    ("venusaur-shadowless-15", "Venusaur"),
    ("alakazam-shadowless-1", "Alakazam"),
    ("mewtwo-shadowless-10", "Mewtwo"),
    ("hitmonchan-shadowless-7", "Hitmonchan"),
    ("ninetales-shadowless-12", "Ninetales"),
    ("raichu-shadowless-14", "Raichu"),
    ("gyarados-shadowless-6", "Gyarados"),
    ("zapdos-shadowless-16", "Zapdos"),
    ("chansey-shadowless-3", "Chansey"),
    ("poliwrath-shadowless-13", "Poliwrath"),
    ("clefairy-shadowless-5", "Clefairy"),
    ("nidoking-shadowless-11", "Nidoking"),
    ("magneton-shadowless-9", "Magneton"),
]


def download_image(url, filepath):
    """Download a single image with proper headers."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "https://www.google.com/",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = resp.read()
            if len(data) < 500:
                print(f"  SKIP (too small: {len(data)} bytes): {url}")
                return False
            with open(filepath, "wb") as f:
                f.write(data)
            print(f"  OK ({len(data):,} bytes): {os.path.basename(filepath)}")
            return True
    except (urllib.error.HTTPError, urllib.error.URLError, Exception) as e:
        print(f"  FAIL ({e}): {url}")
        return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = []
    success = 0
    fail = 0

    print(f"Downloading {len(IMAGES)} images to {OUT_DIR}")
    print("=" * 70)

    for img in IMAGES:
        filepath = os.path.join(OUT_DIR, img["filename"])
        if os.path.exists(filepath):
            print(f"  EXISTS: {img['filename']}")
            labels.append(img)
            success += 1
            continue

        ok = download_image(img["url"], filepath)
        if ok:
            labels.append(img)
            success += 1
        else:
            fail += 1
        time.sleep(DELAY)

    # Try PriceCharting shadowless cards
    print("\n" + "=" * 70)
    print("Trying PriceCharting shadowless card images...")
    for slug, pokemon in PRICECHARTING_SHADOWLESS:
        filename = f"shadowless_{pokemon.lower()}_pricecharting.jpg"
        filepath = os.path.join(OUT_DIR, filename)
        if os.path.exists(filepath):
            print(f"  EXISTS: {filename}")
            continue

        # PriceCharting URL pattern
        pc_url = f"https://www.pricecharting.com/game/pokemon-base-set/{slug}"
        # We can't easily get the image URL without fetching the page,
        # so skip duplicates we already have
        if any(l["filename"] == filename for l in labels):
            continue

    print(f"\n{'=' * 70}")
    print(f"Results: {success} downloaded, {fail} failed")
    print(f"Total labels: {len(labels)}")

    # Count by variant
    shadowless_count = sum(1 for l in labels if l["variant"] == "shadowless")
    unlimited_count = sum(1 for l in labels if l["variant"] == "unlimited")
    comparison_count = sum(1 for l in labels if l["variant"] == "comparison")
    print(f"  Shadowless: {shadowless_count}")
    print(f"  Unlimited: {unlimited_count}")
    print(f"  Comparison: {comparison_count}")

    # Write labels.jsonl
    with open(LABELS_FILE, "w") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")
    print(f"\nLabels written to {LABELS_FILE}")


if __name__ == "__main__":
    main()
