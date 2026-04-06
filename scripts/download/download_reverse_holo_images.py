#!/usr/bin/env python3
"""Download reverse holo training images from various sources."""

import os
import time
import json
import requests
import urllib3

urllib3.disable_warnings()

OUT_DIR = "/home/godli/cardprice/data/condition_training/ground_truth_variants/reverse_holo"
os.makedirs(OUT_DIR, exist_ok=True)

RATE_LIMIT = 2.0  # seconds between downloads

# Each entry: (url, filename, metadata)
IMAGES = [
    # === CODED YELLOW - Reverse Holo Patterns Timeline (one per era) ===
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Legendary-Collection.jpg",
        "codedyellow_legendary_collection.jpg",
        {"variant": "reverse_holo", "era": "legendary_collection", "set_id": "lc", "pattern": "fireworks", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/EX-Series-Rainbow.jpg",
        "codedyellow_ex_series_rainbow.jpg",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_ruby_sapphire", "pattern": "rainbow", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Hidden-Legends.jpg",
        "codedyellow_ex_hidden_legends.jpg",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_hidden_legends", "pattern": "energy_symbols", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/FireRed-LeafGreen.jpg",
        "codedyellow_ex_frlg.jpg",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_frlg", "pattern": "energy_pokeball", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/EX-Deoxys.jpg",
        "codedyellow_ex_deoxys.jpg",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_deoxys", "pattern": "stamped", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Legend-Maker.jpg",
        "codedyellow_ex_legend_maker.jpg",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_legend_maker", "pattern": "stamped_plain", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Diamond-and-Pearl.jpg",
        "codedyellow_diamond_pearl.jpg",
        {"variant": "reverse_holo", "era": "diamond_pearl", "set_id": "dp", "pattern": "pixel", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Platinum.jpg",
        "codedyellow_platinum.jpg",
        {"variant": "reverse_holo", "era": "platinum", "set_id": "pl", "pattern": "shattered_glass", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/HeartGold-Soul-Silver-2.jpg",
        "codedyellow_hgss.jpg",
        {"variant": "reverse_holo", "era": "hgss", "set_id": "hgss", "pattern": "shattered_glass", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Black-and-White.jpg",
        "codedyellow_bw_base.jpg",
        {"variant": "reverse_holo", "era": "black_white", "set_id": "bw", "pattern": "sparkle", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/BW-Emerging-Powers.jpg",
        "codedyellow_bw_emerging.jpg",
        {"variant": "reverse_holo", "era": "black_white", "set_id": "bw_ep", "pattern": "crosshatch", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Black-and-White-Plasma.jpg",
        "codedyellow_bw_plasma.jpg",
        {"variant": "reverse_holo", "era": "black_white", "set_id": "bw_plasma", "pattern": "plasma", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/X-and-Y.jpg",
        "codedyellow_xy.jpg",
        {"variant": "reverse_holo", "era": "xy", "set_id": "xy", "pattern": "confetti", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Sun-and-Moon.jpg",
        "codedyellow_sun_moon.jpg",
        {"variant": "reverse_holo", "era": "sun_moon", "set_id": "sm", "pattern": "confetti", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Sword-and-Shield-1.jpg",
        "codedyellow_sword_shield.jpg",
        {"variant": "reverse_holo", "era": "sword_shield", "set_id": "swsh", "pattern": "type_chevrons", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Scarlet-and-Violet.jpg",
        "codedyellow_scarlet_violet.jpg",
        {"variant": "reverse_holo", "era": "scarlet_violet", "set_id": "sv", "pattern": "type_cobblestone", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Pokeball-PE.jpg",
        "codedyellow_prismatic_pokeball.jpg",
        {"variant": "reverse_holo", "era": "scarlet_violet", "set_id": "sv_prismatic", "pattern": "pokeball", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Master-ball-PE.jpg",
        "codedyellow_prismatic_masterball.jpg",
        {"variant": "reverse_holo", "era": "scarlet_violet", "set_id": "sv_prismatic", "pattern": "master_ball", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Wizards-of-the-coast-first-ever-2.jpg",
        "codedyellow_wotc_first.jpg",
        {"variant": "reverse_holo", "era": "wotc", "set_id": "expedition", "pattern": "first_reverse_holo", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Mega-Evolution-base.jpg",
        "codedyellow_mega_evolution.jpg",
        {"variant": "reverse_holo", "era": "mega_evolution", "set_id": "me", "pattern": "type_symbol", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Ascended-Heroes-Type-Reverse-Holo.jpg",
        "codedyellow_ascended_heroes_type.jpg",
        {"variant": "reverse_holo", "era": "ascended_heroes", "set_id": "ah", "pattern": "type_symbol", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/AH-Love-Ball-Reverse.jpg",
        "codedyellow_ah_love_ball.jpg",
        {"variant": "reverse_holo", "era": "ascended_heroes", "set_id": "ah", "pattern": "love_ball", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/AH-Team-Rocket-Reverse.jpg",
        "codedyellow_ah_team_rocket.jpg",
        {"variant": "reverse_holo", "era": "ascended_heroes", "set_id": "ah", "pattern": "team_rocket", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/Black-Bolt-Poke-Ball.jpg",
        "codedyellow_black_bolt_pokeball.jpg",
        {"variant": "reverse_holo", "era": "black_bolt", "set_id": "bb", "pattern": "pokeball", "source": "codedyellow"}
    ),
    (
        "https://www.codedyellow.com/wp-content/uploads/2023/04/White-Flare-Master-Ball-Reverse-Holo.jpg",
        "codedyellow_white_flare_masterball.jpg",
        {"variant": "reverse_holo", "era": "white_flare", "set_id": "wf", "pattern": "master_ball", "source": "codedyellow"}
    ),

    # === BULBAPEDIA - Reverse holo examples from different sets ===
    (
        "https://archives.bulbagarden.net/media/upload/a/a0/OnixLegendaryCollection84Reverse.jpg",
        "bulba_onix_lc_fireworks.jpg",
        {"variant": "reverse_holo", "era": "legendary_collection", "set_id": "lc", "pattern": "fireworks", "source": "bulbapedia"}
    ),
    (
        "https://archives.bulbagarden.net/media/upload/8/83/FeraligatrExpedition12.jpg",
        "bulba_feraligatr_expedition.jpg",
        {"variant": "reverse_holo", "era": "e_series", "set_id": "expedition", "pattern": "e_series", "source": "bulbapedia"}
    ),
    (
        "https://archives.bulbagarden.net/media/upload/c/c3/ExploudEXHiddenLegends6.jpg",
        "bulba_exploud_ex_hl.jpg",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_hidden_legends", "pattern": "energy_symbols", "source": "bulbapedia"}
    ),
    (
        "https://archives.bulbagarden.net/media/upload/d/d3/PidgeotEXFireRedLeafGreen10.jpg",
        "bulba_pidgeot_ex_frlg.jpg",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_frlg", "pattern": "energy_pokeball", "source": "bulbapedia"}
    ),
    (
        "https://archives.bulbagarden.net/media/upload/1/10/DarkTyranitarEXTeamRocketReturns19.jpg",
        "bulba_dark_tyranitar_ex_trr.jpg",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_trr", "pattern": "stamped", "source": "bulbapedia"}
    ),
    (
        "https://archives.bulbagarden.net/media/upload/0/06/AltariaEXDeoxys1.jpg",
        "bulba_altaria_ex_deoxys.jpg",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_deoxys", "pattern": "stamped", "source": "bulbapedia"}
    ),
    (
        "https://archives.bulbagarden.net/media/upload/0/07/KyogreEXEmerald6.jpg",
        "bulba_kyogre_ex_emerald.jpg",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_emerald", "pattern": "stamped", "source": "bulbapedia"}
    ),
    (
        "https://archives.bulbagarden.net/media/upload/a/a1/MeganiumEXUnseenForces9.jpg",
        "bulba_meganium_ex_uf.jpg",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_unseen_forces", "pattern": "stamped", "source": "bulbapedia"}
    ),
    (
        "https://archives.bulbagarden.net/media/upload/5/57/MarowakEXDeltaSpecies10.jpg",
        "bulba_marowak_ex_delta.jpg",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_delta_species", "pattern": "stamped_plain", "source": "bulbapedia"}
    ),
    (
        "https://archives.bulbagarden.net/media/upload/9/98/OmastarEXLegendMaker23.jpg",
        "bulba_omastar_ex_lm.jpg",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_legend_maker", "pattern": "stamped_plain", "source": "bulbapedia"}
    ),
    (
        "https://archives.bulbagarden.net/media/upload/b/b2/HippowdonDiamondPearl29.jpg",
        "bulba_hippowdon_dp.jpg",
        {"variant": "reverse_holo", "era": "diamond_pearl", "set_id": "dp", "pattern": "pixel", "source": "bulbapedia"}
    ),

    # === ELITE FOURUM - EX era stamped reverse holos (high-res scans) ===
    (
        "https://efour.b-cdn.net/uploads/default/original/3X/f/8/f8a7c53e871ca1080bce88c0d3d87c5448f33069.png",
        "efour_trr_stamped_01.png",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_team_rocket_returns", "pattern": "stamped", "source": "elitefourum"}
    ),
    (
        "https://efour.b-cdn.net/uploads/default/original/3X/6/e/6e668426697ee7faf0eea87b996eba6d43c65f80.png",
        "efour_trr_stamped_02.png",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_team_rocket_returns", "pattern": "stamped", "source": "elitefourum"}
    ),
    (
        "https://efour.b-cdn.net/uploads/default/original/3X/c/1/c1079f55c033a91a01332d7c087691d3912b8f9f.png",
        "efour_trr_stamped_03.png",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_team_rocket_returns", "pattern": "stamped", "source": "elitefourum"}
    ),
    (
        "https://efour.b-cdn.net/uploads/default/original/3X/8/2/82a043010eab817ac81e1b062236a835fbe01aca.jpeg",
        "efour_deoxys_stamped_01.jpeg",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_deoxys", "pattern": "stamped", "source": "elitefourum"}
    ),
    (
        "https://efour.b-cdn.net/uploads/default/original/3X/4/b/4b8daabde4b77a20cbc14fac190ae527bd6e22d0.jpeg",
        "efour_deoxys_stamped_02.jpeg",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_deoxys", "pattern": "stamped", "source": "elitefourum"}
    ),
    (
        "https://efour.b-cdn.net/uploads/default/original/3X/e/a/ea1e0f9bf58df2a3c1428b90c7a0c6ecd379fc38.webp",
        "efour_emerald_stamped_01.webp",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_emerald", "pattern": "stamped", "source": "elitefourum"}
    ),
    (
        "https://efour.b-cdn.net/uploads/default/original/3X/c/9/c9bb320ef3624b5a45d937ee5306ddc5ccf9fddf.webp",
        "efour_unseen_forces_stamped_01.webp",
        {"variant": "reverse_holo", "era": "ex", "set_id": "ex_unseen_forces", "pattern": "stamped", "source": "elitefourum"}
    ),

    # === CARD GAMER - Holo vs Reverse Holo comparisons ===
    (
        "https://cardgamer.com/wp-content/uploads/2023/09/holo-vs-reverse-holo-pokemon-cards-1024x576.webp",
        "cardgamer_holo_vs_reverse_comparison.webp",
        {"variant": "comparison", "era": "mixed", "set_id": "mixed", "pattern": "comparison", "source": "cardgamer"}
    ),
    (
        "https://cardgamer.com/wp-content/uploads/2023/09/holo-foil-pokemon-cards-1024x576.webp",
        "cardgamer_holo_foil.webp",
        {"variant": "holofoil", "era": "mixed", "set_id": "mixed", "pattern": "standard_holo", "source": "cardgamer"}
    ),
    (
        "https://cardgamer.com/wp-content/uploads/2023/09/reverse-holo-pokemon-cards-1024x576.webp",
        "cardgamer_reverse_holo.webp",
        {"variant": "reverse_holo", "era": "mixed", "set_id": "mixed", "pattern": "reverse_holo", "source": "cardgamer"}
    ),
    (
        "https://cardgamer.com/wp-content/uploads/2023/09/reverse-holo-foil-rarity-1024x576.webp",
        "cardgamer_reverse_holo_rarity.webp",
        {"variant": "reverse_holo", "era": "mixed", "set_id": "mixed", "pattern": "reverse_holo_rarity", "source": "cardgamer"}
    ),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Referer": "https://www.google.com/",
}

def download_image(url, filename):
    filepath = os.path.join(OUT_DIR, filename)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        print(f"  SKIP (exists): {filename}")
        return True
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, verify=False)
        if resp.status_code == 200 and len(resp.content) > 500:
            with open(filepath, "wb") as f:
                f.write(resp.content)
            print(f"  OK ({len(resp.content):,} bytes): {filename}")
            return True
        else:
            print(f"  FAIL (status={resp.status_code}, size={len(resp.content)}): {filename}")
            return False
    except Exception as e:
        print(f"  ERROR: {filename} - {e}")
        return False

def main():
    print(f"Downloading {len(IMAGES)} images to {OUT_DIR}")
    print(f"Rate limit: {RATE_LIMIT}s between downloads\n")

    metadata = {}
    success = 0
    fail = 0

    for i, (url, filename, meta) in enumerate(IMAGES):
        print(f"[{i+1}/{len(IMAGES)}] {url[:80]}...")
        if download_image(url, filename):
            metadata[filename] = meta
            success += 1
        else:
            fail += 1
        if i < len(IMAGES) - 1:
            time.sleep(RATE_LIMIT)

    # Save metadata
    meta_path = os.path.join(OUT_DIR, "labels.json")
    metadata_existing = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata_existing = json.load(f)
    metadata_existing.update(metadata)
    with open(meta_path, "w") as f:
        json.dump(metadata_existing, f, indent=2)

    print(f"\nDone: {success} downloaded, {fail} failed")
    print(f"Labels saved to {meta_path}")
    print(f"Total labeled images: {len(metadata_existing)}")

    # Print era coverage
    eras = set()
    patterns = set()
    for m in metadata_existing.values():
        eras.add(m.get("era", "unknown"))
        patterns.add(m.get("pattern", "unknown"))
    print(f"\nEras covered: {sorted(eras)}")
    print(f"Patterns covered: {sorted(patterns)}")

if __name__ == "__main__":
    main()
