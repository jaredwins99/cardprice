#!/usr/bin/env python3
"""Download prerelease-stamped Pokemon card images for variant training data."""

import json
import os
import time
import urllib.request
import ssl

OUTPUT_DIR = "/home/godli/cardprice/data/condition_training/ground_truth_variants/prerelease"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SSL context for downloads
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

CARDS = [
    # === PRERELEASE STAMPED (classic "PRERELEASE" text stamp on artwork) ===
    # From pokemonflashfire.com - these have the "PRERELEASE" foil stamp
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/aerodactyl-62-1st-edition-pre-release-pokemon-card.jpg",
        "filename": "aerodactyl_fossil_prerelease.jpg",
        "variant": "prerelease", "set_id": "fossil", "stamp_text": "PRERELEASE",
        "card_name": "Aerodactyl", "card_number": "1/62"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/mistys-seadra-9-prerelease-pokemon-card.jpg",
        "filename": "mistys_seadra_gym_heroes_prerelease.jpg",
        "variant": "prerelease", "set_id": "gym-heroes", "stamp_text": "PRERELEASE",
        "card_name": "Misty's Seadra", "card_number": "9/132"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/arcanine-12-99-prerelease-holo-pokemon-promo.jpg",
        "filename": "arcanine_next_destinies_prerelease.jpg",
        "variant": "prerelease", "set_id": "next-destinies", "stamp_text": "PRERELEASE",
        "card_name": "Arcanine", "card_number": "12/99"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/raichu-27-prerelease-pokemon-card.jpg",
        "filename": "raichu_arceus_prerelease.jpg",
        "variant": "prerelease", "set_id": "arceus", "stamp_text": "PRERELEASE",
        "card_name": "Raichu", "card_number": "27/99"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/chesnaught-pre-release-break-through-xy68.jpg",
        "filename": "chesnaught_xy_prerelease.jpg",
        "variant": "prerelease", "set_id": "breakthrough", "stamp_text": "set_logo",
        "card_name": "Chesnaught", "card_number": "XY68"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/lucario-53-prerelease-pokemon-card.jpg",
        "filename": "lucario_platinum_prerelease.jpg",
        "variant": "prerelease", "set_id": "platinum", "stamp_text": "PRERELEASE",
        "card_name": "Lucario", "card_number": "53/127"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/luxio-52-prerelease-pokemon-card.jpg",
        "filename": "luxio_diamond_pearl_prerelease.jpg",
        "variant": "prerelease", "set_id": "diamond-pearl", "stamp_text": "PRERELEASE",
        "card_name": "Luxio", "card_number": "52/130"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/gabite-48-prerelease-pokemon-card.jpg",
        "filename": "gabite_mysterious_treasures_prerelease.jpg",
        "variant": "prerelease", "set_id": "mysterious-treasures", "stamp_text": "PRERELEASE",
        "card_name": "Gabite", "card_number": "48/123"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/kirlia-53-prerelease-pokemon-card.jpg",
        "filename": "kirlia_secret_wonders_prerelease.jpg",
        "variant": "prerelease", "set_id": "secret-wonders", "stamp_text": "PRERELEASE",
        "card_name": "Kirlia", "card_number": "53/132"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/ivysaur-35-prerelease-pokemon-card.jpg",
        "filename": "ivysaur_crystal_guardians_prerelease.jpg",
        "variant": "prerelease", "set_id": "crystal-guardians", "stamp_text": "PRERELEASE",
        "card_name": "Ivysaur", "card_number": "35/100"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/grumpig-29-prerelease-pokemon-card.jpg",
        "filename": "grumpig_emerald_prerelease.jpg",
        "variant": "prerelease", "set_id": "emerald", "stamp_text": "PRERELEASE",
        "card_name": "Grumpig", "card_number": "29/106"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/gyrados-32-prerelease-pokemon-card.jpg",
        "filename": "gyarados_dragon_prerelease.jpg",
        "variant": "prerelease", "set_id": "dragon", "stamp_text": "PRERELEASE",
        "card_name": "Gyarados", "card_number": "32/97"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/metang-49-prerelease-pokemon-card.jpg",
        "filename": "metang_delta_species_prerelease.jpg",
        "variant": "prerelease", "set_id": "delta-species", "stamp_text": "PRERELEASE",
        "card_name": "Metang", "card_number": "49/113"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/manectric-38-prerelease-pokemon-card.jpg",
        "filename": "manectric_deoxys_prerelease.jpg",
        "variant": "prerelease", "set_id": "deoxys", "stamp_text": "PRERELEASE",
        "card_name": "Manectric", "card_number": "38/107"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/milotic-70-prerelease-pokemon-card.jpg",
        "filename": "milotic_supreme_victors_prerelease.jpg",
        "variant": "prerelease", "set_id": "supreme-victors", "stamp_text": "PRERELEASE",
        "card_name": "Milotic", "card_number": "70/147"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/piloswine-46-prerelease-pokemon-card.jpg",
        "filename": "piloswine_stormfront_prerelease.jpg",
        "variant": "prerelease", "set_id": "stormfront", "stamp_text": "PRERELEASE",
        "card_name": "Piloswine", "card_number": "46/100"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/porygon2-49-prerelease-pokemon-card.jpg",
        "filename": "porygon2_great_encounters_prerelease.jpg",
        "variant": "prerelease", "set_id": "great-encounters", "stamp_text": "PRERELEASE",
        "card_name": "Porygon2", "card_number": "49/106"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/mothim-47-prerelease-pokemon-card.jpg",
        "filename": "mothim_majestic_dawn_prerelease.jpg",
        "variant": "prerelease", "set_id": "majestic-dawn", "stamp_text": "PRERELEASE",
        "card_name": "Mothim", "card_number": "47/100"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/pichuprerelease.jpg",
        "filename": "pichu_hgss_prerelease.jpg",
        "variant": "prerelease", "set_id": "hgss", "stamp_text": "PRERELEASE",
        "card_name": "Pichu", "card_number": "28/123"
    },
    # BW era - set logo stamps
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/gigalith-53-prerelease-pokemon-card.jpg",
        "filename": "gigalith_emerging_powers_prerelease.jpg",
        "variant": "prerelease", "set_id": "emerging-powers", "stamp_text": "set_logo",
        "card_name": "Gigalith", "card_number": "53/98"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/victini-43-prerelease-pokemon-card.jpg",
        "filename": "victini_noble_victories_prerelease.jpg",
        "variant": "prerelease", "set_id": "noble-victories", "stamp_text": "set_logo",
        "card_name": "Victini", "card_number": "43/101"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/leafeon-17-prerelease-pokemon-card.jpg",
        "filename": "leafeon_undaunted_prerelease.jpg",
        "variant": "prerelease", "set_id": "undaunted", "stamp_text": "PRERELEASE",
        "card_name": "Leafeon", "card_number": "17/90"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/snorlax-23-prerelease-pokemon-card.jpg",
        "filename": "snorlax_call_of_legends_prerelease.jpg",
        "variant": "prerelease", "set_id": "call-of-legends", "stamp_text": "PRERELEASE",
        "card_name": "Snorlax", "card_number": "23/95"
    },
    # XY era
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/kingdra-xy39-holo-rare-prerelease-primal-clash.jpg",
        "filename": "kingdra_primal_clash_prerelease.jpg",
        "variant": "prerelease", "set_id": "primal-clash", "stamp_text": "set_logo",
        "card_name": "Kingdra", "card_number": "XY39"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/trevenant-xy94-prerelease-holo-pokemon-card.jpg",
        "filename": "trevenant_steam_siege_prerelease.jpg",
        "variant": "prerelease", "set_id": "steam-siege", "stamp_text": "set_logo",
        "card_name": "Trevenant", "card_number": "XY94"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/tropius-52-prerelease-pokemon-card.jpg",
        "filename": "tropius_furious_fists_prerelease.jpg",
        "variant": "prerelease", "set_id": "furious-fists", "stamp_text": "set_logo",
        "card_name": "Tropius", "card_number": "52/111"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/wartortle-50-prerelease-pokemon-card.jpg",
        "filename": "wartortle_plasma_blast_prerelease.jpg",
        "variant": "prerelease", "set_id": "plasma-blast", "stamp_text": "set_logo",
        "card_name": "Wartortle", "card_number": "50/112"
    },
    # BW era promos
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/volcarona-bw40-holo-prerelease-pokemon-card.jpg",
        "filename": "volcarona_bw_prerelease.jpg",
        "variant": "prerelease", "set_id": "bw-promos", "stamp_text": "set_logo",
        "card_name": "Volcarona", "card_number": "BW40"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/porygon-z-bw84-prerelease-pokemon-card.jpg",
        "filename": "porygonz_bw_prerelease.jpg",
        "variant": "prerelease", "set_id": "bw-promos", "stamp_text": "set_logo",
        "card_name": "Porygon-Z", "card_number": "BW84"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/tornadus-ex-bw96-prerelease-pokemon-card.jpg",
        "filename": "tornadus_ex_bw_prerelease.jpg",
        "variant": "prerelease", "set_id": "bw-promos", "stamp_text": "set_logo",
        "card_name": "Tornadus EX", "card_number": "BW96"
    },

    # === STAFF STAMPED ===
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/arcanine-12-99-prerelease-staff-holo-pokemon-promo.jpg",
        "filename": "arcanine_next_destinies_staff.jpg",
        "variant": "staff", "set_id": "next-destinies", "stamp_text": "STAFF",
        "card_name": "Arcanine", "card_number": "12/99"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/gabite-48-prerelease-staff-pokemon-card.jpg",
        "filename": "gabite_mysterious_treasures_staff.jpg",
        "variant": "staff", "set_id": "mysterious-treasures", "stamp_text": "STAFF",
        "card_name": "Gabite", "card_number": "48/123"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/pichuprereleasestaff.jpg",
        "filename": "pichu_hgss_staff.jpg",
        "variant": "staff", "set_id": "hgss", "stamp_text": "STAFF",
        "card_name": "Pichu", "card_number": "28/123"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/snorlax-23-prerelease-staff-pokemon-card.jpg",
        "filename": "snorlax_call_of_legends_staff.jpg",
        "variant": "staff", "set_id": "call-of-legends", "stamp_text": "STAFF",
        "card_name": "Snorlax", "card_number": "23/95"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/raichu-27-prerelease-staff-pokemon-card.jpg",
        "filename": "raichu_arceus_staff.jpg",
        "variant": "staff", "set_id": "arceus", "stamp_text": "STAFF",
        "card_name": "Raichu", "card_number": "27/99"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/kirlia-53-prerelease-staff-pokemon-card.jpg",
        "filename": "kirlia_secret_wonders_staff.jpg",
        "variant": "staff", "set_id": "secret-wonders", "stamp_text": "STAFF",
        "card_name": "Kirlia", "card_number": "53/132"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/lucario-53-staff-prerelease-pokemon-card.jpg",
        "filename": "lucario_platinum_staff.jpg",
        "variant": "staff", "set_id": "platinum", "stamp_text": "STAFF",
        "card_name": "Lucario", "card_number": "53/127"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/milotic-70-staff-prerelease-pokemon-card.jpg",
        "filename": "milotic_supreme_victors_staff.jpg",
        "variant": "staff", "set_id": "supreme-victors", "stamp_text": "STAFF",
        "card_name": "Milotic", "card_number": "70/147"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/piloswine-46-staff-prerelease-pokemon-card.jpg",
        "filename": "piloswine_stormfront_staff.jpg",
        "variant": "staff", "set_id": "stormfront", "stamp_text": "STAFF",
        "card_name": "Piloswine", "card_number": "46/100"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/leafeon-17-prerelease-staff-pokemon-card.jpg",
        "filename": "leafeon_undaunted_staff.jpg",
        "variant": "staff", "set_id": "undaunted", "stamp_text": "STAFF",
        "card_name": "Leafeon", "card_number": "17/90"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/victini-43-prerelease-staff-pokemon-card.jpg",
        "filename": "victini_noble_victories_staff.jpg",
        "variant": "staff", "set_id": "noble-victories", "stamp_text": "STAFF",
        "card_name": "Victini", "card_number": "43/101"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/tropius-52-prerelease-staff-pokemon-card.jpg",
        "filename": "tropius_furious_fists_staff.jpg",
        "variant": "staff", "set_id": "furious-fists", "stamp_text": "STAFF",
        "card_name": "Tropius", "card_number": "52/111"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/trevenant-xy94-staff-prerelease-holo-pokemon-card.jpg",
        "filename": "trevenant_steam_siege_staff.jpg",
        "variant": "staff", "set_id": "steam-siege", "stamp_text": "STAFF",
        "card_name": "Trevenant", "card_number": "XY94"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/tornadus-ex-bw96-staff-prerelease-legendary-treasures.jpg",
        "filename": "tornadus_ex_bw_staff.jpg",
        "variant": "staff", "set_id": "bw-promos", "stamp_text": "STAFF",
        "card_name": "Tornadus EX", "card_number": "BW96"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/porygon-z-bw84-prerelease-staff-pokemon-card.jpg",
        "filename": "porygonz_bw_staff.jpg",
        "variant": "staff", "set_id": "bw-promos", "stamp_text": "STAFF",
        "card_name": "Porygon-Z", "card_number": "BW84"
    },
    {
        "url": "https://pokemonflashfire.com/wp-content/uploads/2016/04/metagross-bw75-staff-prerelease-plasma-freeze.jpg",
        "filename": "metagross_plasma_freeze_staff.jpg",
        "variant": "staff", "set_id": "plasma-freeze", "stamp_text": "STAFF",
        "card_name": "Metagross", "card_number": "BW75"
    },
]

def download_card(card):
    filepath = os.path.join(OUTPUT_DIR, card["filename"])
    if os.path.exists(filepath):
        print(f"  SKIP (exists): {card['filename']}")
        return True

    try:
        req = urllib.request.Request(card["url"], headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = resp.read()
            if len(data) < 1000:
                print(f"  FAIL (too small {len(data)}b): {card['filename']}")
                return False
            with open(filepath, "wb") as f:
                f.write(data)
            print(f"  OK ({len(data)//1024}KB): {card['filename']}")
            return True
    except Exception as e:
        print(f"  FAIL ({e}): {card['filename']}")
        return False

def main():
    labels = {}
    success = 0
    fail = 0

    prerelease_cards = [c for c in CARDS if c["variant"] == "prerelease"]
    staff_cards = [c for c in CARDS if c["variant"] == "staff"]

    print(f"Downloading {len(prerelease_cards)} prerelease + {len(staff_cards)} staff = {len(CARDS)} total")
    print()

    for i, card in enumerate(CARDS):
        print(f"[{i+1}/{len(CARDS)}] {card['card_name']} ({card['variant']})")
        ok = download_card(card)
        if ok:
            success += 1
            labels[card["filename"]] = {
                "variant": card["variant"],
                "set_id": card["set_id"],
                "stamp_text": card["stamp_text"],
                "card_name": card["card_name"],
                "card_number": card["card_number"],
            }
        else:
            fail += 1
        time.sleep(2)  # rate limit

    # Save labels
    labels_path = os.path.join(OUTPUT_DIR, "labels.json")
    # Merge with existing labels if present
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            existing = json.load(f)
        existing.update(labels)
        labels = existing

    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nDone: {success} downloaded, {fail} failed")
    print(f"Labels saved to {labels_path}")

    # Summary
    variants = {}
    for v in labels.values():
        vt = v["variant"]
        variants[vt] = variants.get(vt, 0) + 1
    for vt, count in sorted(variants.items()):
        print(f"  {vt}: {count}")

if __name__ == "__main__":
    main()
