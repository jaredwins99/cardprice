#!/usr/bin/env python3
"""Download EX-era stamped/reverse holo card images from Elite Fourum gallery.

Source: https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653
Images are real photographs of reverse holo stamped cards hosted on BunnyCDN.

Priority sets (completely missing from card_images_variants/stamped/):
  ex1 - EX Ruby & Sapphire
  ex2 - EX Sandstorm
  ex3 - EX Dragon
  ex4 - EX Team Magma vs Team Aqua
  ex5 - EX Hidden Legends
  ex6 - EX FireRed & LeafGreen

Secondary sets (need more images, currently have ~10 each):
  ex7 - EX Team Rocket Returns
  ex8 - EX Deoxys
  ex9 - EX Emerald
"""

import os
import time
import urllib.request
import urllib.error
from pathlib import Path

OUT_BASE = Path("/home/godli/cardprice/data/card_images_variants/stamped")
SOURCE_URL = "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"
CDN_BASE = "https://efour.b-cdn.net/uploads/default/original/3X"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}

# Card data: (set_id, card_num, card_name, hash_path)
# hash_path is the /X/Y/hash.ext portion after the CDN base
DOWNLOADS = [
    # ===== EX Ruby & Sapphire (ex1) - Mirror Holos =====
    ("ex1", 1, "Aggron", "7/b/7b6e7cfdb82ac066c8a05d4d949c4d78e68577ee.webp"),
    ("ex1", 2, "Beautifly", "5/e/5e8dd2543bb9ef07262299373a0f71d2ad1a7860.webp"),
    ("ex1", 3, "Blaziken", "4/0/404aa257b64a98564cc979ecab4185592a0d05a1.webp"),
    ("ex1", 4, "Camerupt", "7/5/75ae854785c181df4be2a523d0855cbf0382532e.webp"),
    ("ex1", 5, "Delcatty", "8/5/857c9b714bc09142ab915c23b8955c0a9c31cd4c.webp"),
    ("ex1", 6, "Dustox", "4/f/4fba847cee3970bdd9fdecd94f2c787e326957d4.webp"),
    ("ex1", 7, "Gardevoir", "1/5/15222a06b09d7ee585b7a825f9244c34e9881ccc.webp"),
    ("ex1", 8, "Hariyama", "2/3/23262f83e90532a96d153923e7a25cf1eaf38206.webp"),
    ("ex1", 9, "Manectric", "4/c/4c44f001c99ea32ffca1c4974e1a3504b1dedb3d.webp"),
    ("ex1", 10, "Mightyena", "c/3/c361f28af4f927ffcaad60e9682b11d8abc3c62d.webp"),

    # ===== EX Sandstorm (ex2) - Mirror Holos =====
    ("ex2", 1, "Armaldo", "4/b/4b2838ad75581d0bf4a5fde0bd99c11339144755.webp"),
    ("ex2", 2, "Cacturne", "7/f/7f7cb689d357df2254772ff81eb8e3c3cb9d1193.webp"),
    ("ex2", 3, "Cradily", "4/b/4b520219b5778c96a65256d7af2357baf98f4074.webp"),
    ("ex2", 4, "Dusclops", "c/4/c4b8467807ebaaed1fb6854f4df95469a67ba4aa.webp"),
    ("ex2", 5, "Flareon", "5/8/583c8ab71ed2e05f1b1e2a77a44f05825735e68e.webp"),
    ("ex2", 6, "Jolteon", "a/9/a99de7a2eab99641db069fe7a8ccc20ce62f858d.webp"),
    ("ex2", 7, "Ludicolo", "0/2/028589337811835588ba9840e743b34bd5977d30.webp"),
    ("ex2", 8, "Lunatone", "2/2/22ae04932ddcd03b7c7bb82f65c1405e6357b214.webp"),
    ("ex2", 9, "Mawile", "c/2/c20195e70214400de2d91f032cbe0fe5aae1854f.webp"),
    ("ex2", 10, "Sableye", "8/8/8893cb1a984270d290f75a8375b326947595ac38.webp"),

    # ===== EX Dragon (ex3) - Mirror Holos =====
    ("ex3", 1, "Absol", "5/5/5558fd4b1d111458171eed179defa696cbae01a1.webp"),
    ("ex3", 2, "Altaria", "0/9/097820973269fe274ee24f83a694559f24db88ff.webp"),
    ("ex3", 3, "Crawdaunt", "4/c/4c3610567ac1a0074a9db3e83ad4e0b6ab6fcd76.webp"),
    ("ex3", 4, "Flygon", "a/7/a7b9b1e27a64a5415621385f45c75dd10019cb46.webp"),
    ("ex3", 5, "Golem", "4/3/4317ca26a445f2a34401a3f6faf89c24315ddf70.webp"),
    ("ex3", 6, "Grumpig", "c/4/c4ee9ba2fd7595bf6bb86db50574726f6a4a3de8.webp"),
    ("ex3", 7, "Minun", "c/6/c6dfc176c3567543b9e6365e2cae99ccacf0fb78.webp"),
    ("ex3", 8, "Plusle", "a/3/a309a64aaebcdd4d0afc4515c66924a6872dea61.webp"),
    ("ex3", 9, "Roselia", "8/c/8c14b8f9baa1b004987c14d0ec7cd9ad155fb94a.webp"),
    ("ex3", 10, "Salamence", "d/e/de9dbbd328cab62ac28710a7b3423cb859d79119.webp"),

    # ===== EX Team Magma vs Team Aqua (ex4) - Mirror Holos =====
    ("ex4", 1, "Team Aqua's Cacturne", "8/c/8c80abaf5f452c5c1d05c48052fe6d94b7cf5b4c.webp"),
    ("ex4", 2, "Team Aqua's Crawdaunt", "a/6/a6709d4b1854b1238a00f7b3412074b60df1ce01.webp"),
    ("ex4", 3, "Team Aqua's Kyogre", "8/e/8e638bfcac860c6c696a1c3f94309d4fd875cc0f.webp"),
    ("ex4", 4, "Team Aqua's Manectric", "b/b/bb9f7b5072ef829fe353e9f78f70b0f073d7a163.webp"),
    ("ex4", 5, "Team Aqua's Sharpedo", "2/6/26f81ad00dd89147b34fe1d5edc8dc2212989644.webp"),
    ("ex4", 6, "Team Aqua's Walrein", "2/a/2a87fef44a52a314b64a6d20bdb4861c4294103f.webp"),
    ("ex4", 7, "Team Magma's Aggron", "e/7/e7f8aed5b218a062b6a1b0673343121b4e9166d5.webp"),
    ("ex4", 8, "Team Magma's Claydol", "7/c/7c3529d755ac6e42a7ed7dca2b367271622e919c.webp"),
    ("ex4", 9, "Team Magma's Groudon", "d/5/d51a8ead5f93827b1c94d9b6832b785c6e90a6b9.webp"),
    ("ex4", 10, "Team Magma's Houndoom", "e/a/eaadd7675310b32ac51a8018b629fc0831890dd4.webp"),

    # ===== EX Hidden Legends (ex5) - Energy Symbol Pattern =====
    ("ex5", 1, "Banette", "8/b/8b3309816b03e0c3a31ba6d0c474e8eaa83803f4.webp"),
    ("ex5", 2, "Claydol", "9/7/97027feef13c36ea4584ee0d445bb529322562c2.webp"),
    ("ex5", 3, "Crobat", "4/8/483dbe31adc95ad5937c49ae71a78b10abba655c.webp"),
    ("ex5", 4, "Dark Celebi", "3/8/388a1946424a0882dd39ec7b6dd868d48ed0e24e.webp"),
    ("ex5", 5, "Electrode", "8/1/81aef6641ad5dddb73bc8057a8b838eb2703535f.webp"),
    ("ex5", 6, "Exploud", "a/d/ade6b9ea7e9b07cc2b14600ceee55b031ad2ef91.webp"),
    ("ex5", 7, "Heracross", "d/4/d452447f5fe8e74b46c9c9db5ed456f2c33d6fc6.webp"),
    ("ex5", 8, "Jirachi", "2/0/208f7398748bd83e09ec62b8e03965dccc1217fb.webp"),
    ("ex5", 9, "Machamp", "9/4/9493de7c3bdef05d7ce2cd0cc027ab98e5925ebd.webp"),
    ("ex5", 10, "Medicham", "d/c/dc511fec19661a28099034aa407294b740e795f8.webp"),

    # ===== EX FireRed & LeafGreen (ex6) - Pokeball/Energy Pattern =====
    ("ex6", 1, "Beedrill", "a/2/a2976e064bef200ef6759307234201ddc86b1bf1.webp"),
    ("ex6", 2, "Butterfree", "6/9/69f2d833f1778ab1bd19519ef590437a8ff8785d.webp"),
    ("ex6", 3, "Dewgong", "0/d/0d061f5cc41882043b5a8ebb56f73b3a7167cea6.webp"),
    ("ex6", 4, "Ditto", "5/f/5ff157790546b52853b61c7d70bde8ad70e88658.webp"),
    ("ex6", 5, "Exeggutor", "1/7/17f8d2ea6d70906c3f270f4aecfb1f9861dd6796.webp"),
    ("ex6", 6, "Kangaskhan", "5/1/512e2eac89e179b20dd4ab7ffac6388c5550ab4a.webp"),
    ("ex6", 7, "Marowak", "7/e/7ea60a8154b6a0abfd30463d8cf523748f1c258f.webp"),
    ("ex6", 8, "Nidoking", "e/a/ea18d7bc1b8480e0492487c79ceeb540f86f034a.webp"),
    ("ex6", 9, "Nidoqueen", "3/1/31b3bbf87af3bf2ecc2c3b820705bc183f9e14c6.webp"),
    ("ex6", 10, "Pidgeot", "b/4/b4714a59f79917079ca02bfa78265b11211af75d.webp"),

    # ===== EX Team Rocket Returns (ex7) - Additional stamped cards =====
    # Existing: card_01 through card_10. Add card_11 through card_20.
    ("ex7", 11, "Card 11", "e/8/e885ee58127d3c8626030c7f71ee21545c978279.png"),
    ("ex7", 12, "Card 12", "7/5/7523efab1b2f02b5502551f8ab7af23ff6587523.png"),
    ("ex7", 13, "Card 13", "3/1/31c54c0ecf5b52e2ca48b9eb60e3b9eb5efc93bb.png"),
    ("ex7", 14, "Card 14", "9/4/94e984d2a1adb6103ea58f1d848c2b59f47f4023.png"),
    ("ex7", 15, "Card 15", "f/d/fd5b16399d96649b899444229e272406d55af960.png"),
    ("ex7", 16, "Card 16", "8/f/8f4edf9f502f67d43d51775848153fa6da1d64fe.png"),
    ("ex7", 17, "Card 17", "a/4/a4dd8f0269e65b1812af0cb0954c00477a8c63ba.png"),
    ("ex7", 18, "Card 18", "f/9/f9eed93fb5c2d736bb6f59f0442d4ed66169f22f.png"),
    ("ex7", 19, "Card 19", "7/a/7a486d37d3e54e13029bc676ba7fd3e40767ab97.png"),
    ("ex7", 20, "Card 20", "c/8/c8b28b86deed9d173344c6ba9c42dcf909b3c0bf.png"),

    # ===== EX Deoxys (ex8) - Additional stamped cards =====
    # Existing: card_01 through card_10. Add card_11 through card_15.
    ("ex8", 11, "Card 11", "d/2/d27c8e5dbbf3fbd3ad8709fdd680da04c8cb6be5.jpeg"),
    ("ex8", 12, "Card 12", "4/b/4babbbd6f889ac4fae6672800a294689c44fd10d.jpeg"),
    ("ex8", 13, "Card 13", "e/b/eb693b7a0243fdc748a8e51e8b45bbffed53e9bc.jpeg"),
    ("ex8", 14, "Card 14", "3/9/3985d1440f7c69fd049ba4b188e9fd3e1e9268f9.jpeg"),
    ("ex8", 15, "Card 15", "2/0/20830e608f623f96eaf555bf517757ec29d05bb9.jpeg"),

    # ===== EX Emerald (ex9) - Additional stamped cards =====
    # Existing: card_01 through card_10. Add card_11 through card_15.
    ("ex9", 11, "Card 11", "5/c/5c26ba06bac3d78a8f8aa4c7dc4fb610c8068801.webp"),
    ("ex9", 12, "Card 12", "8/f/8f49283431175cccbba75587f8fa1bfabc5f9966.webp"),
    ("ex9", 13, "Card 13", "c/7/c7a913701d3108269720621508f111232db5d3d9.webp"),
    ("ex9", 14, "Card 14", "2/0/20a8a4794e9e4d84410ef4d3d636e6b946e21180.webp"),
    ("ex9", 15, "Card 15", "8/3/83bacd135572f9146a9508a3f275bdc774397a57.webp"),
]


def download_one(set_id, card_num, card_name, hash_path):
    """Download a single image. Returns (success, skip, fail)."""
    out_dir = OUT_BASE / set_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = hash_path.rsplit(".", 1)[-1]
    filename = f"card_{card_num:02d}.{ext}"
    dest = out_dir / filename

    if dest.exists():
        print(f"  SKIP (exists): {set_id}/{filename} ({card_name})")
        return "skip"

    url = f"{CDN_BASE}/{hash_path}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            if len(data) < 1000:
                print(f"  FAIL (too small {len(data)}B): {set_id}/{filename}")
                return "fail"
            dest.write_bytes(data)
            print(f"  OK ({len(data)//1024}KB): {set_id}/{filename} - {card_name}")
            return "ok"
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        print(f"  FAIL ({e}): {set_id}/{filename}")
        return "fail"


def main():
    ok = skip = fail = 0
    total = len(DOWNLOADS)
    print(f"Downloading {total} stamped/reverse holo card images...")
    print(f"Source: {SOURCE_URL}")
    print(f"Output: {OUT_BASE}")
    print()

    current_set = None
    for i, (set_id, card_num, card_name, hash_path) in enumerate(DOWNLOADS):
        if set_id != current_set:
            current_set = set_id
            print(f"\n=== {set_id} ===")

        result = download_one(set_id, card_num, card_name, hash_path)
        if result == "ok":
            ok += 1
        elif result == "skip":
            skip += 1
        else:
            fail += 1

        # Rate limit (be polite to CDN)
        if i < total - 1:
            time.sleep(0.5)

    print(f"\n{'='*40}")
    print(f"Done: {ok} downloaded, {skip} skipped, {fail} failed")
    print(f"Total images expected: {total}")

    # Print summary per set
    from collections import Counter
    set_counts = Counter()
    for set_id, _, _, _ in DOWNLOADS:
        set_counts[set_id] += 1
    print("\nPer-set breakdown:")
    for set_id in sorted(set_counts.keys()):
        existing = len(list((OUT_BASE / set_id).glob("*"))) if (OUT_BASE / set_id).exists() else 0
        print(f"  {set_id}: {set_counts[set_id]} new + {existing - set_counts[set_id] if existing > set_counts[set_id] else 0} existing = {max(existing, set_counts[set_id])} total")


if __name__ == "__main__":
    main()
