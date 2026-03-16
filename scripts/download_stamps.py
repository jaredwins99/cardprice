#!/usr/bin/env python3
"""Download more EX-era stamped reverse holo Pokemon card images."""

import os
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

OUT_DIR = Path("/home/godli/cardprice/data/condition_training/stamps_real")
SOURCES_FILE = OUT_DIR / "sources.jsonl"

# Load existing URLs to skip
existing_urls = set()
if SOURCES_FILE.exists():
    for line in SOURCES_FILE.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            existing_urls.add(rec.get("original_url", ""))

# Also track existing filenames
existing_files = set(os.listdir(OUT_DIR))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}

# New images to download
# Format: (url, filename, set_id, stamped, card_name, source_url)
downloads = [
    # === EFOUR GALLERY - New stamped reverse holos (Ruby & Sapphire set) ===
    ("https://efour.b-cdn.net/uploads/default/original/3X/7/b/7b6e7cfdb82ac066c8a05d4d949c4d78e68577ee.webp",
     "rs_01r_stamped.webp", "ex1", True, "Ruby Sapphire Reverse Holo #1",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/5/e/5e8dd2543bb9ef07262299373a0f71d2ad1a7860.webp",
     "rs_02r_stamped.webp", "ex1", True, "Ruby Sapphire Reverse Holo #2",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/4/0/404aa257b64a98564cc979ecab4185592a0d05a1.webp",
     "rs_03r_stamped.webp", "ex1", True, "Ruby Sapphire Reverse Holo #3",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/7/5/75ae854785c181df4be2a523d0855cbf0382532e.webp",
     "rs_04r_stamped.webp", "ex1", True, "Ruby Sapphire Reverse Holo #4",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/8/5/857c9b714bc09142ab915c23b8955c0a9c31cd4c.webp",
     "rs_05r_stamped.webp", "ex1", True, "Ruby Sapphire Reverse Holo #5",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/4/f/4fba847cee3970bdd9fdecd94f2c787e326957d4.webp",
     "sandstorm_01r_stamped.webp", "ex2", True, "Sandstorm Reverse Holo #1",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/1/5/15222a06b09d7ee585b7a825f9244c34e9881ccc.webp",
     "sandstorm_02r_stamped.webp", "ex2", True, "Sandstorm Reverse Holo #2",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/2/3/23262f83e90532a96d153923e7a25cf1eaf38206.webp",
     "dragon_01r_stamped.webp", "ex3", True, "Dragon Reverse Holo #1",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/4/c/4c44f001c99ea32ffca1c4974e1a3504b1dedb3d.webp",
     "dragon_02r_stamped.webp", "ex3", True, "Dragon Reverse Holo #2",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/c/3/c361f28af4f927ffcaad60e9682b11d8abc3c62d.webp",
     "tma_01r_stamped.webp", "ex4", True, "Team Magma vs Aqua Reverse Holo #1",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),

    # === EFOUR "best release ever" thread - stamped card photos ===
    ("https://i.imgur.com/7oCW8Wp.jpg",
     "ex_imgur_stamped_01.jpg", "ex_misc", True, "EX Stamped Reverse Holo (imgur 1)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://i.imgur.com/DtbjBzs.jpg",
     "ex_imgur_stamped_02.jpg", "ex_misc", True, "EX Stamped Reverse Holo (imgur 2)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://i.imgur.com/NMBKF8L.jpg",
     "ex_imgur_stamped_03.jpg", "ex_misc", True, "EX Stamped Reverse Holo (imgur 3)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://i.imgur.com/j2l1nJf.jpg",
     "ex_imgur_stamped_04.jpg", "ex_misc", True, "EX Stamped Reverse Holo (imgur 4)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://i.imgur.com/60JSX8G.jpg",
     "ex_imgur_stamped_05.jpg", "ex_misc", True, "EX Stamped Reverse Holo (imgur 5)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://i.imgur.com/GR5xZTw.jpg",
     "ex_imgur_stamped_06.jpg", "ex_misc", True, "EX Stamped Reverse Holo (imgur 6)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://i.imgur.com/bufXGyU.jpg",
     "ex_imgur_stamped_07.jpg", "ex_misc", True, "EX Stamped Reverse Holo (imgur 7)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),

    # === EFOUR "best release ever" - elitefourum hosted images ===
    ("https://www.elitefourum.com/uploads/default/original/3X/b/a/bae0bf4b6fa1bbb2f996f0f8bfe9041c4298396d.jpeg",
     "ex_efour_stamped_01.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 1)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/5/e/5e3a32d90035330021b30999a81a295dd58a27ea.jpeg",
     "ex_efour_stamped_02.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 2)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/8/8/888783f30939689b96948d47ae5bbedaa5bf11b0.jpeg",
     "ex_efour_stamped_03.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 3)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/4/c/4c940fda97f3d207419e4b1c5dabe9ed232255fb.jpeg",
     "ex_efour_stamped_04.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 4)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/a/b/ab25746407abb444a2d16610dd1ab603852f362b.jpeg",
     "ex_efour_stamped_05.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 5)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/a/b/ab08e830bc61f5dd35e7b33accc1bcb54d2fb7ad.jpeg",
     "ex_efour_stamped_06.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 6)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/4/6/463837ded9d31c4b4f30a212c812ba8852fc7283.jpeg",
     "ex_efour_stamped_07.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 7)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/2/e/2ecb3c9b0e38cc9d9aa66fbda7b823e93e185711.jpeg",
     "ex_efour_stamped_08.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 8)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/7/6/76b222dd4d1cbb8c8adfe1307dbf5209ec9de108.jpeg",
     "ex_efour_stamped_09.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 9)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/3/1/310045f901da25e36d5a3e88cb9ac6ea2d85d104.jpeg",
     "ex_efour_stamped_10.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 10)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/f/1/f1011b26b5dfdab67fdf2671783191e5208dd44a.jpeg",
     "ex_efour_stamped_11.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 11)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/3/e/3e2d1435953c657fe9e4cc7b7502a66945e7ef95.jpeg",
     "ex_efour_stamped_12.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 12)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/d/3/d3d4c2f7349bfa3f2e451b2d20e068ed9e4c7307.jpeg",
     "ex_efour_stamped_13.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 13)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/7/4/743a9d730ea91d623d76bf0b97d3186cd75fbde4.jpeg",
     "ex_efour_stamped_14.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 14)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/4/2/427642802ee9eec25ed115fb6abb8700226d1615.jpeg",
     "ex_efour_stamped_15.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 15)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/6/e/6e3a3cdadc6ba01ec9fb8d17ea0676a9a2231efc.jpeg",
     "ex_efour_stamped_16.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 16)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/c/4/c4b2514ba0197b625080cafff5115d19fdb913c9.jpeg",
     "ex_efour_stamped_17.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 17)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/a/0/a04c344f70ccc13fc400f1af7d4e3d0ed01ad9a0.jpeg",
     "ex_efour_stamped_18.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 18)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),
    ("https://www.elitefourum.com/uploads/default/original/3X/1/0/109ca54c7926214922e73487a5e148d3feef8e94.jpeg",
     "ex_efour_stamped_19.jpeg", "ex_misc", True, "EX Stamped Reverse Holo (efour 19)",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-are-the-best-english-release-ever/31835"),

    # === EFOUR favorites poll - multi-card comparison images (stamped) ===
    ("https://efour.b-cdn.net/uploads/default/original/3X/d/9/d916ffcf60f1c4bc4b13385915821bad4fe63078.jpeg",
     "ex_poll_stamped_01.jpeg", "ex_misc", True, "EX Favorites Poll Set #1",
     "https://www.elitefourum.com/t/e4s-favorite-ex-era-reverse-holo-patterns-results/58382"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/d/3/d312d64d050f718e7260d57f3a7ee239cfc5c596.jpeg",
     "ex_poll_stamped_02.jpeg", "ex_misc", True, "EX Favorites Poll Set #2",
     "https://www.elitefourum.com/t/e4s-favorite-ex-era-reverse-holo-patterns-results/58382"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/0/4/04114db71ef4232b026141ce18a2eb78b43888f1.jpeg",
     "ex_poll_stamped_03.jpeg", "ex_misc", True, "EX Favorites Poll Set #3",
     "https://www.elitefourum.com/t/e4s-favorite-ex-era-reverse-holo-patterns-results/58382"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/c/b/cb70bed334faf9e9d81b1c0eaed6125d2411a21d.jpeg",
     "ex_poll_stamped_04.jpeg", "ex_misc", True, "EX Favorites Poll Set #4",
     "https://www.elitefourum.com/t/e4s-favorite-ex-era-reverse-holo-patterns-results/58382"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/9/7/97af7249558098161c1a28aa08566165e88bb22e.jpeg",
     "ex_poll_stamped_05.jpeg", "ex_misc", True, "EX Favorites Poll Set #5",
     "https://www.elitefourum.com/t/e4s-favorite-ex-era-reverse-holo-patterns-results/58382"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/8/7/87aaa75b88dbce37df9460dec5d15be726bd56d5.jpeg",
     "ex_poll_stamped_06.jpeg", "ex_misc", True, "EX Favorites Poll Set #6",
     "https://www.elitefourum.com/t/e4s-favorite-ex-era-reverse-holo-patterns-results/58382"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/3/c/3c63cf293da6c68e570ed68f2e180e4b429cb8f6.jpeg",
     "ex_poll_stamped_07.jpeg", "ex_misc", True, "EX Favorites Poll Set #7",
     "https://www.elitefourum.com/t/e4s-favorite-ex-era-reverse-holo-patterns-results/58382"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/8/c/8ccb24f4600c5ddd984ed436ec427ff71d53482d.jpeg",
     "ex_poll_stamped_08.jpeg", "ex_misc", True, "EX Favorites Poll Set #8",
     "https://www.elitefourum.com/t/e4s-favorite-ex-era-reverse-holo-patterns-results/58382"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/3/7/376e579683aae2935d6ccd846049a0f3563515a9.jpeg",
     "ex_poll_stamped_09.jpeg", "ex_misc", True, "EX Favorites Poll Set #9",
     "https://www.elitefourum.com/t/e4s-favorite-ex-era-reverse-holo-patterns-results/58382"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/1/2/126e376d0b4bfb280719548c3472cf5f60db12a1.jpeg",
     "ex_poll_stamped_10.jpeg", "ex_misc", True, "EX Favorites Poll Set #10",
     "https://www.elitefourum.com/t/e4s-favorite-ex-era-reverse-holo-patterns-results/58382"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/1/7/175cc7bb8848ee8fc8eae83188301d2d9db36b02.jpeg",
     "ex_poll_stamped_11.jpeg", "ex_misc", True, "EX Favorites Poll Set #11",
     "https://www.elitefourum.com/t/e4s-favorite-ex-era-reverse-holo-patterns-results/58382"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/e/a/eaed5e295f1b0cd8dc313a3119a15223ea0ab87f.jpeg",
     "ex_poll_stamped_12.jpeg", "ex_misc", True, "EX Favorites Poll Set #12",
     "https://www.elitefourum.com/t/e4s-favorite-ex-era-reverse-holo-patterns-results/58382"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/9/4/94f8c45ac2efd22474451fde0bc4ad0f1a3c1a3c.jpeg",
     "ex_poll_stamped_13.jpeg", "ex_misc", True, "EX Favorites Poll Set #13",
     "https://www.elitefourum.com/t/e4s-favorite-ex-era-reverse-holo-patterns-results/58382"),

    # === eBay EX Deoxys set listing - multiple stamped cards ===
    ("https://i.ebayimg.com/images/g/F5cAAOSwOvJe2SDG/s-l1600.jpg",
     "ebay_ex_deoxys_set_01.jpg", "ex8", True, "EX Deoxys Set Stamped (eBay 1)",
     "https://www.ebay.com/itm/233609472430"),
    ("https://i.ebayimg.com/images/g/F18AAOSwOvJe2SCR/s-l1600.jpg",
     "ebay_ex_deoxys_set_02.jpg", "ex8", True, "EX Deoxys Set Stamped (eBay 2)",
     "https://www.ebay.com/itm/233609472430"),
    ("https://i.ebayimg.com/images/g/Vf8AAOSwhEte2SEm/s-l1600.jpg",
     "ebay_ex_deoxys_set_03.jpg", "ex8", True, "EX Deoxys Set Stamped (eBay 3)",
     "https://www.ebay.com/itm/233609472430"),
    ("https://i.ebayimg.com/images/g/5Y8AAOSwFYle2SDf/s-l1600.jpg",
     "ebay_ex_deoxys_set_04.jpg", "ex8", True, "EX Deoxys Set Stamped (eBay 4)",
     "https://www.ebay.com/itm/233609472430"),

    # === Coded Yellow - reverse holo pattern reference images ===
    ("https://www.codedyellow.com/wp-content/uploads/2023/04/EX-Deoxys.jpg",
     "codedyellow_ex_deoxys_stamped.jpg", "ex8", True, "EX Deoxys Pattern (Coded Yellow)",
     "https://www.codedyellow.com/reverse-holo-patterns/"),
    ("https://www.codedyellow.com/wp-content/uploads/2023/04/FireRed-LeafGreen.jpg",
     "codedyellow_ex_frlg_stamped.jpg", "ex6", True, "EX FRLG Pattern (Coded Yellow)",
     "https://www.codedyellow.com/reverse-holo-patterns/"),
    ("https://www.codedyellow.com/wp-content/uploads/2023/04/Hidden-Legends.jpg",
     "codedyellow_ex_hl_stamped.jpg", "ex5", True, "EX Hidden Legends Pattern (Coded Yellow)",
     "https://www.codedyellow.com/reverse-holo-patterns/"),
    ("https://www.codedyellow.com/wp-content/uploads/2023/04/Legend-Maker.jpg",
     "codedyellow_ex_lm_stamped.jpg", "ex12", True, "EX Legend Maker Pattern (Coded Yellow)",
     "https://www.codedyellow.com/reverse-holo-patterns/"),
    ("https://www.codedyellow.com/wp-content/uploads/2023/04/EX-Series-Rainbow.jpg",
     "codedyellow_ex_rainbow_stamped.jpg", "ex1", True, "EX Series Rainbow Pattern (Coded Yellow)",
     "https://www.codedyellow.com/reverse-holo-patterns/"),

    # === Additional regular (non-stamped) cards from pokemontcg.io for balance ===
    # Hidden Legends
    ("https://images.pokemontcg.io/ex5/1_hires.png",
     "hl_01_regular.png", "ex5", False, "Crobat",
     "https://images.pokemontcg.io/"),
    ("https://images.pokemontcg.io/ex5/2_hires.png",
     "hl_02_regular.png", "ex5", False, "Dark Celebi",
     "https://images.pokemontcg.io/"),
    ("https://images.pokemontcg.io/ex5/3_hires.png",
     "hl_03_regular.png", "ex5", False, "Electrode",
     "https://images.pokemontcg.io/"),
    # FRLG
    ("https://images.pokemontcg.io/ex6/1_hires.png",
     "frlg_01_regular.png", "ex6", False, "Aerodactyl",
     "https://images.pokemontcg.io/"),
    ("https://images.pokemontcg.io/ex6/2_hires.png",
     "frlg_02_regular.png", "ex6", False, "Articuno ex",
     "https://images.pokemontcg.io/"),
    ("https://images.pokemontcg.io/ex6/3_hires.png",
     "frlg_03_regular.png", "ex6", False, "Blastoise ex",
     "https://images.pokemontcg.io/"),
    # Delta Species
    ("https://images.pokemontcg.io/ex11/1_hires.png",
     "ds_01_regular.png", "ex11", False, "Beedrill d",
     "https://images.pokemontcg.io/"),
    ("https://images.pokemontcg.io/ex11/2_hires.png",
     "ds_02_regular.png", "ex11", False, "Crobat d",
     "https://images.pokemontcg.io/"),
    ("https://images.pokemontcg.io/ex11/3_hires.png",
     "ds_03_regular.png", "ex11", False, "Dragonite d",
     "https://images.pokemontcg.io/"),
    # Legend Maker
    ("https://images.pokemontcg.io/ex12/1_hires.png",
     "lm_01_regular.png", "ex12", False, "Absol",
     "https://images.pokemontcg.io/"),
    ("https://images.pokemontcg.io/ex12/2_hires.png",
     "lm_02_regular.png", "ex12", False, "Aggron",
     "https://images.pokemontcg.io/"),
    ("https://images.pokemontcg.io/ex12/3_hires.png",
     "lm_03_regular.png", "ex12", False, "Arcanine",
     "https://images.pokemontcg.io/"),
    # Holon Phantoms
    ("https://images.pokemontcg.io/ex13/1_hires.png",
     "hp_01_regular.png", "ex13", False, "Armaldo d",
     "https://images.pokemontcg.io/"),
    ("https://images.pokemontcg.io/ex13/2_hires.png",
     "hp_02_regular.png", "ex13", False, "Blaziken",
     "https://images.pokemontcg.io/"),
    # Crystal Guardians
    ("https://images.pokemontcg.io/ex14/1_hires.png",
     "cg_01_regular.png", "ex14", False, "Banette",
     "https://images.pokemontcg.io/"),
    ("https://images.pokemontcg.io/ex14/2_hires.png",
     "cg_02_regular.png", "ex14", False, "Blastoise",
     "https://images.pokemontcg.io/"),
    # Dragon Frontiers
    ("https://images.pokemontcg.io/ex15/1_hires.png",
     "df_01_regular.png", "ex15", False, "Altaria ex d",
     "https://images.pokemontcg.io/"),
    ("https://images.pokemontcg.io/ex15/2_hires.png",
     "df_02_regular.png", "ex15", False, "Feraligatr d",
     "https://images.pokemontcg.io/"),
]

def download_one(url, filename, set_id, stamped, card_name, source_url):
    """Download a single image, return True if successful."""
    if url in existing_urls:
        print(f"  SKIP (already have URL): {filename}")
        return False
    if filename in existing_files:
        print(f"  SKIP (filename exists): {filename}")
        return False

    dest = OUT_DIR / filename
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            if len(data) < 1000:
                print(f"  SKIP (too small {len(data)}B): {filename}")
                return False
            dest.write_bytes(data)
            # Append to sources.jsonl
            record = {
                "image": filename,
                "source_url": source_url,
                "set_id": set_id,
                "stamped": stamped,
                "card_name": card_name,
                "original_url": url,
            }
            with open(SOURCES_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"  OK ({len(data)//1024}KB): {filename}")
            return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        print(f"  FAIL ({e}): {filename}")
        return False


if __name__ == "__main__":
    success = 0
    fail = 0
    skip = 0
    total = len(downloads)
    print(f"Processing {total} downloads...")
    for i, (url, filename, set_id, stamped, card_name, source_url) in enumerate(downloads):
        result = download_one(url, filename, set_id, stamped, card_name, source_url)
        if result is True:
            success += 1
        elif result is False:
            skip += 1
        else:
            fail += 1
        # Rate limit
        if i < total - 1:
            time.sleep(2)
    print(f"\nDone: {success} downloaded, {skip} skipped, {fail} failed")
