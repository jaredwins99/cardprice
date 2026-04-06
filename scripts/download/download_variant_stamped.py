#!/usr/bin/env python3
"""Download stamped/reverse holo variant images for EX-era sets.

Downloads real photographs of EX-era reverse holo cards into
data/card_images_variants/stamped/{set_id}/ directories.

Key finding: ex1-ex4 have rainbow holographic reverse holos (no stamped logo).
ex5+ have stamped logos (energy symbols, Poke Ball, set-specific logos).
All are "reverse holo variants" but only ex5+ are technically "stamped".
"""

import os
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path("/home/godli/cardprice/data/card_images_variants/stamped")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}

# Format: (url, filename, set_id, card_name, source_url)
DOWNLOADS = [
    # =========================================================================
    # EX1 - Ruby & Sapphire (rainbow holo, no stamp logo)
    # Source: Elite Fourum Gallery
    # =========================================================================
    ("https://efour.b-cdn.net/uploads/default/original/3X/7/b/7b6e7cfdb82ac066c8a05d4d949c4d78e68577ee.webp",
     "aggron_1_109_reverse_holo.webp", "ex1", "Aggron 1/109",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/5/e/5e8dd2543bb9ef07262299373a0f71d2ad1a7860.webp",
     "beautifly_2_109_reverse_holo.webp", "ex1", "Beautifly 2/109",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/4/0/404aa257b64a98564cc979ecab4185592a0d05a1.webp",
     "blaziken_3_109_reverse_holo.webp", "ex1", "Blaziken 3/109",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/7/5/75ae854785c181df4be2a523d0855cbf0382532e.webp",
     "camerupt_4_109_reverse_holo.webp", "ex1", "Camerupt 4/109",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/8/5/857c9b714bc09142ab915c23b8955c0a9c31cd4c.webp",
     "delcatty_5_109_reverse_holo.webp", "ex1", "Delcatty 5/109",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/1/5/15222a06b09d7ee585b7a825f9244c34e9881ccc.webp",
     "gardevoir_7_109_reverse_holo.webp", "ex1", "Gardevoir 7/109",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/5/5/55d057315b297267efa0445d5a15f0493676faac.webp",
     "sceptile_11_109_reverse_holo.webp", "ex1", "Sceptile 11/109",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/b/f/bfd4f36ddbab4c67a0e70fd0709693e3dde5e41c.webp",
     "slaking_12_109_reverse_holo.webp", "ex1", "Slaking 12/109",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/d/0/d0076dc1c8c8a90de2bfd7b1464d36609db3b3e7.webp",
     "swampert_13_109_reverse_holo.webp", "ex1", "Swampert 13/109",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/c/6/c6c9ae39f1c76fcc540434486295ca5db35c101a.webp",
     "sceptile_20_109_reverse_holo.webp", "ex1", "Sceptile 20/109",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),

    # =========================================================================
    # EX2 - Sandstorm (rainbow holo, no stamp logo)
    # Source: eBay seller photos
    # =========================================================================
    ("https://i.ebayimg.com/images/g/obsAAOSwTfVoCC~U/s-l1600.png",
     "omastar_19_100_reverse_holo.png", "ex2", "Omastar 19/100",
     "https://www.ebay.com/itm/156316607853"),
    ("https://i.ebayimg.com/images/g/~~4AAeSwZappZCDa/s-l1600.jpg",
     "psyduck_73_100_reverse_holo.jpg", "ex2", "Psyduck 73/100",
     "https://www.ebay.com/itm/364546554947"),
    ("https://i.ebayimg.com/images/g/HAUAAeSwMXhoEzj3/s-l1600.jpg",
     "flareon_5_100_reverse_holo.jpg", "ex2", "Flareon 5/100",
     "https://www.ebay.com/itm/326559160337"),
    ("https://i.ebayimg.com/images/g/V4cAAOSwKeBlsx4I/s-l1600.jpg",
     "omanyte_70_100_reverse_holo.jpg", "ex2", "Omanyte 70/100",
     "https://www.ebay.com/itm/296587526153"),
    ("https://i.redd.it/ioymdvaeo5q61.jpg",
     "eevee_63_100_reverse_holo.jpg", "ex2", "Eevee 63/100",
     "https://reddit.com/r/PokemonTCG/comments/mgefb6/"),
    ("https://i.ebayimg.com/images/g/pkoAAeSwT3xpvxXG/s-l1600.jpg",
     "umbreon_24_100_reverse_holo.jpg", "ex2", "Umbreon 24/100",
     "https://www.ebay.com/itm/257418512455"),
    ("https://i.ebayimg.com/images/g/q1MAAeSwUUNph8Qa/s-l1600.jpg",
     "shiftry_22_100_reverse_holo.jpg", "ex2", "Shiftry 22/100",
     "https://www.ebay.com/itm/297666146590"),
    ("https://i.ebayimg.com/images/g/~uoAAOSwXMtnMMBr/s-l1600.jpg",
     "collection_spread_reverse_holo.jpg", "ex2", "Collection Spread",
     "https://www.ebay.com/itm/365224144297"),

    # =========================================================================
    # EX3 - Dragon (rainbow holo, no stamp logo)
    # Source: Elite Fourum Gallery
    # =========================================================================
    ("https://efour.b-cdn.net/uploads/default/original/3X/5/5/5558fd4b1d111458171eed179defa696cbae01a1.webp",
     "absol_1_97_reverse_holo.webp", "ex3", "Absol 1/97",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/0/9/097820973269fe274ee24f83a694559f24db88ff.webp",
     "altaria_2_97_reverse_holo.webp", "ex3", "Altaria 2/97",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/4/c/4c3610567ac1a0074a9db3e83ad4e0b6ab6fcd76.webp",
     "crawdaunt_3_97_reverse_holo.webp", "ex3", "Crawdaunt 3/97",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/a/7/a7b9b1e27a64a5415621385f45c75dd10019cb46.webp",
     "flygon_4_97_reverse_holo.webp", "ex3", "Flygon 4/97",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/4/3/4317ca26a445f2a34401a3f6faf89c24315ddf70.webp",
     "golem_5_97_reverse_holo.webp", "ex3", "Golem 5/97",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/d/e/de9dbbd328cab62ac28710a7b3423cb859d79119.webp",
     "salamence_10_97_reverse_holo.webp", "ex3", "Salamence 10/97",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/7/7/77c72d16a3229bb8d7c5019bfaf855deb1c255ab.webp",
     "tv_reporter_88_97_reverse_holo.webp", "ex3", "TV Reporter 88/97",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),

    # =========================================================================
    # EX4 - Team Magma vs Team Aqua (rainbow holo, no stamp logo)
    # Source: Elite Fourum Gallery
    # =========================================================================
    ("https://efour.b-cdn.net/uploads/default/original/3X/8/c/8c80abaf5f452c5c1d05c48052fe6d94b7cf5b4c.webp",
     "team_aquas_cacturne_1_95_reverse_holo.webp", "ex4", "Team Aqua's Cacturne 1/95",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/a/6/a6709d4b1854b1238a00f7b3412074b60df1ce01.webp",
     "team_aquas_crawdaunt_2_95_reverse_holo.webp", "ex4", "Team Aqua's Crawdaunt 2/95",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/8/e/8e638bfcac860c6c696a1c3f94309d4fd875cc0f.webp",
     "team_aquas_kyogre_3_95_reverse_holo.webp", "ex4", "Team Aqua's Kyogre 3/95",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/b/b/bb9f7b5072ef829fe353e9f78f70b0f073d7a163.webp",
     "team_aquas_manectric_4_95_reverse_holo.webp", "ex4", "Team Aqua's Manectric 4/95",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/2/6/26f81ad00dd89147b34fe1d5edc8dc2212989644.webp",
     "team_aquas_sharpedo_5_95_reverse_holo.webp", "ex4", "Team Aqua's Sharpedo 5/95",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/2/a/2a87fef44a52a314b64a6d20bdb4861c4294103f.webp",
     "team_aquas_walrein_6_95_reverse_holo.webp", "ex4", "Team Aqua's Walrein 6/95",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/e/7/e7f8aed5b218a062b6a1b0673343121b4e9166d5.webp",
     "team_magmas_aggron_7_95_reverse_holo.webp", "ex4", "Team Magma's Aggron 7/95",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/7/c/7c3529d755ac6e42a7ed7dca2b367271622e919c.webp",
     "team_magmas_claydol_8_95_reverse_holo.webp", "ex4", "Team Magma's Claydol 8/95",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/d/5/d51a8ead5f93827b1c94d9b6832b785c6e90a6b9.webp",
     "team_magmas_groudon_9_95_reverse_holo.webp", "ex4", "Team Magma's Groudon 9/95",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/e/a/eaadd7675310b32ac51a8018b629fc0831890dd4.webp",
     "team_magmas_houndoom_10_95_reverse_holo.webp", "ex4", "Team Magma's Houndoom 10/95",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),

    # =========================================================================
    # EX5 - Hidden Legends (energy symbol pattern, transitional)
    # Source: Elite Fourum threads
    # =========================================================================
    ("https://efour.b-cdn.net/uploads/default/original/3X/1/7/173f18f62802ad0f10765db507a095cd9e992a3c.jpeg",
     "milotic_12_101_reverse_holo.jpeg", "ex5", "Milotic 12/101",
     "https://www.elitefourum.com/t/set-of-the-fortnight-60-ex-hidden-legends/56198"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/c/e/ce126951f390c3089505b6482582b875116a838b.jpeg",
     "heracross_reverse_holo.jpeg", "ex5", "Heracross",
     "https://www.elitefourum.com/t/set-of-the-fortnight-60-ex-hidden-legends/56198"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/1/4/146d1ba4fbd6ac3377d6c79c00fed104535cd599.jpeg",
     "dark_celebi_4_reverse_holo.jpeg", "ex5", "Dark Celebi 4/101",
     "https://www.elitefourum.com/t/set-of-the-fortnight-60-ex-hidden-legends/56198"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/9/5/955b56925a63b3d96dd5d40017882442236eced1.jpeg",
     "banette_reverse_holo.jpeg", "ex5", "Banette",
     "https://www.elitefourum.com/t/set-of-the-fortnight-60-ex-hidden-legends/56198"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/d/7/d78db9de48f80a870e83efa2d6cb2d38b645146f.jpeg",
     "jirachi_reverse_holo.jpeg", "ex5", "Jirachi",
     "https://www.elitefourum.com/t/set-of-the-fortnight-60-ex-hidden-legends/56198"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/c/2/c295f9f68e13c47795bcbe0c01ae5b642f0e3295.jpeg",
     "bellossom_reverse_holo.jpeg", "ex5", "Bellossom",
     "https://www.elitefourum.com/t/set-of-the-fortnight-60-ex-hidden-legends/56198"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/6/1/61cb9993676580a937af882d3f8472065caffebe.jpeg",
     "dodrio_33_101_reverse_holo.jpeg", "ex5", "Dodrio 33/101",
     "https://www.elitefourum.com/t/potential-hidden-legends-holo-print-error/43213"),
    # Gallery images
    ("https://efour.b-cdn.net/uploads/default/original/3X/8/b/8b3309816b03e0c3a31ba6d0c474e8eaa83803f4.webp",
     "gallery_card_01_reverse_holo.webp", "ex5", "Hidden Legends Gallery 1",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/9/7/97027feef13c36ea4584ee0d445bb529322562c2.webp",
     "gallery_card_02_reverse_holo.webp", "ex5", "Hidden Legends Gallery 2",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/4/8/483dbe31adc95ad5937c49ae71a78b10abba655c.webp",
     "gallery_card_03_reverse_holo.webp", "ex5", "Hidden Legends Gallery 3",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),

    # =========================================================================
    # EX6 - FireRed & LeafGreen (Poke Ball foil pattern)
    # Source: Elite Fourum Gallery
    # =========================================================================
    ("https://efour.b-cdn.net/uploads/default/original/3X/a/2/a2976e064bef200ef6759307234201ddc86b1bf1.webp",
     "beedrill_1_112_reverse_holo.webp", "ex6", "Beedrill 1/112",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/5/f/5ff157790546b52853b61c7d70bde8ad70e88658.webp",
     "ditto_4_112_reverse_holo.webp", "ex6", "Ditto 4/112",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/e/a/ea18d7bc1b8480e0492487c79ceeb540f86f034a.webp",
     "nidoking_8_112_reverse_holo.webp", "ex6", "Nidoking 8/112",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/3/1/31b3bbf87af3bf2ecc2c3b820705bc183f9e14c6.webp",
     "nidoqueen_9_112_reverse_holo.webp", "ex6", "Nidoqueen 9/112",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/b/4/b4714a59f79917079ca02bfa78265b11211af75d.webp",
     "pidgeot_10_112_reverse_holo.webp", "ex6", "Pidgeot 10/112",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/7/a/7a451bb1726bc14c71ae2d1d3434de413f8ff7c9.webp",
     "raichu_12_112_reverse_holo.webp", "ex6", "Raichu 12/112",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/e/3/e391f3b956394c27c1a9fbdf758e92d14777d506.webp",
     "arcanine_18_112_reverse_holo.webp", "ex6", "Arcanine 18/112",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/6/9/69f2d833f1778ab1bd19519ef590437a8ff8785d.webp",
     "butterfree_2_112_reverse_holo.webp", "ex6", "Butterfree 2/112",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/0/d/0d061f5cc41882043b5a8ebb56f73b3a7167cea6.webp",
     "dewgong_3_112_reverse_holo.webp", "ex6", "Dewgong 3/112",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),
    ("https://efour.b-cdn.net/uploads/default/original/3X/1/7/17f8d2ea6d70906c3f270f4aecfb1f9861dd6796.webp",
     "exeggutor_5_112_reverse_holo.webp", "ex6", "Exeggutor 5/112",
     "https://www.elitefourum.com/t/ex-era-reverse-holos-gallery/50653"),

    # =========================================================================
    # EX11 - Delta Species (NEW - adding to existing 8)
    # Source: eBay seller photos
    # =========================================================================
    ("https://i.ebayimg.com/images/g/VKIAAeSwuRZorPOf/s-l1600.jpg",
     "salamence_14_113_ebay.jpg", "ex11", "Salamence 14/113",
     "https://www.ebay.com/itm/157271743016"),
    ("https://i.ebayimg.com/images/g/5fAAAeSwzLJpMTWz/s-l1600.jpg",
     "espeon_4_113_ebay.jpg", "ex11", "Espeon 4/113",
     "https://www.ebay.com/itm/376752260431"),
    ("https://i.ebayimg.com/images/g/BR0AAeSweJxo~mdx/s-l1600.jpg",
     "ralts_57_113_ebay.jpg", "ex11", "Ralts 57/113",
     "https://www.ebay.com/itm/397197270350"),
    ("https://i.ebayimg.com/images/g/yxYAAOSwA9hoSSw2/s-l1600.png",
     "porygon_80_113_ebay.png", "ex11", "Porygon 80/113",
     "https://www.ebay.com/itm/205552084807"),
    ("https://i.ebayimg.com/images/g/UsAAAeSwO5Zpudso/s-l1600.jpg",
     "marill_76_113_ebay.jpg", "ex11", "Marill 76/113",
     "https://www.ebay.com/itm/298134699427"),
    ("https://i.ebayimg.com/images/g/uOoAAeSwlxdpAWV9/s-l1600.jpg",
     "ditto_pikachu_40_113_ebay.jpg", "ex11", "Ditto (Pikachu) 40/113",
     "https://www.ebay.com/itm/286903988331"),
    ("https://i.ebayimg.com/images/g/Rk4AAeSwqTxouxqN/s-l1600.jpg",
     "weedle_87_113_ebay.jpg", "ex11", "Weedle 87/113",
     "https://www.ebay.com/itm/297590674656"),
    # PSA slabs (stamp visible through case)
    ("https://i.ebayimg.com/images/g/XyoAAeSwTyVpuqhM/s-l1600.jpg",
     "crobat_2_113_psa.jpg", "ex11", "Crobat 2/113 (PSA)",
     "https://www.ebay.com/itm/187453680145"),
    ("https://i.ebayimg.com/images/g/NcsAAeSwfLxpuIqh/s-l1600.jpg",
     "metagross_11_113_psa.jpg", "ex11", "Metagross 11/113 (PSA)",
     "https://www.ebay.com/itm/358212855713"),
    ("https://i.ebayimg.com/images/g/wfgAAOSwaZNnko0j/s-l1600.jpg",
     "staryu_84_113_psa.jpg", "ex11", "Staryu 84/113 (PSA)",
     "https://www.ebay.com/itm/326878188002"),

    # =========================================================================
    # EX12 - Legend Maker (NEW - adding to existing 8)
    # Source: eBay seller photos via SportsCardInvestor
    # =========================================================================
    ("https://i.ebayimg.com/images/g/ep0AAeSw0GlpyYRY/s-l1600.jpg",
     "kabutops_7_92_ebay.jpg", "ex12", "Kabutops 7/92",
     "https://www.ebay.com/itm/225629442792"),
    ("https://i.ebayimg.com/images/g/-2YAAeSwPvlprx8X/s-l1600.jpg",
     "aerodactyl_1_92_ebay.jpg", "ex12", "Aerodactyl 1/92",
     "https://www.sportscardinvestor.com"),
    ("https://i.ebayimg.com/images/g/SuYAAeSwtDdpxD8A/s-l1600.jpg",
     "wailord_14_92_ebay.jpg", "ex12", "Wailord 14/92",
     "https://www.sportscardinvestor.com"),
    ("https://i.ebayimg.com/images/g/0PsAAeSwgIdpMOP8/s-l1600.jpg",
     "magmar_21_92_ebay.jpg", "ex12", "Magmar 21/92",
     "https://www.sportscardinvestor.com"),
    ("https://i.ebayimg.com/images/g/F64AAOSwvk5lRwLx/s-l1600.jpg",
     "lunatone_20_92_ebay.jpg", "ex12", "Lunatone 20/92",
     "https://www.sportscardinvestor.com"),
    ("https://i.ebayimg.com/images/g/ngkAAeSwTyVptK2p/s-l1600.jpg",
     "golem_6_92_ebay.jpg", "ex12", "Golem 6/92",
     "https://www.sportscardinvestor.com"),
    ("https://i.ebayimg.com/images/g/BWQAAeSwBy5ph4Wq/s-l1600.jpg",
     "lapras_8_92_ebay.jpg", "ex12", "Lapras 8/92",
     "https://www.sportscardinvestor.com"),
    ("https://i.ebayimg.com/images/g/dKIAAeSwL95pfmrL/s-l1600.jpg",
     "victreebel_13_92_ebay.jpg", "ex12", "Victreebel 13/92",
     "https://www.sportscardinvestor.com"),
    ("https://i.ebayimg.com/images/g/YggAAeSwZ35psZzA/s-l1600.jpg",
     "cradily_3_92_ebay.jpg", "ex12", "Cradily 3/92",
     "https://www.sportscardinvestor.com"),
    ("https://i.ebayimg.com/images/g/hY0AAeSwytJprdhZ/s-l1600.jpg",
     "pinsir_24_92_ebay.jpg", "ex12", "Pinsir 24/92",
     "https://www.sportscardinvestor.com"),
]


def download_one(url, filename, set_id, card_name, source_url):
    """Download a single image into set_id directory."""
    dest_dir = BASE_DIR / set_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename

    if dest.exists():
        print(f"  SKIP (exists): {set_id}/{filename}")
        return "skip"

    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
            if len(data) < 1000:
                print(f"  SKIP (too small {len(data)}B): {set_id}/{filename}")
                return "skip"
            dest.write_bytes(data)
            print(f"  OK ({len(data)//1024}KB): {set_id}/{filename} - {card_name}")
            return "ok"
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        print(f"  FAIL ({e}): {set_id}/{filename}")
        return "fail"


if __name__ == "__main__":
    counts = {"ok": 0, "skip": 0, "fail": 0}
    total = len(DOWNLOADS)
    print(f"Downloading {total} variant images...")

    for i, (url, filename, set_id, card_name, source_url) in enumerate(DOWNLOADS):
        result = download_one(url, filename, set_id, card_name, source_url)
        counts[result] += 1
        if i < total - 1:
            time.sleep(1.5)

    print(f"\nDone: {counts['ok']} downloaded, {counts['skip']} skipped, {counts['fail']} failed")

    # Summary per set
    print("\nPer-set summary:")
    for d in sorted(BASE_DIR.iterdir()):
        if d.is_dir():
            n = len(list(d.iterdir()))
            print(f"  {d.name}: {n} files")
