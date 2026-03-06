"""
Map Pokemon TCG set IDs to eras/generations.

Eras:
  1 = WotC Classic (1999-2003)
  2 = EX era (2003-2007)
  3 = Diamond & Pearl (2007-2010)
  4 = HeartGold SoulSilver (2010-2011)
  5 = Black & White (2011-2013)
  6 = XY (2014-2016)
  7 = Sun & Moon (2017-2019)
  8 = Sword & Shield (2020-2022)
  9 = Scarlet & Violet (2023+)
"""

# Exhaustive mapping of every set prefix to its era number.
# Built from the full list of 171 prefixes in card_names.json.
SET_TO_ERA: dict[str, int] = {
    # Era 1: WotC Classic (1999-2003)
    "base1": 1,   # Base Set
    "base2": 1,   # Jungle
    "base3": 1,   # Fossil
    "base4": 1,   # Base Set 2
    "base5": 1,   # Team Rocket
    "base6": 1,   # Legendary Collection
    "basep": 1,   # Wizards Black Star Promos
    "gym1": 1,    # Gym Heroes
    "gym2": 1,    # Gym Challenge
    "neo1": 1,    # Neo Genesis
    "neo2": 1,    # Neo Discovery
    "neo3": 1,    # Neo Revelation
    "neo4": 1,    # Neo Destiny
    "ecard1": 1,  # Expedition Base Set
    "ecard2": 1,  # Aquapolis
    "ecard3": 1,  # Skyridge
    "bp": 1,      # Best of Game
    "si1": 1,     # Southern Islands

    # Era 2: EX era (2003-2007)
    "ex1": 2,     # Ruby & Sapphire
    "ex2": 2,     # Sandstorm
    "ex3": 2,     # Dragon
    "ex4": 2,     # Team Magma vs Team Aqua
    "ex5": 2,     # Hidden Legends
    "ex6": 2,     # FireRed & LeafGreen
    "ex7": 2,     # Team Rocket Returns
    "ex8": 2,     # Deoxys
    "ex9": 2,     # Emerald
    "ex10": 2,    # Unseen Forces
    "ex11": 2,    # Delta Species
    "ex12": 2,    # Legend Maker
    "ex13": 2,    # Holon Phantoms
    "ex14": 2,    # Crystal Guardians
    "ex15": 2,    # Dragon Frontiers
    "ex16": 2,    # Power Keepers
    "pop1": 2,    # POP Series 1
    "pop2": 2,    # POP Series 2
    "pop3": 2,    # POP Series 3
    "pop4": 2,    # POP Series 4
    "pop5": 2,    # POP Series 5
    "tk1a": 2,    # Trainer Kit (EX)
    "tk1b": 2,    # Trainer Kit (EX)
    "tk2a": 2,    # Trainer Kit 2 (EX)
    "tk2b": 2,    # Trainer Kit 2 (EX)
    "np": 2,      # Nintendo Black Star Promos

    # Era 3: Diamond & Pearl (2007-2010)
    "dp1": 3,     # Diamond & Pearl
    "dp2": 3,     # Mysterious Treasures
    "dp3": 3,     # Secret Wonders
    "dp4": 3,     # Great Encounters
    "dp5": 3,     # Majestic Dawn
    "dp6": 3,     # Legends Awakened
    "dp7": 3,     # Stormfront
    "dpp": 3,     # DP Black Star Promos
    "pl1": 3,     # Platinum
    "pl2": 3,     # Rising Rivals
    "pl3": 3,     # Supreme Victors
    "pl4": 3,     # Arceus
    "pop6": 3,    # POP Series 6
    "pop7": 3,    # POP Series 7
    "pop8": 3,    # POP Series 8
    "pop9": 3,    # POP Series 9

    # Era 4: HeartGold SoulSilver (2010-2011)
    "hgss1": 4,   # HeartGold & SoulSilver
    "hgss2": 4,   # Unleashed
    "hgss3": 4,   # Undaunted
    "hgss4": 4,   # Triumphant
    "hsp": 4,     # HGSS Black Star Promos
    "col1": 4,    # Call of Legends
    "ru1": 4,     # Rumble

    # Era 5: Black & White (2011-2013)
    "bw1": 5,     # Black & White
    "bw2": 5,     # Emerging Powers
    "bw3": 5,     # Noble Victories
    "bw4": 5,     # Next Destinies
    "bw5": 5,     # Dark Explorers
    "bw6": 5,     # Dragons Exalted
    "bw7": 5,     # Boundaries Crossed
    "bw8": 5,     # Plasma Storm
    "bw9": 5,     # Plasma Freeze
    "bw10": 5,    # Plasma Blast
    "bw11": 5,    # Legendary Treasures
    "bwp": 5,     # BW Black Star Promos
    "dv1": 5,     # Dragon Vault
    "dc1": 5,     # Double Crisis

    # Era 6: XY (2014-2016)
    "xy0": 6,     # Kalos Starter Set
    "xy1": 6,     # XY
    "xy2": 6,     # Flashfire
    "xy3": 6,     # Furious Fists
    "xy4": 6,     # Phantom Forces
    "xy5": 6,     # Primal Clash
    "xy6": 6,     # Roaring Skies
    "xy7": 6,     # Ancient Origins
    "xy8": 6,     # BREAKthrough
    "xy9": 6,     # BREAKpoint
    "xy10": 6,    # Fates Collide
    "xy11": 6,    # Steam Siege
    "xy12": 6,    # Evolutions
    "xyp": 6,     # XY Black Star Promos
    "g1": 6,      # Generations
    "me1": 6,     # Mythical Collection (Mew)
    "me2": 6,     # Mythical Collection (Celebi onwards)
    "me2pt5": 6,  # Mythical Collection (additional)

    # Era 7: Sun & Moon (2017-2019)
    "sm1": 7,     # Sun & Moon
    "sm2": 7,     # Guardians Rising
    "sm3": 7,     # Burning Shadows
    "sm35": 7,    # Shining Legends
    "sm4": 7,     # Crimson Invasion
    "sm5": 7,     # Ultra Prism
    "sm6": 7,     # Forbidden Light
    "sm7": 7,     # Celestial Storm
    "sm75": 7,    # Dragon Majesty
    "sm8": 7,     # Lost Thunder
    "sm9": 7,     # Team Up
    "sm10": 7,    # Unbroken Bonds
    "sm11": 7,    # Unified Minds
    "sm115": 7,   # Hidden Fates
    "sm12": 7,    # Cosmic Eclipse
    "smp": 7,     # SM Black Star Promos
    "sma": 7,     # Shiny Vault (Hidden Fates)
    "det1": 7,    # Detective Pikachu
    "mcd18": 7,   # McDonald's Collection 2018
    "mcd19": 7,   # McDonald's Collection 2019

    # Era 8: Sword & Shield (2020-2022)
    "swsh1": 8,   # Sword & Shield
    "swsh2": 8,   # Rebel Clash
    "swsh3": 8,   # Darkness Ablaze
    "swsh35": 8,  # Champion's Path
    "swsh4": 8,   # Vivid Voltage
    "swsh45": 8,  # Shining Fates
    "swsh45sv": 8,  # Shining Fates Shiny Vault
    "swsh5": 8,   # Battle Styles
    "swsh6": 8,   # Chilling Reign
    "swsh7": 8,   # Evolving Skies
    "swsh8": 8,   # Fusion Strike
    "swsh9": 8,   # Brilliant Stars
    "swsh9tg": 8, # Brilliant Stars Trainer Gallery
    "swsh10": 8,  # Astral Radiance
    "swsh10tg": 8,  # Astral Radiance Trainer Gallery
    "swsh11": 8,  # Lost Origin
    "swsh11tg": 8,  # Lost Origin Trainer Gallery
    "swsh12": 8,  # Silver Tempest
    "swsh12tg": 8,  # Silver Tempest Trainer Gallery
    "swsh12pt5": 8,   # Crown Zenith
    "swsh12pt5gg": 8, # Crown Zenith Galarian Gallery
    "swshp": 8,   # SWSH Black Star Promos
    "cel25": 8,   # Celebrations
    "cel25c": 8,  # Celebrations Classic Collection
    "pgo": 8,     # Pokemon GO
    "fut20": 8,   # Futsal Collection
    "mcd21": 8,   # McDonald's Collection 2021
    "mcd22": 8,   # McDonald's Collection 2022

    # Era 9: Scarlet & Violet (2023+)
    "sv1": 9,     # Scarlet & Violet
    "sv2": 9,     # Paldea Evolved
    "sv3": 9,     # Obsidian Flames
    "sv3pt5": 9,  # 151
    "sv4": 9,     # Paradox Rift
    "sv4pt5": 9,  # Paldean Fates
    "sv5": 9,     # Temporal Forces
    "sv6": 9,     # Twilight Masquerade
    "sv6pt5": 9,  # Shrouded Fable
    "sv7": 9,     # Stellar Crown
    "sv8": 9,     # Surging Sparks
    "sv8pt5": 9,  # Prismatic Evolutions
    "sv9": 9,     # Journey Together
    "sv10": 9,    # ???
    "sve": 9,     # SV Energy
    "svp": 9,     # SV Black Star Promos
    "rsv10pt5": 9,  # ???
    "zsv10pt5": 9,  # ???
}

# McDonald's promos spanning multiple eras - map to the closest era
SET_TO_ERA.update({
    "mcd11": 5,   # McDonald's Collection 2011 (BW era)
    "mcd12": 5,   # McDonald's Collection 2012 (BW era)
    "mcd14": 6,   # McDonald's Collection 2014 (XY era)
    "mcd15": 6,   # McDonald's Collection 2015 (XY era)
    "mcd16": 6,   # McDonald's Collection 2016 (XY era)
    "mcd17": 7,   # McDonald's Collection 2017 (SM era)
})

ERA_NAMES: dict[int, str] = {
    0: "Unknown",
    1: "WotC Classic (1999-2003)",
    2: "EX Era (2003-2007)",
    3: "Diamond & Pearl (2007-2010)",
    4: "HeartGold SoulSilver (2010-2011)",
    5: "Black & White (2011-2013)",
    6: "XY (2014-2016)",
    7: "Sun & Moon (2017-2019)",
    8: "Sword & Shield (2020-2022)",
    9: "Scarlet & Violet (2023+)",
}


def get_card_era(card_id: str) -> int:
    """Return the era number (1-9) for a card ID, or 0 if unknown.

    Args:
        card_id: Full card ID like "base1-4/holofoil" or set-prefixed ID like "base1-4".

    Returns:
        Era number 1-9, or 0 if the set prefix is not recognized.
    """
    # Strip variant suffix if present (e.g. "base1-4/holofoil" -> "base1-4")
    bare_id = card_id.split("/")[0]
    # Extract set prefix (e.g. "base1-4" -> "base1")
    set_id = bare_id.rsplit("-", 1)[0]
    return SET_TO_ERA.get(set_id, 0)


def get_era_name(era: int) -> str:
    """Return human-readable name for an era number."""
    return ERA_NAMES.get(era, "Unknown")


def get_era_sets(era: int) -> list[str]:
    """Return all set prefixes belonging to a given era."""
    return [s for s, e in SET_TO_ERA.items() if e == era]
