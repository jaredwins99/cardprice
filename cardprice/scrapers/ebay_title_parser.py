"""Parse eBay listing titles into structured Pokemon card metadata.

Handles common title formats like:
    "Pokemon Card Charizard Base Set 4/102 PSA 10"
    "Pikachu VMAX 044/185 Vivid Voltage CGC 9.5 Near Mint+"
    "Mew ex 151 NM 053/165"
"""

import re

# ---------------------------------------------------------------------------
# Grading authorities and patterns
# ---------------------------------------------------------------------------

GRADING_AUTHORITIES = {
    "PSA": r"\bPSA\b",
    "BGS": r"\bBGS\b",
    "CGC": r"\bCGC\b",
    "SGC": r"\bSGC\b",
    "ACE": r"\bACE\b",
    "AGS": r"\bAGS\b",
}

# Matches grading patterns like "PSA 10", "BGS 9.5", "CGC 8"
GRADE_PATTERN = re.compile(
    r"\b(PSA|BGS|CGC|SGC|ACE|AGS)\s+([\d]+(?:\.[\d]+)?)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Condition keywords (raw/ungraded cards)
# ---------------------------------------------------------------------------

CONDITION_MAP = {
    "NM": r"\bNM\b|\bNear\s*Mint\b",
    "LP": r"\bLP\b|\bLight(?:ly)?\s*Play(?:ed)?\b",
    "MP": r"\bMP\b|\bModer?at(?:e|ely)\s*Play(?:ed)?\b",
    "HP": r"\bHP\b|\bHeav(?:y|ily)\s*Play(?:ed)?\b",
    "DMG": r"\bDMG\b|\bDamaged\b",
}

# ---------------------------------------------------------------------------
# Card number patterns (e.g. 4/102, 044/185, SV049/SV122, TG30/TG30)
# ---------------------------------------------------------------------------

CARD_NUMBER_PATTERN = re.compile(
    r"\b([A-Z]*\d+)\s*/\s*([A-Z]*\d+)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Set name indicators and known set keywords
# ---------------------------------------------------------------------------

# Common noise words to strip from titles
NOISE_WORDS = re.compile(
    r"\b(pokemon|card|tcg|trading|game|holo|rare|ultra|secret|full\s*art|"
    r"alt\s*art|illustration|special|promo|japanese|english|mint|pack\s*fresh|"
    r"lot|bundle|collection|singles?|official|authentic|genuine|nm|lp|mp|hp|dmg|"
    r"near\s*mint|lightly\s*played|moderately\s*played|heavily\s*played|damaged|"
    r"new|sealed|opened)\b",
    re.IGNORECASE,
)

# Noise at the start: "Pokemon Card", "Pokemon TCG", etc.
TITLE_PREFIX_NOISE = re.compile(
    r"^(?:pokemon\s+(?:card|tcg|trading\s+card\s+game)\s*[-:]?\s*)+",
    re.IGNORECASE,
)

# Known set names (partial list — the matcher does fuzzy matching against dim_sets)
KNOWN_SET_KEYWORDS = [
    "Base Set", "Jungle", "Fossil", "Team Rocket", "Gym Heroes", "Gym Challenge",
    "Neo Genesis", "Neo Discovery", "Neo Revelation", "Neo Destiny",
    "Legendary Collection", "Expedition", "Aquapolis", "Skyridge",
    "Ruby & Sapphire", "Sandstorm", "Dragon", "Team Magma vs Team Aqua",
    "Hidden Legends", "FireRed & LeafGreen",
    "Scarlet & Violet", "Paldea Evolved", "Obsidian Flames", "Paradox Rift",
    "Paldean Fates", "Temporal Forces", "Twilight Masquerade", "Shrouding Storm",
    "Surging Sparks", "Prismatic Evolutions", "Journey Together",
    "Sword & Shield", "Brilliant Stars", "Astral Radiance", "Lost Origin",
    "Silver Tempest", "Crown Zenith",
    "Sun & Moon", "Burning Shadows", "Cosmic Eclipse", "Hidden Fates",
    "Vivid Voltage", "Evolving Skies", "Fusion Strike", "Chilling Reign",
    "Battle Styles", "Celebrations",
    "XY", "Evolutions", "Generations", "Breakpoint", "Fates Collide",
    "151", "Shining Fates",
]

# Pre-compile set patterns for matching in titles
_SET_PATTERNS = [
    (name, re.compile(re.escape(name), re.IGNORECASE))
    for name in sorted(KNOWN_SET_KEYWORDS, key=len, reverse=True)
]


def parse_title(title: str) -> dict:
    """Parse an eBay listing title into structured card metadata.

    Args:
        title: Raw eBay listing title string.

    Returns:
        Dict with keys:
            card_name:         str | None — Pokemon/card name
            set_name:          str | None — set name if detected
            card_number:       str | None — e.g. "4/102"
            grading_authority: str | None — e.g. "PSA", "BGS", "CGC"
            grade:             str | None — e.g. "10", "9.5"
            is_graded:         bool
            condition:         str | None — NM/LP/MP/HP/DMG (for raw cards)
    """
    result = {
        "card_name": None,
        "set_name": None,
        "card_number": None,
        "grading_authority": None,
        "grade": None,
        "is_graded": False,
        "condition": None,
    }

    if not title or not title.strip():
        return result

    working = title.strip()

    # --- Grading ---
    grade_match = GRADE_PATTERN.search(working)
    if grade_match:
        result["grading_authority"] = grade_match.group(1).upper()
        result["grade"] = grade_match.group(2)
        result["is_graded"] = True
        # Remove the grade text from working title
        working = working[:grade_match.start()] + working[grade_match.end():]
    else:
        # Check for grading authority without numeric grade (e.g. "PSA Gem Mint")
        for authority, pattern in GRADING_AUTHORITIES.items():
            if re.search(pattern, working, re.IGNORECASE):
                result["grading_authority"] = authority
                result["is_graded"] = True
                working = re.sub(pattern, "", working, flags=re.IGNORECASE)
                break

    # --- Card number ---
    number_match = CARD_NUMBER_PATTERN.search(working)
    if number_match:
        result["card_number"] = f"{number_match.group(1)}/{number_match.group(2)}"
        working = working[:number_match.start()] + working[number_match.end():]

    # --- Set name ---
    for set_name, pattern in _SET_PATTERNS:
        if pattern.search(working):
            result["set_name"] = set_name
            working = pattern.sub("", working)
            break

    # --- Condition (raw cards only) ---
    if not result["is_graded"]:
        for condition, pattern in CONDITION_MAP.items():
            if re.search(pattern, working, re.IGNORECASE):
                result["condition"] = condition
                working = re.sub(pattern, "", working, flags=re.IGNORECASE)
                break

    # --- Card name: whatever meaningful text remains ---
    # Strip common prefix noise
    working = TITLE_PREFIX_NOISE.sub("", working)
    # Strip remaining noise words
    working = NOISE_WORDS.sub("", working)
    # Clean up extra whitespace and punctuation debris
    working = re.sub(r"[#\-–—|,]+", " ", working)
    working = re.sub(r"\s{2,}", " ", working).strip()
    # Remove trailing/leading punctuation
    working = re.sub(r"^[\s\-–—:,!.]+|[\s\-–—:,!.]+$", "", working)

    if working and len(working) >= 2:
        result["card_name"] = working
    else:
        # Fallback: try to grab the first capitalized words from original title
        fallback = TITLE_PREFIX_NOISE.sub("", title.strip())
        words = fallback.split()
        name_parts = []
        for w in words:
            # Stop at card numbers, grades, or known noise
            if re.match(r"^\d+/\d+$", w) or re.match(r"^(PSA|BGS|CGC|SGC|NM|LP|MP|HP|DMG)$", w, re.IGNORECASE):
                break
            if not re.match(r"^\d+$", w):
                name_parts.append(w)
            if len(name_parts) >= 3:
                break
        if name_parts:
            result["card_name"] = " ".join(name_parts)

    return result
