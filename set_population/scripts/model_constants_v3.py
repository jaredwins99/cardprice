"""v3 shared constants: English-share layer + checkpoint ladder loader.

Single source of truth for the things fit_grading_rate.py and combine.py
previously duplicated (the TPC checkpoint table) or silently omitted (the
English-vs-all-languages unit distinction).

v3's central change (2026-07-21 deep-research pass, see
docs/anchor_research_2026-07-21.md and known_print_runs.json):

  * All TPC "Pokemon in Figures" cumulative checkpoints are GLOBAL,
    all-languages production. The model's universe is the ENGLISH catalog
    (dim_sets). v2 compared the two directly - an implicit english_share of
    1.0. v3 makes the share explicit, sourced, and propagated.
  * The WOTC per-set community anchors are ALL-LANGUAGES numbers (their author
    says so; halve for English). Anchors tagged unit="cards_all_languages" are
    converted before use.

ENGLISH SHARE ASSUMPTION (documented in known_print_runs.json
conversion_evidence.english_share_of_global):
  No official language-share figure exists. Evidence triangulated:
  - INDIRECT value datum: Japan card-game retail (~US$1.9B FY2024, ALL card
    games) is the same order as US Pokemon retail (Circana >$1B). Retail VALUE,
    not print volume (JP packs 5 cards vs 11 EN) — a loose constraint only.
  - Production split: TPCi prints Western languages, TPC prints Japan/Asia.
  - Language count grew 11 (2015) -> 13 (2020) -> 16 (2026), diluting EN.
  Central values: 0.40 before 2020, 0.35 from 2020 (SWSH/SV; more languages,
  Japan boom). Uncertainty +/-0.10 is reported as a sensitivity, and is one of
  the components of the +/-3x band on absolutes.
  The SEC-revenue-derived WOTC window (4.2-7.4B EN 1999-2001) is CONSISTENT
  with 12B x 0.40 = 4.8B, but this consistency partly informed the choice of
  0.40 — it is a coherence check between assumption-sharing derivations, NOT
  independent corroboration of the share (docs audit 2026-07-22).

CUMULATIVE CONVERSION (v3.1 fix): the share is applied to production
INCREMENTS per regime, not to cumulative totals — multiplying a cumulative
total by the share at its date made the English ladder non-monotonic across
the 2020 switch (cumulative EN cannot decrease). EN_cum(T) =
0.40 * G(min(T, switch)) + 0.35 * max(0, G(T) - G(switch)), with G(switch)
interpolated from the global ladder itself.
"""

from datetime import date, datetime

EN_SHARE_PRE2020 = 0.40
EN_SHARE_POST2020 = 0.35
EN_SHARE_BAND = 0.10  # +/- absolute, for sensitivity reporting
EN_SHARE_SWITCH = date(2020, 1, 1)

MODEL_VERSION_V3 = "v3-english-only"

# A set's production is not instantaneous at release: modern sets get reprint
# waves 12-24 months out (documented: the 2021 Cosmic Eclipse reprint ~18
# months after release). A cumulative checkpoint at date T therefore only
# contains part of the lifetime run of sets released shortly before T. We
# credit a set released d days before a checkpoint with min(1, d / ramp) of
# its lifetime production — a uniform-over-ramp approximation.
#
# The ramp is SHORTER for boom-era sets (released before 2003): Hasbro's
# FY2001 10-K documents obsolescence writeoffs from Pokemon OVERproduction and
# a >=50% revenue collapse — print runs were front-loaded during the mania and
# reprinting stopped dead in the 2001-02 glut, so a 12-month ramp fits that
# era; without it the fit overshoots the official 13B Mar-2005 checkpoint by
# ~1.6x purely from phantom post-crash production of 2000-01 sets.
# ASSUMPTION (mechanistically motivated, not fitted); consequence: per-set
# absolute estimates are projected LIFETIME production.
PRODUCTION_RAMP_DAYS = 730
PRODUCTION_RAMP_DAYS_BOOM = 365
# Cutoff aligns with the EX-era start so the 2003 crash-bottom ECARD tail
# (Aquapolis, Skyridge — released into the glut, famously never reprinted)
# also gets the short ramp.
BOOM_RAMP_CUTOFF = date(2003, 6, 1)

# Subset products physically contained in a parent set's sealed product; their
# production is part of the parent's and must not double-claim checkpoint
# window volume. Estimates for these are emitted flagged, not calibrated.
SUBSET_PARENT = {
    "cel25c": "cel25",
    "swsh12pt5gg": "swsh12pt5",
    "swsh9tg": "swsh9",
    "swsh10tg": "swsh10",
    "swsh11tg": "swsh11",
    "swsh12tg": "swsh12",
}


def production_weight(release_date, cp_date):
    """Fraction of a set's lifetime production completed by cp_date."""
    if release_date is None or release_date > cp_date:
        return 0.0
    ramp = (PRODUCTION_RAMP_DAYS_BOOM if release_date < BOOM_RAMP_CUTOFF
            else PRODUCTION_RAMP_DAYS)
    return min(1.0, (cp_date - release_date).days / ramp)

# Loss weight for cumulative-checkpoint terms, by anchor credibility.
CHECKPOINT_W = {"official": 8.0, "well-sourced-estimate": 4.0}


def english_share(d):
    """Fraction of GLOBAL cumulative card production that is English-language,
    as of date d. Piecewise-constant assumption with documented evidence; see
    module docstring. Applied to cumulative totals, so the pre-1999
    Japanese-only years are implicitly part of the non-English share."""
    if d is None:
        return EN_SHARE_PRE2020
    return EN_SHARE_POST2020 if d >= EN_SHARE_SWITCH else EN_SHARE_PRE2020


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return datetime.fromisoformat(s[:10]).date()


# Rungs dated after (chase-pop snapshot - ROSTER_LAG_DAYS) are excluded from
# the fit loss and never used as the calibration pin: the scored roster lacks
# all 2025-26 releases (absent from the PSA mirror) and recently released sets
# have collapsed graded-pop numerators (e.g. Prismatic Evolutions), so late
# rungs would force older sets to absorb recent production (red-team audit
# 2026-07-22). With the 2026-05-27 snapshot this caps usable rungs at
# Mar-2024.
ROSTER_LAG_DAYS = 730


def load_checkpoints(known_runs):
    """Return the dated cumulative-checkpoint ladder from GLOBAL anchors.

    Each entry: {value_global, value_english, date, credibility, weight,
    usable_for_fit}. value_english applies the share to per-regime INCREMENTS
    (see module docstring) so the English ladder is monotone. Only GLOBAL
    total_print_run anchors with value_mid + as_of_date qualify.
    """
    raw = []
    for a in known_runs["anchors"]:
        if a.get("set_id") != "GLOBAL":
            continue
        if a.get("estimate_type") != "total_print_run":
            continue
        if a.get("value_mid") is None or not a.get("as_of_date"):
            continue
        d = _parse_date(a["as_of_date"])
        cred = a.get("source_credibility", "well-sourced-estimate")
        if cred not in CHECKPOINT_W:
            continue  # never let low-trust data in at checkpoint weight
        raw.append((d, float(a["value_mid"]), cred))
    raw.sort()

    # G(switch): interpolate global cumulative at the share-switch date
    g_switch = None
    for i in range(len(raw) - 1):
        (d0, v0, _), (d1, v1, _) = raw[i], raw[i + 1]
        if d0 <= EN_SHARE_SWITCH <= d1:
            frac = (EN_SHARE_SWITCH - d0).days / max(1, (d1 - d0).days)
            g_switch = v0 + (v1 - v0) * frac
            break
    if g_switch is None and raw:
        g_switch = raw[-1][1] if raw[-1][0] < EN_SHARE_SWITCH else raw[0][1]

    def en_cum(d, v_global):
        if d < EN_SHARE_SWITCH:
            return v_global * EN_SHARE_PRE2020
        return (g_switch * EN_SHARE_PRE2020
                + (v_global - g_switch) * EN_SHARE_POST2020)

    out = []
    for d, v, cred in raw:
        out.append({
            "value_global": v,
            "value_english": en_cum(d, v),
            "date": d,
            "credibility": cred,
            "weight": CHECKPOINT_W[cred],
            "g_at_switch": g_switch,
        })
    return out


def calibration_scale(rung_terms):
    """Credibility-weighted geometric-mean scale across usable rungs.

    rung_terms: [(target_english, predicted_unscaled_sum, weight), ...].
    Replaces v2's exact-pin-to-latest-rung, which concentrated the ladder's
    residual slope misfit into whichever rung was last (post-fit scale 1.38
    inflating every earlier rung ~1.4-1.9x). With the geomean no rung is
    privileged and residuals are reported where they actually lie.
    """
    import math as _math
    num = den = 0.0
    for target, pred, w in rung_terms:
        if target > 0 and pred > 0:
            num += w * (_math.log(target) - _math.log(pred))
            den += w
    return _math.exp(num / den) if den else 1.0


def usable_rungs(checkpoints, pop_snapshot_date):
    """Rungs usable in the fit loss / as the calibration pin: dated no later
    than pop_snapshot_date - ROSTER_LAG_DAYS (see ROSTER_LAG_DAYS comment)."""
    from datetime import timedelta
    cutoff = pop_snapshot_date - timedelta(days=ROSTER_LAG_DAYS)
    return [c for c in checkpoints if c["date"] <= cutoff]


def load_english_window_anchors(known_runs):
    """Window anchors already denominated in ENGLISH cards (e.g. the
    SEC-revenue-derived WOTC 1999-2001 total). Each entry:
    {set_id, window_start, window_end, value_mid, value_low, value_high,
     weight, source_url}."""
    out = []
    for a in known_runs["anchors"]:
        if a.get("estimate_type") != "english_window_total":
            continue
        if a.get("value_mid") is None:
            continue
        ws = _parse_date(a.get("window_start")) or date(1999, 1, 1)
        we = _parse_date(a.get("window_end")) or _parse_date(a.get("as_of_date"))
        if we is None:
            continue  # a window without an end date cannot be masked
        cred = a.get("source_credibility", "well-sourced-estimate")
        out.append({
            "set_id": a["set_id"],
            "window_start": ws,
            "window_end": we,
            "value_mid": float(a["value_mid"]),
            "value_low": a.get("value_low"),
            "value_high": a.get("value_high"),
            "weight": CHECKPOINT_W.get(cred, 4.0),
            "source_url": a.get("source_url"),
        })
    return out


def anchor_value_english(anchor, set_release_date):
    """Convert a per-set anchor's value_mid to English cards.

    Anchors tagged unit="cards_all_languages" get multiplied by the English
    share at the set's release date; untagged anchors are assumed English
    already (none of the currently-used per-set anchors are untagged after the
    2026-07-21 provenance audit, but the fallback is explicit)."""
    v = anchor.get("value_mid")
    if v is None:
        return None, False
    if anchor.get("unit") == "cards_all_languages":
        return float(v) * english_share(set_release_date), True
    return float(v), False


def share_doc():
    """Metadata block describing the share assumption, embedded in outputs."""
    return {
        "english_share_pre2020": EN_SHARE_PRE2020,
        "english_share_post2020": EN_SHARE_POST2020,
        "english_share_band": EN_SHARE_BAND,
        "evidence": ("No official language share exists. Japan market ~ US market "
                     "(FY2024); TPCi prints Western langs, TPC prints JP/Asia; "
                     "language count 11 (2015) -> 16 (2026). WOTC cross-check: "
                     "12B global end-2001 x 0.40 = 4.8B EN vs SEC-revenue-derived "
                     "4.2-7.4B EN for 1999-2001. See known_print_runs.json "
                     "conversion_evidence.english_share_of_global."),
    }
