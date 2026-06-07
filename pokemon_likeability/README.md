# pokemon_likeability — pairwise Pokémon ranking via ELO

> **Self-contained sub-project, isolated from the rest of the cardprice repo.**
> Nothing here is imported by the card scanner, the pricing pipeline, or the
> ML code. The only external touch-point is a small set of routes registered
> in `cardprice/server.py` so the UI can be served from the same host as
> `/annotate` and `/cropper`. All data lives under `pokemon_likeability/data/`.
>
> If you're working on the scanner / pricing / ML pipeline, you can ignore
> this directory entirely.

## Goal

Produce a defensible 10-tier ranking of all ~1025 Pokémon species by personal
likeability, using pairwise comparisons. The user is shown two Pokémon at a
time and picks one (or "tie"); an ELO engine updates the ratings; the next
pair is chosen by active learning to maximise information gain.

## Components

| Piece | Where |
|---|---|
| Roster (species + sprite URLs) | `data/roster.json` |
| Live ELO ratings | `data/ratings.json` |
| Append-only vote log | `data/votes.jsonl` |
| Bootstrap script (PokéAPI fetch) | `scripts/bootstrap_roster.py` |
| ELO engine + active pair picker | `cardprice/server.py` (`_likeability_*` methods) |
| HTML page + JSON routes | `cardprice/server.py` (`/likeability*`) |

## ELO

Standard ELO. `E_a = 1/(1+10^((R_b-R_a)/400))`. After outcome `S_a ∈ {1, 0.5, 0}`
the update is `R_a += K*(S_a - E_a)`, symmetric for B. K is `32 / 16 / 8` for
`min(n_a, n_b) < 10 / < 30 / else`, so early comparisons move ratings fast and
late comparisons fine-tune.

## Active pair selection

1. Pick the 200 species with the fewest comparisons (`n` ascending) — this is
   the coverage pool.
2. From that pool, sample two with ratings within ~150 ELO of each other (most
   informative comparison). If no close pair, fall back to random pair from the
   pool.
3. Avoid pairs we've shown in the last ~50 votes (to reduce repeat-fatigue).

## Seed boost

A handful of species start above 1000 so the early questions are about the
user's favourites:

* Tier 1 (1500): Charizard, Gengar, Umbreon, Pikachu, Mewtwo
* Tier 2 (1350): Espeon, Dragonite, Bulbasaur, Mew
* Tier 3 (1200): Snorlax, Eevee, Lugia, Giratina
* Tier 4 (1100): Arcanine
* Tier 5 (1000): everyone else (default)

## Routes (live on the cardprice server)

| Verb + path | Body / params | Purpose |
|---|---|---|
| `GET /likeability` | — | Serve HTML page |
| `GET /likeability/next` | — | Pick next pair, return `{a, b, comparisons_total, seeded_count}` |
| `POST /likeability/vote` | `{winner_id, loser_id}` or `{a_id, b_id, tie: true}` | Update ratings, append vote |
| `DELETE /likeability/vote` | — | Undo last vote (rewind ratings) |
| `GET /likeability/stats` | — | Top 50 + bottom 20 + total + histogram + 10-quantile tier boundaries |

## Usage

```bash
# One-time roster fetch (idempotent — won't blow away ratings/votes)
python3 pokemon_likeability/scripts/bootstrap_roster.py

# Then open http://<server>:8888/likeability in a browser
```

## Output: 10 ordinal tiers

`GET /likeability/stats` returns `tier_boundaries`, the 10-quantile cuts over
the rating distribution. After enough comparisons each Pokémon falls into one
of 10 tiers based on its final ELO.
