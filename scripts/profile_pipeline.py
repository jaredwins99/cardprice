#!/usr/bin/env python3
"""Profile the V2 card identification pipeline to find speed bottlenecks.

Two modes:
  1. Live profiling (default): Runs identify_page_v2() on eval pages,
     captures internal timing logs, reports per-stage breakdown.
  2. Report mode (--report): Analyzes existing eval results + partial
     live profiling data when live run is not possible (e.g. OOM).

Usage:
    python scripts/profile_pipeline.py             # all pages, live
    python scripts/profile_pipeline.py 0            # page 0 only, live
    python scripts/profile_pipeline.py --report     # analyze existing data
"""

import gc
import json
import logging
import os
import re
import sys
import time
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "binder_eval.json"
EVAL_RESULTS_PATH = PROJECT_ROOT / "data" / "eval" / "v2_eval_results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_time(log_text, pattern):
    """Extract a float timing value from log output using regex."""
    m = re.search(pattern, log_text)
    return float(m.group(1)) if m else None


def _fmt(val):
    if val is None:
        return "   N/A"
    return f"{val:6.1f}s"


def _avg_stage(timings, key):
    vals = [t[key] for t in timings if t.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Report mode: analyze existing eval results
# ---------------------------------------------------------------------------
def report_mode():
    """Analyze timing from existing eval results + live profiling observations."""
    if not EVAL_RESULTS_PATH.exists():
        print(f"ERROR: {EVAL_RESULTS_PATH} not found. Run eval_v2.py first.")
        sys.exit(1)

    d = json.load(open(EVAL_RESULTS_PATH))
    summary = d["summary"]

    print("=" * 72)
    print("V2 PIPELINE PROFILING REPORT")
    print("=" * 72)

    # --- Page-level wall time from eval ---
    print("\n--- PAGE-LEVEL WALL TIME (from eval_v2.py results) ---")
    page_data = {}
    for r in d["results"]:
        pi = r.get("page_index", -1)
        if pi not in page_data:
            page_data[pi] = {"cards": 0, "time": 0.0, "methods": {}}
        page_data[pi]["cards"] += 1
        page_data[pi]["time"] += r.get("time_seconds", 0)
        m = r.get("predicted_method", "?")
        page_data[pi]["methods"][m] = page_data[pi]["methods"].get(m, 0) + 1

    for pi in sorted(page_data):
        p = page_data[pi]
        avg = p["time"] / p["cards"] if p["cards"] else 0
        methods_str = ", ".join(f"{m}={n}" for m, n in sorted(p["methods"].items()))
        print(f"  Page {pi}: {p['cards']:2d} cards, {p['time']:5.1f}s total, "
              f"{avg:.1f}s/card  [{methods_str}]")

    total_time = summary["total_time_seconds"]
    total_cards = summary["scored_cards"]
    print(f"  TOTAL:  {summary['total_cards']:2d} cards ({total_cards} scored), "
          f"{total_time:.1f}s, {summary['avg_time_per_card']:.1f}s/card")

    # --- Internal stage timing from live profiling ---
    print("\n--- INTERNAL STAGE TIMING (from live profiling, 2 partial runs) ---")
    print()
    print("  The pipeline runs 2 parallel threads for pre-computation:")
    print()
    print("  Thread 1 - PaddleOCR (sequential per card):")
    print("    Per card: name OCR + HP detect + color detect (~0.5s)")
    print("            + attack OCR via PaddleOCR det+rec  (~12-18s cold, ~7-9s warm)")
    print("    9 cards total (cold, first page): ~100-140s estimated")
    print("    9 cards total (warm, cached):     ~65-80s estimated")
    print()
    print("  Thread 2 - Embeddings (batch):")
    print("    Preprocessing (CLAHE + border crop): ~1s/card = ~9s")
    print("    DINOv2 model loading (first call):   ~13s")
    print("    FSRCNN upscaler loading (first call): ~3s")
    print("    DINOv2 batch inference (9 cards):    ~2s")
    print("    Total (cold): 33-34s   Total (warm): ~5-8s")
    print()
    print("  Pre-computation wall = max(Thread 1, Thread 2)")
    print("    Cold: max(~120s, ~34s) = ~120s")
    print("    Warm: max(~70s,  ~7s)  = ~70s")
    print()
    print("  Card identification (parallel threads after precomp):")
    print("    DB candidate lookup + DINOv2 dot product + fuzzy attack match")
    print("    ~0.5-2s total (all cards in parallel)")
    print()
    print("  Page context (Pass 2 + Pass 3): ~1-2s")

    # --- Bottleneck analysis ---
    print("\n" + "=" * 72)
    print("BOTTLENECK ANALYSIS")
    print("=" * 72)

    print()
    print("  Bottleneck #1: PaddleOCR Attack OCR  (>80% of wall time)")
    print("  " + "-" * 55)
    print("    - Sequential: each card needs full PaddleOCR det+rec pass")
    print("    - ~12-18s/card cold, ~7-9s/card warm")
    print("    - 9 cards = ~65-135s depending on cache state")
    print("    - The embeddings thread finishes 3-10x earlier and idles")
    print()
    print("  Bottleneck #2: DINOv2 Model Loading  (first page only)")
    print("  " + "-" * 55)
    print("    - torch.hub.load + CUDA transfer = ~13s")
    print("    - Amortized in server mode (model stays in memory)")
    print("    - After loading, batch inference is only ~2s for 9 cards")
    print()
    print("  Bottleneck #3: PaddleOCR Model Loading  (first page only)")
    print("  " + "-" * 55)
    print("    - PP-OCRv5 det + rec models = ~10-12s first load")
    print("    - Overlaps with DINOv2 loading in parallel threads")

    # --- Estimated time budget ---
    print("\n" + "=" * 72)
    print("ESTIMATED TIME BUDGET (per 9-card page)")
    print("=" * 72)

    budget = [
        ("PaddleOCR attack OCR (9 cards seq.)", 70, 120, True),
        ("PaddleOCR name+HP+color (9 cards)",   5,   8, False),
        ("DINOv2 batch embedding",              2,   5, False),
        ("Preprocessing (CLAHE+crop)",          5,   9, False),
        ("Card identification (parallel)",      1,   3, False),
        ("Page context passes 2+3",             1,   2, False),
    ]

    print(f"\n  {'Stage':<42s} {'Warm':>5s}  {'Cold':>5s}  {'Critical':>8s}")
    print(f"  {'-'*42} {'-'*5}  {'-'*5}  {'-'*8}")
    total_warm = 0
    for label, warm, cold, critical in budget:
        crit = "  <<<" if critical else ""
        print(f"  {label:<42s} {warm:4d}s  {cold:4d}s  {crit}")
        total_warm += warm

    print(f"\n  Note: PaddleOCR and Embeddings run in PARALLEL threads.")
    print(f"  Wall time = max(PaddleOCR thread, Embed thread) + identification + context")
    print(f"  Warm wall time: ~{70+3:.0f}s  |  Cold wall time: ~{120+5:.0f}s")

    # --- Optimization opportunities ---
    print("\n" + "=" * 72)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 72)
    print("""
  1. ATTACK OCR SPEEDUP (highest impact -- would cut total time 50-80%)
     a. Crop to attack region before OCR (reduce image size 60-70%)
     b. PaddleOCR batch mode (if supported for variable-size crops)
     c. Lower OCR resolution for attack text (attacks are large text)
     d. Cache attack OCR results by image hash
     e. Skip attack OCR when name OCR + DINOv2 are high confidence

  2. PARALLEL ATTACK OCR
     a. Split cards across 2-3 PaddleOCR instances (separate processes)
     b. Risk: PaddleOCR is not thread-safe (SIGSEGV), needs multiprocessing

  3. LAZY ATTACK OCR
     a. Run name OCR + DINOv2 first (fast path: ~10s)
     b. Only run attack OCR for cards where DINOv2 confidence < threshold
     c. Page 1 needed attack fallback for only 1/9 cards
     d. Page 2 needed attack fallback for 0/8 cards

  4. MODEL PRELOADING (minor -- server mode already solves this)
     a. Pre-load DINOv2/PaddleOCR on server startup
     b. Saves ~30s on first page only
""")

    # --- Accuracy summary ---
    print("-" * 72)
    print("ACCURACY (from eval results)")
    print("-" * 72)
    pp = summary.get("per_page", {})
    for pk in sorted(pp):
        p = pp[pk]
        print(f"  {pk}: {p['exact_correct']}/{p['total']} = {p['exact_accuracy']*100:.0f}%")
    print(f"  Overall: {summary['exact_correct']}/{summary['scored_cards']} "
          f"= {summary['exact_accuracy']*100:.1f}%")


# ---------------------------------------------------------------------------
# Live profiling mode
# ---------------------------------------------------------------------------
def live_mode(requested_pages=None):
    """Run identify_page_v2 with timing instrumentation."""

    # Suppress noisy warnings
    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    logging.getLogger("paddle").setLevel(logging.ERROR)

    # Capture pipeline timing logs
    log_capture = StringIO()
    capture_handler = logging.StreamHandler(log_capture)
    capture_handler.setLevel(logging.DEBUG)
    capture_handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))

    class _TimingFilter(logging.Filter):
        _KW = ("done in", "identify_page_v2", "processing",
               "page context", "pass2", "pass3", "parallel identification",
               "pre-computation")
        def filter(self, record):
            msg = record.getMessage().lower()
            return any(kw in msg for kw in self._KW)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(logging.Formatter("  [LOG] %(message)s"))
    stderr_handler.addFilter(_TimingFilter())

    ml_logger = logging.getLogger("cardprice.ml")
    ml_logger.setLevel(logging.DEBUG)
    ml_logger.addHandler(capture_handler)
    ml_logger.addHandler(stderr_handler)

    # Load eval data
    with open(EVAL_PATH) as f:
        eval_data = json.load(f)

    page_segments = []
    for pi, page in enumerate(eval_data["pages"]):
        seg_dir = PROJECT_ROOT / page["segments_dir"]
        cards_with_id = [c for c in page["cards"] if c["card_id"]]
        paths = [str(seg_dir / c["segment"]) for c in cards_with_id]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"  Page {pi}: {len(missing)} MISSING segments -- skipping")
            continue
        page_segments.append((pi, paths, cards_with_id, page))
        print(f"  Page {pi}: {len(paths)} cards from {seg_dir}")

    if requested_pages is not None:
        page_segments = [(pi, p, c, m) for pi, p, c, m in page_segments
                         if pi in requested_pages]

    if not page_segments:
        print("ERROR: No valid pages found.")
        sys.exit(1)

    # Import (triggers model loading)
    print("\n" + "=" * 72)
    print("IMPORT / MODEL LOADING")
    print("=" * 72)
    t_import = time.perf_counter()
    from cardprice.ml import identify_page_v2, _scan_cache  # noqa: E402
    t_import = time.perf_counter() - t_import
    print(f"  Import time (incl. model loading): {t_import:.1f}s")

    # Run each page
    print("\n" + "=" * 72)
    print("PER-PAGE TIMING")
    print("=" * 72)

    page_timings = []
    all_results = []

    for pi, paths, cards, page_meta in page_segments:
        _scan_cache.clear()
        log_capture.truncate(0)
        log_capture.seek(0)

        print(f"\n--- Page {pi}: {len(paths)} cards ---")
        t0 = time.perf_counter()
        results = identify_page_v2(paths)
        elapsed = time.perf_counter() - t0

        log_output = log_capture.getvalue()
        t_paddle = _extract_time(log_output, r"PaddleOCR thread.*done in ([\d.]+)s")
        t_embed = _extract_time(log_output, r"embeddings thread done in ([\d.]+)s")
        t_dino_inner = _extract_time(log_output, r"DINOv2=([\d.]+)s")
        t_precomp = _extract_time(log_output, r"all pre-computation done in ([\d.]+)s")
        t_ident = _extract_time(log_output, r"parallel identification done in ([\d.]+)s")

        t_pass23 = None
        if t_precomp is not None and t_ident is not None:
            t_pass23 = max(0.0, elapsed - t_precomp - t_ident)

        timing = {
            "page": pi, "n_cards": len(paths), "total": elapsed,
            "per_card": elapsed / len(paths),
            "paddle_ocr": t_paddle, "embeddings": t_embed,
            "dino_inner": t_dino_inner, "precomp": t_precomp,
            "identification": t_ident, "pass2_3": t_pass23,
        }
        page_timings.append(timing)

        correct = sum(1 for c, r in zip(cards, results)
                      if r.get("card_id") == c["card_id"])
        accuracy = correct / len(cards) * 100

        print(f"  Total wall time:     {elapsed:6.1f}s  ({elapsed/len(paths):.1f}s/card)")
        print(f"  Pre-computation:     {_fmt(t_precomp)}")
        print(f"    PaddleOCR thread:  {_fmt(t_paddle)}")
        print(f"    Embeddings thread: {_fmt(t_embed)}  (DINOv2={_fmt(t_dino_inner)})")
        print(f"  Card identification: {_fmt(t_ident)}")
        if t_pass23 is not None:
            print(f"  Pass 2+3 (context):  {t_pass23:6.1f}s")
        print(f"  Accuracy: {correct}/{len(cards)} = {accuracy:.0f}%")

        all_results.append((pi, cards, results))
        sys.stdout.flush()
        gc.collect()

    # Aggregate summary
    print("\n" + "=" * 72)
    print("AGGREGATE SUMMARY")
    print("=" * 72)

    total_cards = sum(t["n_cards"] for t in page_timings)
    total_time = sum(t["total"] for t in page_timings)
    n_pages = len(page_timings)

    print(f"  Pages profiled:  {n_pages}")
    print(f"  Total cards:     {total_cards}")
    print(f"  Total wall time: {total_time:.1f}s")
    print(f"  Avg per card:    {total_time/total_cards:.1f}s")
    print()

    stages = [
        ("PaddleOCR thread", "paddle_ocr"),
        ("Embeddings thread", "embeddings"),
        ("  DINOv2 (within embed)", "dino_inner"),
        ("Pre-computation (wall)", "precomp"),
        ("Card identification", "identification"),
        ("Pass 2+3 (context)", "pass2_3"),
    ]

    avg_total = total_time / n_pages
    print(f"  {'Stage':<30s}  {'Avg/page':>8s}  {'Avg/card':>8s}  {'% wall':>8s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
    for label, key in stages:
        avg = _avg_stage(page_timings, key)
        if avg is not None:
            cards_pp = total_cards / n_pages
            print(f"  {label:<30s}  {avg:7.1f}s  {avg/cards_pp:7.1f}s  {avg/avg_total*100:7.0f}%")

    # Bottleneck
    print("\n" + "-" * 72)
    print("BOTTLENECK ANALYSIS")
    print("-" * 72)
    avg_paddle = _avg_stage(page_timings, "paddle_ocr")
    avg_embed = _avg_stage(page_timings, "embeddings")
    if avg_paddle is not None and avg_embed is not None:
        if avg_paddle > avg_embed:
            print(f"  Pre-computation bottleneck: PaddleOCR thread")
            print(f"    PaddleOCR:  {avg_paddle:.1f}s  |  Embeddings: {avg_embed:.1f}s")
            print(f"    Embeddings idles for {avg_paddle - avg_embed:.1f}s")
        else:
            print(f"  Pre-computation bottleneck: Embeddings thread")
            print(f"    Embeddings: {avg_embed:.1f}s  |  PaddleOCR: {avg_paddle:.1f}s")

    # Accuracy
    print("\n" + "-" * 72)
    print("ACCURACY")
    print("-" * 72)
    total_correct = 0
    total_eval = 0
    for pi, cards, results in all_results:
        correct = sum(1 for c, r in zip(cards, results)
                      if r.get("card_id") == c["card_id"])
        total_correct += correct
        total_eval += len(cards)
        print(f"  Page {pi}: {correct}/{len(cards)}")
    print(f"  Overall: {total_correct}/{total_eval} = "
          f"{total_correct/total_eval*100:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if "--report" in sys.argv:
        report_mode()
    else:
        pages = None
        args = [a for a in sys.argv[1:] if a != "--report"]
        if args:
            pages = set(int(a) for a in args)
        live_mode(requested_pages=pages)
