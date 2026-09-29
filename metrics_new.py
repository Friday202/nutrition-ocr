import json
import re
import sys
from pathlib import Path
from itertools import product as iproduct
from collections import defaultdict

try:
    from rapidfuzz import fuzz, process as fuzz_process
    from rapidfuzz.distance import Levenshtein as _rf_levenshtein
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False
    _rf_levenshtein = None
    print("[warning] rapidfuzz not installed — fuzzy recall disabled, using exact only", file=sys.stderr)

# ------------------------------------------------------------------ #
# Best-span CER ceiling
#
# R_fz (fuzzy token recall) checks whether GT *words* appear somewhere in
# the prediction blob, order- and position-agnostic. That's a much weaker
# condition than "a contiguous span of the prediction, if selected exactly,
# would give low CER against GT" — which is what an extractive
# (Sestavine:-anchored) correction model actually needs to achieve.
#
# This computes, for each example, the minimum character-level edit
# distance (normalized by len(GT)) over a set of candidate word-windows
# in the prediction text. That's the best-case CER ceiling: even a
# perfect span-selector can't beat this number. It makes the
# recoverability gap visible directly instead of only inferring it.
# ------------------------------------------------------------------ #

def _levenshtein_distance(a: str, b: str) -> int:
    """Character-level edit distance. Uses rapidfuzz's C implementation
    when available (fast — needed since this runs many times per example),
    falls back to a pure-Python DP otherwise (slow, fine for small runs)."""
    if HAS_FUZZY:
        return _rf_levenshtein.distance(a, b)

    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def normalize_for_cer(text: str) -> str:
    """Normalize cosmetic differences that shouldn't count as CER errors:
    casing, decimal-separator style, and whitespace around punctuation.
    Does NOT touch commas that separate ingredients (only commas sitting
    between two digits, e.g. '0,5' -> '0.5', are converted)."""
    text = text.lower().strip()
    text = re.sub(r'(?<=\d),(?=\d)', '.', text)      # decimal comma -> dot
    text = re.sub(r'\s*%\s*', '% ', text)             # unify spacing around %
    text = re.sub(r'\s*,\s*', ', ', text)             # unify spacing around commas
    text = re.sub(r'\s+', ' ', text)                  # collapse whitespace
    text = text.strip(" .")
    return text


def best_span_cer(pred_text: str, gt_text: str, window_margin: int = 4) -> dict:
    """
    Slide a word-level window over pred_text and find the span whose
    normalized-CER against gt_text is lowest. Window sizes tried range
    from (gt_word_count - window_margin) to (gt_word_count + window_margin),
    covering every start position in pred_text.

    Returns the best CER found, plus the winning span text, so you can
    spot-check whether "ceiling" cases are really unrecoverable or whether
    the window heuristic just missed a better split.
    """
    gt_norm = normalize_for_cer(gt_text)
    if not gt_norm:
        return {"best_cer": None, "best_span": None}

    pred_norm = normalize_for_cer(pred_text)
    pred_words = pred_norm.split(" ")
    gt_word_count = max(1, len(gt_norm.split(" ")))

    lo = max(1, gt_word_count - window_margin)
    hi = gt_word_count + window_margin

    best_cer = None
    best_span = None
    gt_len = max(1, len(gt_norm))

    for size in range(lo, hi + 1):
        if size > len(pred_words):
            continue
        for start in range(0, len(pred_words) - size + 1):
            span = " ".join(pred_words[start:start + size])
            dist = _levenshtein_distance(span, gt_norm)
            cer = dist / gt_len
            if best_cer is None or cer < best_cer:
                best_cer = cer
                best_span = span

    if best_cer is None:
        # pred_text shorter than any candidate window size
        dist = _levenshtein_distance(pred_norm, gt_norm)
        best_cer = dist / gt_len
        best_span = pred_norm

    return {"best_cer": best_cer, "best_span": best_span}

SLOVENIAN_STOPWORDS = {
    "in", "ali", "ter", "da", "se", "je", "so", "ga", "jo", "mu",
    "na", "za", "iz", "pri", "po", "ob", "pod", "nad", "med",
    "ki", "kar", "kot", "tudi", "ne", "le", "pa", "bi", "bo",
    "sta", "smo", "ste", "sem", "si", "ni", "ima", "imajo",
    "the", "and", "or", "of", "in", "to", "a",
}

def normalize_tokens(text: str, min_len: int = 3) -> set[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tokens = [
        t for t in tokens
        if len(t) >= min_len # and t not in SLOVENIAN_STOPWORDS and not t.isdigit()
    ]
    return set(tokens)

def compute_metrics(pred_tokens: set, gt_tokens: set, fuzzy_threshold: int = 85) -> dict:
    if not gt_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": 0.0,
                "exact_hits": 0, "fuzzy_hits": 0, "gt_size": 0, "pred_size": len(pred_tokens)}

    exact_intersection = pred_tokens & gt_tokens
    exact_hits = len(exact_intersection)

    fuzzy_hits = exact_hits
    fuzzy_matched_gt = set(exact_intersection)
    if HAS_FUZZY and pred_tokens:
        for gt_word in gt_tokens - exact_intersection:
            match = fuzz_process.extractOne(
                gt_word, pred_tokens,
                scorer=fuzz.ratio,
                score_cutoff=fuzzy_threshold
            )
            if match:
                fuzzy_hits += 1
                fuzzy_matched_gt.add(gt_word)

    recall_exact    = exact_hits / len(gt_tokens)
    recall_fuzzy    = fuzzy_hits / len(gt_tokens)
    precision_exact = exact_hits / len(pred_tokens) if pred_tokens else 0.0
    precision_fuzzy = fuzzy_hits / len(pred_tokens) if pred_tokens else 0.0

    def f_beta(p, r, beta=1):
        denom = beta**2 * p + r
        return (1 + beta**2) * p * r / denom if denom > 0 else 0.0

    return {
        "exact_hits":      exact_hits,
        "fuzzy_hits":      fuzzy_hits,
        "gt_size":         len(gt_tokens),
        "pred_size":       len(pred_tokens),
        "precision_exact": precision_exact,
        "recall_exact":    recall_exact,
        "f1_exact":        f_beta(precision_exact, recall_exact, beta=1),
        "f2_exact":        f_beta(precision_exact, recall_exact, beta=2),
        "precision_fuzzy": precision_fuzzy,
        "recall_fuzzy":    recall_fuzzy,
        "f1_fuzzy":        f_beta(precision_fuzzy, recall_fuzzy, beta=1),
        "f2_fuzzy":        f_beta(precision_fuzzy, recall_fuzzy, beta=2),
    }

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

import re

SESTAVINE_RE = re.compile(r"\bse\s*stavine\b\s*:?", re.IGNORECASE)

def evaluate_file(
    path: str,
    label: str,
    compute_span_cer: bool = True,
    span_cer_sample_limit: int | None = None,
    span_cer_window_margin: int = 4,
) -> dict:
    records = load_jsonl(path)
    agg = defaultdict(float)
    n = 0
    sestavine_matches = 0

    per_sample = []
    span_cer_values = []      # best_cer per example that got the span search
    span_cer_examples = []    # (image_path/idx, gt, pred_span, cer) for a few worst/best cases
    n_span_cer_computed = 0

    for idx, rec in enumerate(records):
        gt_text   = rec.get("ground_truth", "")
        if gt_text is None or gt_text.strip() == "":
            gt_text = rec.get("gt_ingredients", "")
        pred_text = rec.get("prediction", "")
        if pred_text is None or pred_text.strip() == "":
            pred_text = rec.get("ocr_text", "")
        if SESTAVINE_RE.search(pred_text):
            sestavine_matches += 1
        if not gt_text.strip():
            continue

        gt_tokens   = normalize_tokens(gt_text)
        if not gt_tokens:          # add this
            continue   
        pred_tokens = normalize_tokens(pred_text)
        m = compute_metrics(pred_tokens, gt_tokens)

        for k, v in m.items():
            agg[k] += v
        n += 1
        per_sample.append(m)

        if compute_span_cer and (span_cer_sample_limit is None or n_span_cer_computed < span_cer_sample_limit):
            span_result = best_span_cer(pred_text, gt_text, window_margin=span_cer_window_margin)
            if span_result["best_cer"] is not None:
                span_cer_values.append(span_result["best_cer"])
                span_cer_examples.append({
                    "image_path": rec.get("image_path", idx),
                    "ground_truth": gt_text,
                    "best_span": span_result["best_span"],
                    "best_cer": span_result["best_cer"],
                })
                n_span_cer_computed += 1

    if n == 0:
        return {"label": label, "n": 0}

    avg = {k: v / n for k, v in agg.items()}
    avg["label"] = label
    avg["n"] = n
    avg["sestavine_matches"] = sestavine_matches
    avg["perfect_recall_exact"] = sum(1 for m in per_sample if m["recall_exact"] == 1.0) / n
    avg["perfect_recall_fuzzy"] = sum(1 for m in per_sample if m["recall_fuzzy"] == 1.0) / n

    if span_cer_values:
        n_span = len(span_cer_values)
        avg["span_cer_n"] = n_span
        avg["span_cer_mean"] = sum(span_cer_values) / n_span
        avg["span_cer_below_2pct"]  = sum(1 for c in span_cer_values if c < 0.02) / n_span
        avg["span_cer_below_6pct"]  = sum(1 for c in span_cer_values if c < 0.06) / n_span
        avg["span_cer_below_10pct"] = sum(1 for c in span_cer_values if c < 0.10) / n_span
        # keep the worst few for manual spot-checking — is the ceiling real
        # (info genuinely absent/garbled) or an artifact of the window search?
        worst = sorted(span_cer_examples, key=lambda e: e["best_cer"], reverse=True)[:10]
        avg["span_cer_worst_examples"] = worst

    return avg

def format_row(r: dict, key_metric: str = "f2_fuzzy") -> str:
    label = r["label"]
    n     = r["n"]
    p_e   = r.get("precision_exact", 0) * 100
    rc_e  = r.get("recall_exact",    0) * 100
    f2_e  = r.get("f2_exact",        0) * 100
    p_f   = r.get("precision_fuzzy", 0) * 100
    rc_f  = r.get("recall_fuzzy",    0) * 100
    f2_f  = r.get("f2_fuzzy",        0) * 100
    hits  = r.get("exact_hits",      0)
    pr_e  = r.get("perfect_recall_exact", 0) * 100
    pr_f  = r.get("perfect_recall_fuzzy", 0) * 100

    return (
        f"{label:<28} | n={n:>5} | "
        f"P_ex={p_e:5.1f}% R_ex={rc_e:5.1f}% F2_ex={f2_e:5.1f}% | "
        f"P_fz={p_f:5.1f}% R_fz={rc_f:5.1f}% F2_fz={f2_f:5.1f}% | "
        f"avg_hits={hits:.1f} | perfect_ex={pr_e:4.1f}% perfect_fz={pr_f:4.1f}%"
    )

def format_span_cer_row(r: dict) -> str:
    if "span_cer_n" not in r:
        return f"{r['label']:<28} | span-CER not computed"
    label = r["label"]
    n     = r["span_cer_n"]
    mean  = r["span_cer_mean"] * 100
    b2    = r["span_cer_below_2pct"] * 100
    b6    = r["span_cer_below_6pct"] * 100
    b10   = r["span_cer_below_10pct"] * 100
    return (
        f"{label:<28} | n={n:>5} | "
        f"mean_span_CER={mean:5.1f}% | "
        f"<2%={b2:5.1f}% <6%={b6:5.1f}% <10%={b10:5.1f}%"
    )


def save_json_results(results: list[dict], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    files = [        
        #("paddle_ocr_best_test_qwen_7B_A.jsonl", "A"),
        #("paddle_ocr_best_test_qwen_7B_B.jsonl", "B"),
        #("paddle_ocr_best_test_qwen_1.5B_D.jsonl", "1.5B_D"),
        #("paddle_ocr_best_test_qwen_3B_D.jsonl", "3B_D"),
        #("paddle_ocr_best_test_qwen_7B_D.jsonl", "7B_D"),
        #("paddle_ocr_best_test_qwen_14B_D.jsonl", "14B_D"),
        #("paddle_ocr_best_test_qwen_32B_D.jsonl", "32B_D"),
        #("qwen_results/paddle_ocr_best_test_qwen_7B_A.jsonl", "7B_A"),
        #("qwen_results/paddle_ocr_best_test_qwen_7B_B.jsonl", "7B_B"),
        #("qwen_results/paddle_ocr_best_test_qwen_7B_C.jsonl", "7B_C"),
        #("qwen_results/paddle_ocr_best_test_qwen_7B_D.jsonl", "7B_D"),
        #("qwen_results/paddle_ocr_best_test_qwen_7B_E.jsonl", "7B_E"),
        #("qwen_results/paddle_ocr_best_test_qwen_7B_F.jsonl", "7B_F"),
        #("qwen_results/paddle_ocr_best_test_qwen_7B_G.jsonl", "7B_G"),
        ("paddle_ocr_best_train.jsonl", "train"),

        #("paddleocr_results/paddle_ocr_results_1600_0.3_0.5_0.3_T.jsonl", "T"),
        #("paddleocr_results/paddle_ocr_results_1600_0.3_0.5_0.3_F.jsonl", "F"),                     
    ]

    # Span-CER search is O(window_range * len(pred_words)) Levenshtein calls
    # PER EXAMPLE, so it's much slower than the token-recall metrics above.
    # With rapidfuzz installed it's still fast per-call, but on 22k examples
    # you may want to cap it first to sanity-check before running the full set.
    # Set to None to run on every example.
    SPAN_CER_SAMPLE_LIMIT = None
    SPAN_CER_WINDOW_MARGIN = 4

    results = []
    print(f"\n{'='*110}")
    print(f"{'Config':<28} | {'N':>6} | {'--- Exact ---':^35} | {'--- Fuzzy (≥85%) ---':^35} | avg_exact_hits")
    print(f"{'='*110}")

    for fpath, label in files:
        p = Path(fpath)
        if not p.exists():
            print(f"[skip] {fpath} not found")
            continue
        r = evaluate_file(
            fpath, label,
            compute_span_cer=True,
            span_cer_sample_limit=SPAN_CER_SAMPLE_LIMIT,
            span_cer_window_margin=SPAN_CER_WINDOW_MARGIN,
        )
        results.append(r)
        print(format_row(r))

    if results:
        print(f"{'='*110}")
        best_f2   = max(results, key=lambda x: x.get("f2_fuzzy", 0))
        best_rec  = max(results, key=lambda x: x.get("recall_fuzzy", 0))
        print(f"\n  Best F2 (fuzzy):     {best_f2['label']}  →  F2={best_f2.get('f2_fuzzy',0)*100:.1f}%")
        print(f"  Best Recall (fuzzy): {best_rec['label']}  →  Recall={best_rec.get('recall_fuzzy',0)*100:.1f}%")

        print(f"\n{'='*110}")
        print("  Best-span CER ceiling — best-case CER if the model located the ideal contiguous")
        print(f"  span perfectly (window sizes: GT_word_count ± {SPAN_CER_WINDOW_MARGIN})")
        if SPAN_CER_SAMPLE_LIMIT is not None:
            print(f"  [capped at {SPAN_CER_SAMPLE_LIMIT} examples per file — set SPAN_CER_SAMPLE_LIMIT=None for full run]")
        print(f"{'='*110}")
        for r in results:
            print(format_span_cer_row(r))

        # Print a couple of worst-case examples per file so you can eyeball
        # whether the ceiling reflects genuinely missing/garbled info, or the
        # window search just missed a better split (e.g. margin too tight).
        for r in results:
            worst = r.get("span_cer_worst_examples")
            if not worst:
                continue
            print(f"\n  Worst span-CER cases for '{r['label']}' (spot-check these):")
            for ex in worst[:3]:
                print(f"    CER={ex['best_cer']*100:5.1f}% | GT: {ex['ground_truth']!r}")
                print(f"                  best_span: {ex['best_span']!r}")

        save_json_results(results, "ocr_eval_results.json")
        print(f"\n  Full results saved to: ocr_eval_results.json")
    print()