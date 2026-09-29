import json
import re
import sys
from pathlib import Path
from itertools import product as iproduct
from collections import defaultdict

try:
    from rapidfuzz import fuzz, process as fuzz_process
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False
    print("[warning] rapidfuzz not installed — fuzzy recall disabled, using exact only", file=sys.stderr)

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

def evaluate_file(path: str, label: str) -> dict:
    records = load_jsonl(path)
    agg = defaultdict(float)
    n = 0
    sestavine_matches = 0

    perfect_recall_image_paths = []  # NEW

    per_sample = []
    for rec in records:
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

        image_path = rec.get("image_path")        

        gt_tokens   = normalize_tokens(gt_text)
        if not gt_tokens:          # add this
            continue   
        pred_tokens = normalize_tokens(pred_text)
        m = compute_metrics(pred_tokens, gt_tokens)

        for k, v in m.items():
            agg[k] += v
        n += 1
        per_sample.append(m)

        if m["recall_exact"] == 1.0:
            perfect_recall_image_paths.append(image_path)

    if n == 0:
        return {"label": label, "n": 0}

    avg = {k: v / n for k, v in agg.items()}
    avg["label"] = label
    avg["n"] = n
    avg["sestavine_matches"] = sestavine_matches
    avg["perfect_recall_exact"] = sum(1 for m in per_sample if m["recall_exact"] == 1.0) / n
    avg["perfect_recall_fuzzy"] = sum(1 for m in per_sample if m["recall_fuzzy"] == 1.0) / n
    avg["perfect_recall_image_paths"] = perfect_recall_image_paths  
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

def save_json_results(results: list[dict], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


import csv

def save_perfect_recall_csv(image_paths: list[str], jsonl_path: str):
    """
    Writes image_paths of samples reaching perfect recall to a CSV,
    named after the source jsonl file, e.g.
    internal_test.jsonl -> internal_test_perfect_recall_image_paths.csv
    """
    stem = Path(jsonl_path).stem
    out_path = f"{stem}_perfect_recall_image_paths.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path"])
        for img_path in image_paths:
            writer.writerow([img_path])
    print(f"  Saved {len(image_paths)} perfect-recall image paths to: {out_path}")
    return out_path

if __name__ == "__main__":
    files = [        
        #("paddle_ocr_best_test_qwen_7B_A.jsonl", "A"),
        #("paddle_ocr_best_test_qwen_7B_B.jsonl", "B"),
        #("paddle_ocr_best_test_qwen_1.5B_D.jsonl", "1.5B_D"),
        #("paddle_ocr_best_test_qwen_3B_D.jsonl", "3B_D"),
        #("paddle_ocr_best_test_qwen_7B_D.jsonl", "7B_D"),
        #("paddle_ocr_best_test_qwen_14B_D.jsonl", "14B_D"),
        #("paddle_ocr_best_test_qwen_32B_D.jsonl", "32B_D"),

        #("qwen_best_prompt/paddle_ocr_best_test_qwen_1.5B_G.jsonl", "1.5B_G"),
        #("qwen_best_prompt/paddle_ocr_best_test_qwen_3B_G.jsonl", "3B_G"),
        #("qwen_best_prompt/paddle_ocr_best_test_qwen_7B_G.jsonl", "7B_G"),
        #("qwen_best_prompt/paddle_ocr_best_test_qwen_14B_G.jsonl", "14B_G"),
        #("qwen_best_prompt/paddle_ocr_best_test_qwen_32B_G.jsonl", "32B_G")

        #("qwen_results/paddle_ocr_best_test_qwen_7B_A.jsonl", "7B_A"),
        #("qwen_results/paddle_ocr_best_test_qwen_7B_B.jsonl", "7B_B"),
        #("qwen_results/paddle_ocr_best_test_qwen_7B_C.jsonl", "7B_C"),
        #("qwen_results/paddle_ocr_best_test_qwen_7B_D.jsonl", "7B_D"),
        #("qwen_results/paddle_ocr_best_test_qwen_7B_E.jsonl", "7B_E"),
        #("qwen_results/paddle_ocr_best_test_qwen_7B_F.jsonl", "7B_F"),
        #("qwen_results/paddle_ocr_best_test_qwen_7B_G.jsonl", "7B_G"),

        ("paddle_ocr_best_test.jsonl", "test"),
        #("paddle_ocr_best_train.jsonl", "train"),
        #("internal_test.jsonl", "internal_test"),

        #("paddleocr_results/paddle_ocr_results_1600_0.3_0.5_0.3_T.jsonl", "T"),
        #("paddleocr_results/paddle_ocr_results_1600_0.3_0.5_0.3_F.jsonl", "F"),                     
    ]

    
    results = []
    print(f"\n{'='*110}")
    print(f"{'Config':<28} | {'N':>6} | {'--- Exact ---':^35} | {'--- Fuzzy (≥85%) ---':^35} | avg_exact_hits")
    print(f"{'='*110}")

    for fpath, label in files:
        p = Path(fpath)
        if not p.exists():
            print(f"[skip] {fpath} not found")
            continue
        r = evaluate_file(fpath, label)
        results.append(r)
        print(format_row(r))
        img_paths = r.get("perfect_recall_image_paths", [])
        #img_paths = None
        if img_paths:
            save_perfect_recall_csv(img_paths, fpath)            
        else:
            print(f"  No perfect-recall samples for {label}")

    if results:
        print(f"{'='*110}")
        best_f2   = max(results, key=lambda x: x.get("f2_fuzzy", 0))
        best_rec  = max(results, key=lambda x: x.get("recall_fuzzy", 0))
        print(f"\n  Best F2 (fuzzy):     {best_f2['label']}  →  F2={best_f2.get('f2_fuzzy',0)*100:.1f}%")
        print(f"  Best Recall (fuzzy): {best_rec['label']}  →  Recall={best_rec.get('recall_fuzzy',0)*100:.1f}%")

        save_json_results(results, "ocr_eval_results.json")
        print(f"\n  Full results saved to: ocr_eval_results.json")
    print()