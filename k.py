"""
OCR evaluation script: compares Qwen and Donut predictions against ground truth.

Computes CER (character error rate), flags likely ground-truth labeling errors,
and produces per-category filename lists (txt + csv) plus a summary report.

------------------------------------------------------------------------------
INPUT ASSUMPTIONS:

  Single CSV (merged_predictions.csv) with columns:
    file_name, ground_truth, prediction_qwen, prediction_donut
------------------------------------------------------------------------------
"""

import os
import csv
import math
from collections import Counter

import pandas as pd
import numpy as np

# ============================== CONFIG =======================================



input_csv =  "merged_predictions.csv"

out_dir =  "eval_results\\"

# Column names in the input CSV. Adjust if yours differ.
COL_FILE = "file_name"
COL_GT = "ground_truth"
COL_QWEN_PRED = "prediction_qwen"
COL_DONUT_PRED = "prediction_donut"

# Threshold used throughout for "near match" / "low CER" comparisons
CER_LOW_THRESH = 0.02   # 2%
CER_HIGH_THRESH = 0.90  # 90% ("both fail" threshold)

N_HIST_BINS = 10  # for the 0-100% CER histograms in the report

# ==============================================================================

os.makedirs(out_dir, exist_ok=True)

# --- Try to use a fast Levenshtein implementation, fall back to pure python ---
try:
    import Levenshtein as _lev

    def edit_distance(a, b):
        return _lev.distance(a, b)
except ImportError:
    try:
        from rapidfuzz.distance import Levenshtein as _rf_lev

        def edit_distance(a, b):
            return _rf_lev.distance(a, b)
    except ImportError:
        def edit_distance(a, b):
            """Plain DP Levenshtein distance (character-level), O(len(a)*len(b))."""
            if a == b:
                return 0
            la, lb = len(a), len(b)
            if la == 0:
                return lb
            if lb == 0:
                return la
            prev = list(range(lb + 1))
            for i, ca in enumerate(a, 1):
                curr = [i] + [0] * lb
                for j, cb in enumerate(b, 1):
                    cost = 0 if ca == cb else 1
                    curr[j] = min(
                        prev[j] + 1,       # deletion
                        curr[j - 1] + 1,   # insertion
                        prev[j - 1] + cost # substitution
                    )
                prev = curr
            return prev[lb]


def safe_str(x):
    """Normalize NaN/None to empty string, everything else to str."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def cer(ref, hyp):
    """
    Character Error Rate = edit_distance(ref, hyp) / len(ref).
    Returns (distance, ref_len, cer_value).
    If ref_len == 0: cer_value is 0.0 if hyp is also empty, else 1.0 (treated
    as fully wrong), distance = len(hyp).
    """
    ref = safe_str(ref)
    hyp = safe_str(hyp)
    d = edit_distance(ref, hyp)
    rl = len(ref)
    if rl == 0:
        val = 0.0 if len(hyp) == 0 else 1.0
        return d, rl, val
    return d, rl, d / rl


# ============================== LOAD DATA ====================================

print(f"Loading {input_csv}")
df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)

df = df.rename(columns={
    COL_FILE: "file_name",
    COL_GT: "ground_truth",
    COL_QWEN_PRED: "prediction_qwen",
    COL_DONUT_PRED: "prediction_donut",
})
df = df[["file_name", "ground_truth", "prediction_qwen", "prediction_donut"]]

for c in ["ground_truth", "prediction_qwen", "prediction_donut"]:
    df[c] = df[c].map(safe_str)

print(f"Loaded {len(df)} rows.")

# ============================== COMPUTE CER ==================================

qwen_res = df.apply(lambda r: cer(r["ground_truth"], r["prediction_qwen"]), axis=1)
donut_res = df.apply(lambda r: cer(r["ground_truth"], r["prediction_donut"]), axis=1)
qd_res = df.apply(lambda r: cer(r["prediction_donut"], r["prediction_qwen"]), axis=1)  # donut treated as reference

df["edits_qwen"], df["gtlen_qwen"], df["cer_qwen"] = zip(*qwen_res)
df["edits_donut"], df["gtlen_donut"], df["cer_donut"] = zip(*donut_res)
df["edits_qd"], df["reflen_qd"], df["cer_qwen_donut"] = zip(*qd_res)

df["exact_qwen"] = df["prediction_qwen"] == df["ground_truth"]
df["exact_donut"] = df["prediction_donut"] == df["ground_truth"]
df["exact_qd"] = df["prediction_qwen"] == df["prediction_donut"]

df_out_path = out_dir + "cer_dataframe.csv"
df.to_csv(df_out_path, index=False)
print(f"Full per-row CER dataframe written to: {df_out_path}")

# ============================== HELPER: write pair of files ==================

def write_category(name, sub_df, description, sort_col=None):
    """
    Writes:
      out_dir/name.csv -> file_name column only
      out_dir/name.txt -> human readable listing

    If sort_col is given (a column name, or a Series aligned to sub_df's index),
    rows are ordered with the highest value first (highest error at the top).
    """
    if sort_col is not None:
        if isinstance(sort_col, str):
            sub_df = sub_df.sort_values(sort_col, ascending=False)
        else:
            sub_df = sub_df.loc[sort_col.sort_values(ascending=False).index]

    csv_path = out_dir + name + ".csv"
    txt_path = out_dir + name + ".txt"

    sub_df[["file_name"]].to_csv(csv_path, index=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{name}\n")
        f.write(f"{description}\n")
        f.write(f"Count: {len(sub_df)}\n")
        f.write("=" * 100 + "\n\n")
        for _, r in sub_df.iterrows():
            f.write(f"file_name: {r['file_name']}\n")
            f.write(f"  ground_truth     : {r['ground_truth']}\n")
            f.write(f"  prediction_qwen  : {r['prediction_qwen']}\n")
            f.write(f"  prediction_donut : {r['prediction_donut']}\n")
            f.write(f"  cer_qwen_vs_gt   : {r['cer_qwen']:.4f}\n")
            f.write(f"  cer_donut_vs_gt  : {r['cer_donut']:.4f}\n")
            f.write(f"  cer_qwen_vs_donut: {r['cer_qwen_donut']:.4f}\n")
            f.write("-" * 100 + "\n")

    print(f"[{name}] {len(sub_df)} rows -> {csv_path} , {txt_path}")
    return sub_df


# ============================== CATEGORY FILTERS =============================

# a) qwen == donut, but both differ from GT (likely GT labeling error)
mask_a = df["exact_qd"] & (~df["exact_qwen"])  # exact_qd + not matching GT implies donut also != GT
cat_a = write_category(
    "a_qwen_donut_agree_gt_differs",
    df[mask_a],
    "prediction_qwen == prediction_donut, but ground_truth differs from both (likely GT labeling error).",
    sort_col="cer_qwen",  # qwen==donut here, so cer_qwen and cer_donut vs GT are equal
)

# b) 0 < CER(donut, qwen) < 0.02, and GT matches neither exactly (relaxed GT error check)
mask_b = (df["cer_qwen_donut"] > 0) & (df["cer_qwen_donut"] < CER_LOW_THRESH) & \
         (~df["exact_qwen"]) & (~df["exact_donut"])
cat_b = write_category(
    "b_qwen_donut_near_agree_gt_differs",
    df[mask_b],
    f"0 < CER(donut,qwen) < {CER_LOW_THRESH:.0%}, and ground_truth doesn't exactly match either prediction "
    f"(relaxed likely GT labeling error).",
    sort_col=df.loc[mask_b, ["cer_qwen", "cer_donut"]].mean(axis=1),
)

# c) qwen exact match to GT, donut not
mask_c = df["exact_qwen"] & (~df["exact_donut"])
cat_c = write_category(
    "c_qwen_exact_donut_fails",
    df[mask_c],
    "prediction_qwen is an exact match to ground_truth, prediction_donut is not.",
    sort_col="cer_donut",  # donut is the one failing here
)

# d) donut exact match to GT, qwen not
mask_d = df["exact_donut"] & (~df["exact_qwen"])
cat_d = write_category(
    "d_donut_exact_qwen_fails",
    df[mask_d],
    "prediction_donut is an exact match to ground_truth, prediction_qwen is not.",
    sort_col="cer_qwen",  # qwen is the one failing here
)

# e) qwen CER < 2% vs GT, donut is not < 2%
mask_e = (df["cer_qwen"] < CER_LOW_THRESH) & (~(df["cer_donut"] < CER_LOW_THRESH))
cat_e = write_category(
    "e_qwen_low_cer_donut_not",
    df[mask_e],
    f"CER(qwen, GT) < {CER_LOW_THRESH:.0%}, but CER(donut, GT) is not < {CER_LOW_THRESH:.0%}.",
    sort_col="cer_donut",
)

# f) donut CER < 2% vs GT, qwen is not < 2%
mask_f = (df["cer_donut"] < CER_LOW_THRESH) & (~(df["cer_qwen"] < CER_LOW_THRESH))
cat_f = write_category(
    "f_donut_low_cer_qwen_not",
    df[mask_f],
    f"CER(donut, GT) < {CER_LOW_THRESH:.0%}, but CER(qwen, GT) is not < {CER_LOW_THRESH:.0%}.",
    sort_col="cer_qwen",
)

# g) both donut and qwen have CER > 90% vs GT (both fail hard)
mask_g = (df["cer_donut"] > CER_HIGH_THRESH) & (df["cer_qwen"] > CER_HIGH_THRESH)
cat_g = write_category(
    "g_both_fail_high_cer",
    df[mask_g],
    f"Both CER(donut, GT) and CER(qwen, GT) are > {CER_HIGH_THRESH:.0%} (both models fail badly).",
    sort_col=df.loc[mask_g, ["cer_qwen", "cer_donut"]].mean(axis=1),
)

# h) hardest cases: both fail vs GT hard AND also disagree with each other hard
#    (as opposed to category g, which includes cases where both are wrong but
#    happen to agree with each other)
mask_h = mask_g & (df["cer_qwen_donut"] > CER_HIGH_THRESH)
cat_h = write_category(
    "h_hardest_high_cer_and_high_intercer",
    df[mask_h],
    f"Both CER(donut,GT) and CER(qwen,GT) > {CER_HIGH_THRESH:.0%}, AND CER(qwen,donut) also > "
    f"{CER_HIGH_THRESH:.0%} — the two models don't even agree with each other. These are the "
    f"hardest/most ambiguous images (as opposed to category g, where both models could still be "
    f"failing in a similar/correlated way).",
    sort_col=df.loc[mask_h, ["cer_qwen", "cer_donut", "cer_qwen_donut"]].mean(axis=1),
)

# i) qwen CER > 90% vs GT, regardless of donut (shows how donut does on qwen's hard failures)
mask_i = df["cer_qwen"] > CER_HIGH_THRESH
cat_i = write_category(
    "i_qwen_high_cer_show_donut",
    df[mask_i],
    f"CER(qwen, GT) > {CER_HIGH_THRESH:.0%} — qwen's worst failures. Donut's prediction/CER is shown "
    f"alongside for comparison on the same images.",
    sort_col="cer_qwen",
)

# j) donut CER > 90% vs GT, regardless of qwen (shows how qwen does on donut's hard failures)
mask_j = df["cer_donut"] > CER_HIGH_THRESH
cat_j = write_category(
    "j_donut_high_cer_show_qwen",
    df[mask_j],
    f"CER(donut, GT) > {CER_HIGH_THRESH:.0%} — donut's worst failures. Qwen's prediction/CER is shown "
    f"alongside for comparison on the same images.",
    sort_col="cer_donut",
)

# ============================== SUMMARY REPORT ===============================

n = len(df)

exact_qwen_n = int(df["exact_qwen"].sum())
exact_donut_n = int(df["exact_donut"].sum())

# Histograms: 10 equal bins across 0-100% CER, plus an overflow bin for >100%
bin_edges = np.linspace(0, 1, N_HIST_BINS + 1)  # 0,0.1,...,1.0


def cer_histogram(cer_series):
    capped = cer_series.clip(upper=1.0)  # values >100% go in the last bin; report overflow separately
    overflow = int((cer_series > 1.0).sum())
    counts, _ = np.histogram(capped, bins=bin_edges)
    lines = []
    for i in range(N_HIST_BINS):
        lo, hi = bin_edges[i] * 100, bin_edges[i + 1] * 100
        lines.append(f"    CER in [{lo:.0f}%, {hi:.0f}%{']' if i == N_HIST_BINS - 1 else ')'}: {counts[i]}")
    lines.append(f"    CER > 100% (overflow, insertions exceeded ref length): {overflow}")
    return "\n".join(lines), counts, overflow


donut_hist_str, donut_hist_counts, donut_overflow = cer_histogram(df["cer_donut"])
qwen_hist_str, qwen_hist_counts, qwen_overflow = cer_histogram(df["cer_qwen"])

# Micro-averaged CER = total edits / total GT chars (NOT mean of per-row CER)
total_gt_chars = df["gtlen_qwen"].sum()  # same as gtlen_donut, both computed vs same ground_truth
avg_cer_donut_micro = df["edits_donut"].sum() / total_gt_chars if total_gt_chars else float("nan")
avg_cer_qwen_micro = df["edits_qwen"].sum() / total_gt_chars if total_gt_chars else float("nan")

# Simple (macro) mean CER per row, for comparison/interest
avg_cer_donut_macro = df["cer_donut"].mean()
avg_cer_qwen_macro = df["cer_qwen"].mean()
median_cer_donut = df["cer_donut"].median()
median_cer_qwen = df["cer_qwen"].median()

# Low-CER (<2%) overlap between donut and qwen
low_donut = df["cer_donut"] < CER_LOW_THRESH
low_qwen = df["cer_qwen"] < CER_LOW_THRESH
low_both = (low_donut & low_qwen).sum()
low_union = (low_donut | low_qwen).sum()
low_donut_only = (low_donut & ~low_qwen).sum()
low_qwen_only = (low_qwen & ~low_donut).sum()

# Exact-match overlap
exact_both = (df["exact_donut"] & df["exact_qwen"]).sum()
exact_union = (df["exact_donut"] | df["exact_qwen"]).sum()
exact_donut_only = (df["exact_donut"] & ~df["exact_qwen"]).sum()
exact_qwen_only = (df["exact_qwen"] & ~df["exact_donut"]).sum()

# Exact-match union, also crediting category (a) files (qwen==donut, GT likely wrong) as "covered"
covered_files = set(df.loc[df["exact_donut"] | df["exact_qwen"], "file_name"])
covered_with_a = covered_files | set(cat_a["file_name"])
exact_union_incl_a = len(covered_with_a)

# High-CER (>90%) both-fail overlap, for interest
high_both = int(mask_g.sum())

# Additional interesting stats
pred_len_ratio_qwen = (df["prediction_qwen"].str.len() / df["gtlen_qwen"].replace(0, np.nan)).mean()
pred_len_ratio_donut = (df["prediction_donut"].str.len() / df["gtlen_donut"].replace(0, np.nan)).mean()
n_gt_empty = int((df["gtlen_qwen"] == 0).sum())
n_qd_identical_and_gt_correct = int(((df["exact_qd"]) & (df["exact_qwen"])).sum())  # all 3 agree

report_path = out_dir + "summary_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("OCR EVALUATION SUMMARY REPORT\n")
    f.write("=" * 100 + "\n")
    f.write(f"Total rows evaluated: {n}\n\n")

    f.write("--- EXACT MATCH COUNTS ---\n")
    f.write(f"Exact matches, donut vs GT : {exact_donut_n} ({exact_donut_n / n:.2%})\n")
    f.write(f"Exact matches, qwen vs GT  : {exact_qwen_n} ({exact_qwen_n / n:.2%})\n\n")

    f.write(f"--- CER HISTOGRAM vs GT, donut (nbins={N_HIST_BINS}, 0-100%) ---\n")
    f.write(donut_hist_str + "\n\n")

    f.write(f"--- CER HISTOGRAM vs GT, qwen (nbins={N_HIST_BINS}, 0-100%) ---\n")
    f.write(qwen_hist_str + "\n\n")

    f.write("--- MICRO-AVERAGED CER (sum of edits / sum of GT chars) ---\n")
    f.write("(This is the corpus-level CER, NOT the mean of per-row CER values.)\n")
    f.write(f"Donut micro-avg CER: {avg_cer_donut_micro:.4%}\n")
    f.write(f"Qwen  micro-avg CER: {avg_cer_qwen_micro:.4%}\n\n")

    f.write("--- (for reference) simple per-row mean/median CER ---\n")
    f.write(f"Donut mean CER (macro avg): {avg_cer_donut_macro:.4%}   median: {median_cer_donut:.4%}\n")
    f.write(f"Qwen  mean CER (macro avg): {avg_cer_qwen_macro:.4%}   median: {median_cer_qwen:.4%}\n\n")

    f.write(f"--- LOW-CER (<{CER_LOW_THRESH:.0%}) OVERLAP, donut vs qwen (each vs GT) ---\n")
    f.write(f"Both donut AND qwen < {CER_LOW_THRESH:.0%}     : {low_both}\n")
    f.write(f"Donut only < {CER_LOW_THRESH:.0%}               : {low_donut_only}\n")
    f.write(f"Qwen only < {CER_LOW_THRESH:.0%}                : {low_qwen_only}\n")
    f.write(f"Union (either < {CER_LOW_THRESH:.0%})           : {low_union}  (unique files covered)\n\n")

    f.write("--- EXACT MATCH OVERLAP, donut vs qwen ---\n")
    f.write(f"Both donut AND qwen exact match GT : {exact_both}\n")
    f.write(f"Donut only exact match              : {exact_donut_only}\n")
    f.write(f"Qwen only exact match                : {exact_qwen_only}\n")
    f.write(f"Union (either exact match)          : {exact_union}  (unique files covered)\n")
    f.write(f"Union INCLUDING category (a) files\n")
    f.write(f"  (qwen==donut but GT differs, likely GT error, counted as 'covered'): {exact_union_incl_a}\n\n")

    f.write("--- CATEGORY COUNTS (see corresponding .txt/.csv files) ---\n")
    f.write(f"a) qwen==donut, GT differs (likely GT error)        : {len(cat_a)}\n")
    f.write(f"b) 0<CER(donut,qwen)<{CER_LOW_THRESH:.0%}, GT matches neither (relaxed GT error) : {len(cat_b)}\n")
    f.write(f"c) qwen exact, donut fails                          : {len(cat_c)}\n")
    f.write(f"d) donut exact, qwen fails                          : {len(cat_d)}\n")
    f.write(f"e) qwen CER<{CER_LOW_THRESH:.0%}, donut not                        : {len(cat_e)}\n")
    f.write(f"f) donut CER<{CER_LOW_THRESH:.0%}, qwen not                        : {len(cat_f)}\n")
    f.write(f"g) both CER>{CER_HIGH_THRESH:.0%} (both fail hard)                 : {len(cat_g)}\n")
    f.write(f"h) g) + also CER(qwen,donut)>{CER_HIGH_THRESH:.0%} (hardest, models don't even agree) : {len(cat_h)}\n")
    f.write(f"i) qwen CER>{CER_HIGH_THRESH:.0%} vs GT (qwen's worst failures, donut shown too)      : {len(cat_i)}\n")
    f.write(f"j) donut CER>{CER_HIGH_THRESH:.0%} vs GT (donut's worst failures, qwen shown too)     : {len(cat_j)}\n\n")

    f.write("--- ADDITIONAL / INTERESTING STATS ---\n")
    f.write(f"Rows where GT is an empty string                    : {n_gt_empty}\n")
    f.write(f"Rows where qwen==donut AND both exactly match GT    : {n_qd_identical_and_gt_correct}\n")
    f.write(f"Rows where both models disagree with GT AND with each other\n")
    f.write(f"  (hardest / most ambiguous cases, excluded from all a/b/g buckets): "
            f"{n - len(cat_a) - len(cat_b) - int(((~df['exact_qwen']) & (~df['exact_donut']) & (df['cer_qwen_donut']>=CER_LOW_THRESH) & ~mask_g).sum()) - len(cat_g)}\n")
    f.write(f"Mean ratio of prediction length to GT length, qwen  : {pred_len_ratio_qwen:.3f}\n")
    f.write(f"Mean ratio of prediction length to GT length, donut : {pred_len_ratio_donut:.3f}\n")
    f.write(f"Donut CER>100% overflow count (over-generation)     : {donut_overflow}\n")
    f.write(f"Qwen  CER>100% overflow count (over-generation)     : {qwen_overflow}\n")

print(f"\nSummary report written to: {report_path}")
print("Done.")
