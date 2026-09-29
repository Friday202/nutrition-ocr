
import pandas as pd
import matplotlib.pyplot as plt
#from scipy.stats import pearsonr, spearmanr

# ============================================================
# Configuration
# ============================================================

csv_a = "per_sample_CER_donut.csv"
csv_b = "per_sample_CER_qwen.csv"

# ============================================================
# Read CSVs
# ============================================================

df_a = pd.read_csv(csv_a)
df_b = pd.read_csv(csv_b)

# First column should be file_path
# Second column should be CER
df_a = df_a.iloc[:, :2].copy()
df_b = df_b.iloc[:, :2].copy()

df_a.columns = ["file_path", "CER_A"]
df_b.columns = ["file_path", "CER_B"]

# clamp CER values to [0, 1]
#df_a["CER_A"] = df_a["CER_A"].clip(0, 1)
#df_b["CER_B"] = df_b["CER_B"].clip(0, 1)

# values smalelr than 0.01 are considered as 0 for plotting purposes
#df_a["CER_A"] = df_a["CER_A"].where(df_a["CER_A"] >= 0.01, 0)
#df_b["CER_B"] = df_b["CER_B"].where(df_b["CER_B"] >= 0.01, 0)

# mutiply by 100 to convert to percentage
df_a["CER_A"] = df_a["CER_A"] * 100
df_b["CER_B"] = df_b["CER_B"] * 100

# ============================================================
# Merge based on filename/path
# ============================================================

df = pd.merge(
    df_a,
    df_b,
    on="file_path",
    how="inner"
)

print(f"System A images: {len(df_a)}")
print(f"System B images: {len(df_b)}")
print(f"Matched images:  {len(df)}")

# Check for unmatched images
only_a = set(df_a["file_path"]) - set(df_b["file_path"])
only_b = set(df_b["file_path"]) - set(df_a["file_path"])

print(f"Only in System A: {len(only_a)}")
print(f"Only in System B: {len(only_b)}")

# Remove rows with missing CER values
df = df.dropna(subset=["CER_A", "CER_B"])

# ============================================================
# Statistics
# ============================================================

# pearson_r, pearson_p = pearsonr(df["CER_A"], df["CER_B"])
# spearman_r, spearman_p = spearmanr(df["CER_A"], df["CER_B"])

# Difference: negative means A is better, positive means B is better
df["CER_diff"] = df["CER_A"] - df["CER_B"]

a_better = (df["CER_A"] < df["CER_B"]).sum()
b_better = (df["CER_B"] < df["CER_A"]).sum()
same = (df["CER_A"] == df["CER_B"]).sum()

print("\n--- Results ---")
#print(f"Pearson correlation:  {pearson_r:.3f}")
#print(f"Spearman correlation: {spearman_r:.3f}")
print(f"A has lower CER:      {a_better} ({a_better / len(df) * 100:.1f}%)")
print(f"B has lower CER:      {b_better} ({b_better / len(df) * 100:.1f}%)")
print(f"Same CER:              {same} ({same / len(df) * 100:.1f}%)")

# ============================================================
# Scatter plot
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(
    df["CER_A"],
    df["CER_B"],
    alpha=0.5,
    s=30
)

ax.set_xscale("symlog", linthresh=0.01)
ax.set_yscale("symlog", linthresh=0.01)

# y = x diagonal
max_cer = max(df["CER_A"].max(), df["CER_B"].max())

ax.plot(
    [0, max_cer],
    [0, max_cer],
    linestyle="--",
    linewidth=1.5,
    label="Donut CER = Qwen CER"
)

ax.set_xlabel("Donut CER")
ax.set_ylabel("Qwen CER")
ax.set_title("Primerjava CER napake")

ax.set_xlim(left=0)
ax.set_ylim(bottom=0)



ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save
plt.savefig("ocr_cer_comparison.png", dpi=600)

plt.show()

