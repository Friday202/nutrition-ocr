
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Read CSVs
# ============================================================

csv_a = "per_sample_CER_donut.csv"
csv_b = "per_sample_CER_qwen.csv"

df_a = pd.read_csv(csv_a)
df_b = pd.read_csv(csv_b)

# First column = file_path
# Second column = CER
df_a = df_a.iloc[:, :2].copy()
df_b = df_b.iloc[:, :2].copy()

df_a.columns = ["file_path", "CER_A"]
df_b.columns = ["file_path", "CER_B"]

# clamp CER values to [0, 1]
df_a["CER_A"] = df_a["CER_A"].clip(0, 1)
df_b["CER_B"] = df_b["CER_B"].clip(0, 1)

# ============================================================
# Match the same images
# ============================================================

df = pd.merge(df_a, df_b, on="file_path", how="inner")
df = df.dropna(subset=["CER_A", "CER_B"])

# Difference
# Negative = A better
# Positive = B better
df["CER_diff"] = df["CER_A"] - df["CER_B"]

print(f"Matched images: {len(df)}")

# ============================================================
# Statistics
# ============================================================

a_better = (df["CER_diff"] < 0).sum()
b_better = (df["CER_diff"] > 0).sum()
same = (df["CER_diff"] == 0).sum()

print(f"A better:  {a_better} ({a_better / len(df) * 100:.1f}%)")
print(f"B better:  {b_better} ({b_better / len(df) * 100:.1f}%)")
print(f"Same:      {same} ({same / len(df) * 100:.1f}%)")

# ============================================================
# Plot 1: Histogram of CER differences
# ============================================================

plt.figure(figsize=(9, 5))

plt.hist(
    df["CER_diff"],
    bins=40,
    alpha=0.75
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=2
)

plt.xlabel("CER Donut - CER Qwen")
plt.ylabel("Number of images")
plt.title("Difference in CER Between OCR Systems")

plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()


# ============================================================
# Plot 2: Each image, sorted by CER difference
# ============================================================

df_sorted = df.sort_values("CER_diff").reset_index(drop=True)

plt.figure(figsize=(12, 6))

x = np.arange(len(df_sorted))

plt.scatter(
    x,
    df_sorted["CER_A"],
    s=15,
    label="Donut"
)

plt.scatter(
    x,
    df_sorted["CER_B"],
    s=15,
    label="Qwen"
)

plt.xlabel("Images (sorted by CER difference)")
plt.ylabel("CER")
plt.title("OCR Performance Per Image")

plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()
