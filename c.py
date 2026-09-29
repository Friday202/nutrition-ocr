import pandas as pd
from jiwer import cer

def _as_text(value):
    if pd.isna(value):
        return ""
    return str(value)


# Load CSV
df = pd.read_csv("merged_predictions.csv")

df["cer_qwen"] = df.apply(
    lambda row: cer(_as_text(row["ground_truth"]), _as_text(row["prediction_qwen"])),
    axis=1,
)
df["cer_donut"] = df.apply(
    lambda row: cer(_as_text(row["ground_truth"]), _as_text(row["prediction_donut"])),
    axis=1,
)

df["cer_qwen_donut"] = df.apply(
    lambda row: cer(_as_text(row["prediction_qwen"]), _as_text(row["prediction_donut"])),
    axis=1,
)



mask = (
    (df["cer_qwen_donut"] > 0.8) &
    (df["cer_qwen_donut"] < 1.0) &
    (df["prediction_qwen"] != df["ground_truth"]) &
    (df["prediction_donut"] != df["ground_truth"])
)

matches = df.loc[mask, [
    "file_name",
    "ground_truth",
    "prediction_qwen",
    "prediction_donut",
    "cer_qwen_donut",
    "cer_qwen",
    "cer_donut",
]]

# Print each matching row compactly
# open a .txt file and write the output to it
with open("matching_rows_cer2.txt", "w", encoding="utf-8") as f:
    f.write(f"Total matches: {len(matches)}\n")
    f.write(f"Average CER Qwen:  {df['cer_qwen'].mean() * 100:.2f}%\n")
    f.write(f"Average CER Donut: {df['cer_donut'].mean() * 100:.2f}%\n")
    f.write("=" * 80 + "\n")
    for _, row in matches.iterrows():
        f.write(f"File:       {row['file_name']}\n")
        f.write(f"Ground truth: {row['ground_truth']}\n")
        f.write(f"Qwen:         {row['prediction_qwen']}\n")
        f.write(f"Qwen/Donut CER: {row['cer_qwen_donut'] * 100:.2f}%\n")
        f.write(f"CER Qwen:     {row['cer_qwen'] * 100:.2f}%\n")
        f.write(f"Donut:        {row['prediction_donut']}\n")
        f.write(f"CER Donut:    {row['cer_donut'] * 100:.2f}%\n")
        f.write("-" * 80 + "\n")