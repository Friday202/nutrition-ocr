import pandas as pd
import ast

base = "C:\\Users\\Jakob\\Downloads\\"

# Input CSV files
qwen_csv = base + "final_test_eval_qwen_results.csv"
donut_csv = base + "final_test_eval_donut_results.csv"

# Output CSV
output_csv = "merged_predictions.csv"

# ============================================================
# Normalize target / prediction values
# ============================================================

def normalize_text(value):
    """
    Convert different representations of the same text
    into one canonical format.

    Example:

        "['hello world']"
        "hello world"

    both become:

        "hello world"
    """

    if pd.isna(value):
        return ""

    value = str(value).strip()

    # If the value looks like a Python list, e.g.
    # "['some text']"
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list) and len(parsed) == 1:
                value = str(parsed[0]).strip()

        except (ValueError, SyntaxError):
            # If it isn't a valid Python list, leave it as-is
            pass

    return value.strip()


# ============================================================
# Read CSVs
# ============================================================

qwen = pd.read_csv(qwen_csv)
donut = pd.read_csv(donut_csv)


# ============================================================
# Check required columns
# ============================================================

required_columns = {"file_name", "target", "prediction"}

for name, df in [("Qwen", qwen), ("Donut", donut)]:

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} CSV is missing columns: "
            f"{', '.join(sorted(missing))}"
        )


# ============================================================
# Check duplicate file names
# ============================================================

if qwen["file_name"].duplicated().any():
    raise ValueError("Qwen CSV contains duplicate file_names.")

if donut["file_name"].duplicated().any():
    raise ValueError("Donut CSV contains duplicate file_names.")


# ============================================================
# Normalize target and prediction columns
# ============================================================

qwen["target_normalized"] = qwen["target"].apply(normalize_text)
qwen["prediction_normalized"] = qwen["prediction"].apply(normalize_text)

donut["target_normalized"] = donut["target"].apply(normalize_text)
donut["prediction_normalized"] = donut["prediction"].apply(normalize_text)


# ============================================================
# Check that file_names are the same
# ============================================================

qwen_files = set(qwen["file_name"])
donut_files = set(donut["file_name"])

missing_in_donut = qwen_files - donut_files
missing_in_qwen = donut_files - qwen_files

if missing_in_donut:
    print("Files missing from Donut CSV:")
    for f in sorted(missing_in_donut):
        print(f)

if missing_in_qwen:
    print("\nFiles missing from Qwen CSV:")
    for f in sorted(missing_in_qwen):
        print(f)

if missing_in_donut or missing_in_qwen:
    raise ValueError(
        "The two CSV files do not contain the same file_names."
    )


# ============================================================
# Merge
# ============================================================

merged = qwen[
    ["file_name", "target_normalized", "prediction_normalized"]
].merge(
    donut[
        ["file_name", "target_normalized", "prediction_normalized"]
    ],
    on="file_name",
    suffixes=("_qwen", "_donut")
)


# ============================================================
# Check that targets are identical after normalization
# ============================================================

target_mismatch = (
    merged["target_normalized_qwen"]
    != merged["target_normalized_donut"]
)

if target_mismatch.any():

    mismatches = merged[target_mismatch]

    print("\nTarget mismatches found:")
    print(
        mismatches[
            [
                "file_name",
                "target_normalized_qwen",
                "target_normalized_donut",
            ]
        ].to_string(index=False)
    )

    raise ValueError(
        f"Found {target_mismatch.sum()} target mismatches."
    )


# ============================================================
# Create final output
# ============================================================

result = merged[
    [
        "file_name",
        "target_normalized_qwen",
        "prediction_normalized_qwen",
        "prediction_normalized_donut",
    ]
].rename(
    columns={
        "target_normalized_qwen": "ground_truth",
        "prediction_normalized_qwen": "prediction_qwen",
        "prediction_normalized_donut": "prediction_donut",
    }
)


# ============================================================
# Save
# ============================================================

result.to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig"
)

print()
print(f"Successfully merged {len(result)} files.")
print(f"Output saved to: {output_csv}")