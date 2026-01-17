from common.helpers import get_xslx_dataframe, save_xslx_dataframe

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def split_top_level(cell):
    parts = []
    buf = []
    depth = 0

    n = len(cell)

    for i, ch in enumerate(cell):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)

        # split only on commas that:
        #  - are not inside parentheses
        #  - are not decimal commas (digit , digit)
        if ch == "," and depth == 0:
            prev_is_digit = i > 0 and cell[i - 1].isdigit()
            next_is_digit = i + 1 < n and cell[i + 1].isdigit()

            if not (prev_is_digit and next_is_digit):
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
                continue

        buf.append(ch)

    # last chunk
    part = "".join(buf).strip()
    if part:
        parts.append(part)

    return parts


def clean_ground_truth_text(df):
    total_removed = 0
    cleaned_df = df.copy()

    # Convert Ingredients column to string type and check data validity
    cleaned_df["Ingredients"] = cleaned_df["Ingredients"].astype(str)

    assert not cleaned_df["Ingredients"].isnull().any(), "Null values found in Ingredients column"
    assert not cleaned_df["Ingredients"].isna().any(), "NaN values found in Ingredients column"

    forbidden = {"nan", "none", "null", "N/A", "n/a", "N.a.", "N.a", ""}
    bad_mask = cleaned_df["Ingredients"].str.lower().isin(forbidden)
    assert not bad_mask.any(), f"Forbidden values found in Ingredients column: {cleaned_df[bad_mask]}"

    # 0. Convert "/" into ""
    cleaned_df["Ingredients"] = cleaned_df["Ingredients"].str.replace("/", "", regex=False)
    print("Replaced '/' characters in Ingredients column.")

    # 1. Remove leading/trailing whitespace and normalize internal whitespace
    cleaned_df["Ingredients"] = (
        cleaned_df["Ingredients"]
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    double_or_more_whitespace = cleaned_df["Ingredients"].str.contains(r"\s{2,}", regex=True)
    assert not double_or_more_whitespace.any(), "Double or more consecutive whitespaces found"

    # 2. Convert to lowercase
    cleaned_df["Ingredients"] = cleaned_df["Ingredients"].str.lower()

    # 3. Remove rows that contain string literal "sestavine" (case insensitive)
    sestavine_mask = cleaned_df["Ingredients"].str.contains("sestavine", case=False, na=False)
    print("Removing rows containing 'sestavine':", sestavine_mask.sum())
    total_removed += sestavine_mask.sum()
    cleaned_df = cleaned_df[~sestavine_mask]

    # 4. Remove trailing punctuation or special characters also starting char
    cleaned_df["Ingredients"] = cleaned_df["Ingredients"].str.rstrip(".,;:-_*/\\|")
    cleaned_df["Ingredients"] = cleaned_df["Ingredients"].str.lstrip(".,;:-_*/\\|")

    # Print row that has empty Ingredients after cleaning
    empty_ingredients_mask = cleaned_df["Ingredients"].str.strip() == ""
    if empty_ingredients_mask.any():
        print("[INFO] Number of rows with empty Ingredients after cleaning:", empty_ingredients_mask.sum())

    # 5. Clean up % sign spacing issues
    cleaned_df["Ingredients"] = cleaned_df["Ingredients"].str.replace(
        r"\s+%", "%", regex=True
    )
    assert not cleaned_df["Ingredients"].str.contains(r"\s+%", regex=True).any(), "Whitespace before % still exists"

    # 6. Ensure single space after comma
    cleaned_df["Ingredients"] = cleaned_df["Ingredients"].str.replace(
        r"\s*,\s*(?=[A-Za-z])", ", ", regex=True
    )

    # Ensure no whitespace between number and %
    whitespace_before_after_comma = cleaned_df["Ingredients"].str.contains(r"\s+,\s+", regex=True)
    if whitespace_before_after_comma.any():
        print("Removing rows with whitespace before and after comma: ", whitespace_before_after_comma.sum())
        total_removed += whitespace_before_after_comma.sum()
        cleaned_df = cleaned_df[~whitespace_before_after_comma]

    # 7. Find rows that have repeated ingredients
    def find_repeats(cell):
        parts = [p.lower() for p in split_top_level(cell)]

        duplicates = {p for p in parts if parts.count(p) > 1}
        return ", ".join(sorted(duplicates)) if duplicates else ""

    cleaned_df["Repeated Ingredients"] = cleaned_df["Ingredients"].apply(find_repeats)
    repeated_mask = cleaned_df["Repeated Ingredients"] != ""
    print("Number of rows with repeated ingredients:", repeated_mask.sum())
    total_removed += repeated_mask.sum()
    cleaned_df = cleaned_df[~repeated_mask]
    cleaned_df = cleaned_df.drop(columns=["Repeated Ingredients"])

    # Print final stats
    print("-"*50)
    print(f"Starting number of rows: {len(df)}")
    print(f"Total number of removed rows during cleaning: {total_removed}")
    print(f"Number of rows after cleaning: {len(cleaned_df)}")

    return cleaned_df


def stratified_length_split(df, text_col="Ingredients", test_size=1000, n_bins=20, seed=42, show_plot=False):
    df = df.copy()

    # metric: length of the Ingredients string
    df["_len"] = df[text_col].str.len()

    # bin by quantiles so each bin has ~equal mass
    df["_bin"] = pd.qcut(df["_len"], q=n_bins, duplicates="drop")

    rng = np.random.default_rng(seed)

    test_indices = []

    for _, group in df.groupby("_bin"):
        # proportion of this bin in full data
        frac = len(group) / len(df)
        n_test = max(1, int(round(frac * test_size)))

        sampled = rng.choice(group.index, size=min(n_test, len(group)), replace=False)
        test_indices.extend(sampled)

    # Trim in case rounding overshoots
    if len(test_indices) > test_size:
        test_indices = rng.choice(test_indices, size=test_size, replace=False).tolist()

    test_df = df.loc[test_indices].drop(columns=["_len", "_bin"])
    train_df = df.drop(index=test_indices).drop(columns=["_len", "_bin"])

    if show_plot:
        train_lens = train_df[text_col].str.len()
        test_lens = test_df[text_col].str.len()

        # Compute shared bin edges from all data
        all_lens = np.concatenate([train_lens.values, test_lens.values])
        bins = np.histogram_bin_edges(all_lens, bins=50)

        # Plot length distributions
        plt.figure(figsize=(10, 6))
        plt.hist(train_lens, bins=bins, alpha=0.5, label="Train", density=True)
        plt.hist(test_lens, bins=bins, alpha=0.5, label="Test", density=True)
        plt.xlabel("Length of Ingredients String")
        plt.ylabel("Density")
        plt.title("Length Distribution of Ingredients in Train and Test Sets")
        plt.legend()
        plt.show()

    return train_df, test_df


if __name__ == "__main__":
    data_name = "nutris"

    df = get_xslx_dataframe(data_name)
    cleaned_df = clean_ground_truth_text(df)
    train_df, test_df = stratified_length_split(cleaned_df, test_size=1000, n_bins=20, seed=42)

    # Save cleaned dataframe
    save_xslx_dataframe(cleaned_df, "nutris_cleaned.xlsx", data_name=data_name)
    save_xslx_dataframe(train_df, "nutris_cleaned_train.xlsx", data_name=data_name)
    save_xslx_dataframe(test_df, "nutris_cleaned_test.xlsx", data_name=data_name)
