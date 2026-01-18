import common.helpers as helpers
import pandas as pd
import matplotlib.pyplot as plt
import predict
import numpy as np
import re


def check_data():
    key_file_path = helpers.get_key_folder_path("nutris")
    df = pd.read_excel(key_file_path / "nutris.xlsx")

    # Ensure column is string and handle NaNs
    ingredient_lengths = df["Ingredients"].astype(str).str.len()

    # 1. Summary statistics
    print("=== Length summary ===")
    print(ingredient_lengths.describe())

    # 2. Binned distribution
    bins = [0, 50, 100, 200, 300, 500, 1000, float("inf")]
    labels = ["0–50", "51–100", "101–200", "201–300", "301–500", "501–1000", "1000+"]

    length_bins = pd.cut(ingredient_lengths, bins=bins, labels=labels)
    print("\n=== Length distribution (binned) ===")
    print(length_bins.value_counts().sort_index())

    # Convert to string and compute lengths
    df["ingredient_length"] = df["Ingredients"].astype(str).str.len()

    # Find the longest row
    longest_row = df.loc[df["ingredient_length"].idxmax()]

    print("=== Longest Ingredients entry ===")
    print(f"Filename: {longest_row['FileName']}")
    print(f"Character length: {longest_row['ingredient_length']}")
    print("Ingredients text:")
    print(longest_row["Ingredients"])

    special_chars = "  "
    mask = df["Ingredients"].str.startswith(tuple(special_chars))
    problematic_rows = df[mask]

    if not problematic_rows.empty:
        print("=== Ingredients starting with [ ] ( ) { } ===")
        print(problematic_rows[["FileName", "Ingredients"]])
    else:
        print("No Ingredients entries start with [ ] ( ) { }.")

    # 3. Optional histogram
    ingredient_lengths.plot(kind="hist", bins=100, title="Ingredient Character Length Distribution")
    plt.xlabel("Character count")
    plt.ylabel("Frequency")
    plt.show()


def print_model_info(model_type, default=False):
    if default:
        from transformers import VisionEncoderDecoderModel
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    else:
        model = predict.get_model(model_type)
    print("Model max_length:", model.decoder.config.max_position_embeddings)
    print("Gene:", model.generation_config.max_length)


def check_df():
    key_file_path = helpers.get_key_folder_path("nutris")
    df = pd.read_excel(key_file_path / "nutris_cleaned.xlsx")

    check_array = ["(("]

    for item in check_array:
        check_containing_in_df(df, item, print_rows=True)


def check_containing_in_df(df, characters, print_rows=False):
    pattern = re.escape(characters)
    amount = df["Ingredients"].str.contains(pattern, na=False, regex=True).sum()
    print(f"Number of rows containing '{characters}':", amount)
    if print_rows and amount > 0:
        rows = df[df["Ingredients"].str.contains(pattern, na=False, regex=True)]
        print(rows[["FileName", "Ingredients"]].to_string(index=False))


if __name__ == "__main__":


    # check_data()
    check_df()
    quit()


    dataset = predict.get_processed_dataset("nutris-slim")
    print(dataset)
    # print first label of train set
    label = dataset["train"][0]["labels"]
    print("First label tokens:", label)
    quit()

    IGNORE_ID = -100


    def label_length(example):
        labels = np.array(example["labels"])
        return int((labels != IGNORE_ID).sum())


    lengths = dataset["train"].map(
        lambda x: {"length": label_length(x)}
    )["length"]

    lengths_np = np.array(lengths)

    print("Min:", lengths_np.min())
    print("Max:", lengths_np.max())
    print("Mean:", lengths_np.mean())
    print("Median:", np.median(lengths_np))
    print("95th percentile:", np.percentile(lengths_np, 95))
    print("99th percentile:", np.percentile(lengths_np, 99))

    plt.figure(figsize=(8, 4))
    plt.hist(lengths_np, bins=50)
    plt.axvline(512, color="red", linestyle="--", label="512")
    plt.axvline(1024, color="green", linestyle="--", label="1024")
    plt.legend()
    plt.title("Training Label Token Length Distribution")
    plt.xlabel("Tokens")
    plt.ylabel("Count")
    plt.show()
