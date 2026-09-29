import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import common.helpers as helpers

"""
Script is used to plot results from traning (qwen and donut)
"""

def plot_loss(model_type, version='', path=None):
    if path is None:
        if version:
            path = f'outputs/{model_type}/{version}/trainer_state.json'
        else:
            checkpoints = [d for d in os.listdir(f'outputs/{model_type}/') if d.startswith('checkpoint-')]
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found.")
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            path = f'outputs/{model_type}/{latest_checkpoint}/trainer_state.json'

    with open(path, 'r') as file:
        trainer_state = json.load(file)

    # Extract log history from the trainer_state
    log_history = trainer_state['log_history']

    # Initialize lists to store the training and validation loss
    train_loss = []
    validation_loss = []
    epochs = []

    # Iterate through the log history
    for entry in log_history:
        epochs.append(entry['epoch'])

        # Append training loss (always available as 'loss' in log history)
        if 'loss' in entry:
            train_loss.append(entry['loss'])
        else:
            train_loss.append(None)  # In case there's no training loss entry

        # Append validation (evaluation) loss (if it exists)
        if 'eval_loss' in entry:
            validation_loss.append(entry['eval_loss'])
        else:
            validation_loss.append(None)  # Use NaN for gaps

    def filter_valid(x, y):
        xs, ys = zip(*[(xi, yi) for xi, yi in zip(x, y) if yi is not None])
        return xs, ys

    train_epochs, train_vals = filter_valid(epochs, train_loss)
    val_epochs, val_vals = filter_valid(epochs, validation_loss)

    # Print values of validation loss on 2 decimals
    print("Validation Loss values:")
    for epoch, val in zip(val_epochs, val_vals):
        print(f"Epoch {epoch:.2f}: {val:.4f}")

    # print train loss but only every 5th one
    print("\nTraining Loss values:")
    amount = 0
    for epoch, val in zip(train_epochs, train_vals):
        amount += 1
        if amount % 5 == 0:
            print(f"Epoch {epoch:.2f}: {val:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot training loss (always connected)
    plt.plot(train_epochs, train_vals, label='Training Loss', color='blue', marker='o')

    # Plot validation loss (will handle gaps automatically with NaN)
    plt.plot(val_epochs, val_vals, label='Validation Loss', color='red', marker='x')

    # Adding title and labels
    plt.title('Training and Validation Loss Over Epochs for: ' + model_type + version)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_learning_rate(model_type, version='', path=None):
    if path is None:
        if version:
            path = f'outputs/{model_type}/{version}/trainer_state.json'
        else:
            checkpoints = [d for d in os.listdir(f'outputs/{model_type}/') if d.startswith('checkpoint-')]
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found.")
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            path = f'outputs/{model_type}/{latest_checkpoint}/trainer_state.json'

    with open(path, 'r') as file:
        trainer_state = json.load(file)

    log_history = trainer_state['log_history']

    steps = []
    learning_rates = []

    for entry in log_history:
        if 'learning_rate' in entry and 'step' in entry:
            steps.append(entry['step'])
            learning_rates.append(entry['learning_rate'])

    if not learning_rates:
        raise ValueError("No learning rate information found in trainer_state.json")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, learning_rates, color='green')
    plt.title('Learning Rate Schedule\n' + model_type + version)
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.show()

def plot_cer_and_wer_histogram(csv_path="ocr_eval_results_test.csv", show_imgs=False, bins=50, clip_max=1.0):
    """
    Plot histograms of per-sample CER and WER using only pandas + matplotlib,
    with horizontal lines at 2% and 5% for CER to indicate 'good' and 'acceptable' thresholds.
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    import ast

    def parse(x):
        if not isinstance(x, str):
            return x
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x

    # Calculate CER and WER if not already present
    df["CER"] = df.apply(
        lambda row: helpers.compute_cer(parse(row["target"]), parse(row["prediction"])),
        axis=1,
    )

    df["WER"] = df.apply(
        lambda row: helpers.compute_wer(parse(row["target"]), parse(row["prediction"])),
        axis=1,
    )

    # go thru whole dataframe
    for idx, row in df.iterrows():
        gt = parse(row["target"])
        pred = parse(row["prediction"])
        cer = row["CER"]
        wer = row["WER"]
        file_name = row["file_name"]

        out_filename = f"per_sample_CER_qwen.csv"
        # write to csv file
        with open(out_filename, "a") as f:
            f.write(f"{file_name},{cer:.4f},{wer:.4f}\n")

    #if "CER" not in df.columns:
    #    df["CER"] = df.apply(lambda row: helpers.compute_cer(eval(row["target"]), eval(row["prediction"])), axis=1)
    #if "WER" not in df.columns:
    #    df["WER"] = df.apply(lambda row: helpers.compute_wer(eval(row["target"]), eval(row["prediction"])), axis=1)

    CER_MAX = 0.5
    CER_MIN = 0.4

    mid_cer_df = df[(df["CER"] >= CER_MIN) & (df["CER"] < CER_MAX)]

    print(f"Found {len(mid_cer_df)} samples with CER between {CER_MIN*100} and {CER_MAX*100}%\n")

    for idx, row in mid_cer_df.iterrows():
        #gt = eval(row["target"])
        #pred = eval(row["prediction"])
        gt = parse(row["target"])
        pred = parse(row["prediction"])
        file_name = row["file_name"]
        cer = row["CER"]
        wer = row["WER"]

        print(f"Sample index: {idx}")
        print(f"GT   : {helpers.get_normalized_text(gt)}")
        print(f"Pred : {helpers.get_normalized_text(pred)}")
        print(f"CER  : {cer:.4f}, WER: {wer:.4f}")
        print(f"File : {file_name}")
        print("-" * 50)
        if show_imgs:
            # show image pillow
            from PIL import Image
            img = Image.open(file_name)
            img.show()
            # wait for user input to continue
            input("Press Enter to continue...")

    # print total number of samples
    print(f"Total number of samples: {len(df)}")

    num_below_2_percent = (df["CER"] < 0.02).sum()
    num_below_6_percent = (df["CER"] < 0.06).sum()
    num_below_10_percent = (df["CER"] < 0.10).sum()

    print(f"Number of samples with CER below 2%: {num_below_2_percent}, which is {(num_below_2_percent / len(df)) * 100:.2f}% of total")
    print(f"Number of samples with CER below 6%: {num_below_6_percent}, which is {(num_below_6_percent / len(df)) * 100:.2f}% of total")
    print(f"Number of samples with CER below 10%: {num_below_10_percent}, which is {(num_below_10_percent / len(df)) * 100:.2f}% of total")

    num_wer_below_2_percent = (df["WER"] < 0.02).sum()
    num_wer_below_6_percent = (df["WER"] < 0.06).sum()

    print(f"Number of samples with WER below 2%: {num_wer_below_2_percent}, which is {(num_wer_below_2_percent / len(df)) * 100:.2f}% of total")
    print(f"Number of samples with WER below 6%: {num_wer_below_6_percent}, which is {(num_wer_below_6_percent / len(df)) * 100:.2f}% of total")

    # Clip values for visualization
    cer = df["CER"].clip(0, clip_max)
    wer = df["WER"].clip(0, clip_max)

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69), sharey=True)

    # CER histogram
    counts_cer, bins_cer, patches_cer = axes[0].hist(cer, bins=bins, color="steelblue", edgecolor="black")
    axes[0].set_title("Porazdelitev CER po slikah")
    axes[0].set_xlabel("CER")
    axes[0].set_ylabel("Število slik")

    # Add vertical lines at 2% and 6%
    axes[0].axvline(0.02, color="red", linestyle="--", linewidth=2, label="2% threshold")
    axes[0].axvline(0.06, color="orange", linestyle="--", linewidth=2, label="6% threshold")
    axes[0].legend()

    # WER histogram
    axes[1].hist(wer, bins=bins, color="darkorange", edgecolor="black")
    axes[1].set_title("Porazdelitev WER po slikah")
    axes[1].set_xlabel("WER")

    # Add vertical lines at 2% and 6%
    axes[1].axvline(0.02, color="red", linestyle="--", linewidth=2, label="2% threshold")
    axes[1].axvline(0.06, color="orange", linestyle="--", linewidth=2, label="6% threshold")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_fer_histogram(csv_path="ocr_eval_results.csv", bins=50):
    # Load CSV
    df = pd.read_csv(csv_path)

    # calculate fer

    df['FER'] = df.apply(lambda row: helpers.compute_fer(eval(row['target']), eval(row['prediction'])), axis=1)
    # print first target and prediction with their FER
    print(df[['target', 'prediction', 'FER']].head())

    df['FER'] = df['FER'].clip(0, 1.0)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['FER'], bins=bins, color='purple', edgecolor='black')
    plt.title('Per-image FER distribution')
    plt.xlabel('FER')
    plt.ylabel('Number of images')
    plt.grid(True)
    plt.show()

def compute_overall_cer(csv_path="ocr_eval_results_test.csv", cer_exclude_threshold=0.95):
    """
    Computes corpus-level (micro-averaged) CER:
    sum(edit_distance) / sum(len(GT)) across all samples.
    Excludes samples whose per-image CER exceeds cer_exclude_threshold
    (treated as catastrophic failures, e.g. wrong-section extraction),
    since they distort the overall number differently than ordinary noise.
    """
    df = pd.read_csv(csv_path)

    import ast
    def parse(x):
        if not isinstance(x, str):
            return x
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x

    def levenshtein(a, b):
        m, n = len(a), len(b)
        if m == 0:
            return n
        if n == 0:
            return m
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[n]

    # First pass: per-image CER using existing helper, so we can filter
    df["CER"] = df.apply(
        lambda row: helpers.compute_cer(parse(row["target"]), parse(row["prediction"])),
        axis=1,
    )

    total_samples = len(df)
    excluded_df = df[df["CER"] > cer_exclude_threshold]
    kept_df = df[df["CER"] <= cer_exclude_threshold]

    num_excluded = len(excluded_df)
    print(f"Excluding {num_excluded} samples with CER > {cer_exclude_threshold*100:.0f}% "
          f"({num_excluded/total_samples*100:.2f}% of {total_samples} total)")

    total_edits = 0
    total_chars = 0

    for _, row in kept_df.iterrows():
        gt = helpers.get_normalized_text(parse(row["target"]))
        pred = helpers.get_normalized_text(parse(row["prediction"]))
        total_edits += levenshtein(gt, pred)
        total_chars += len(gt)

    overall_cer = total_edits / total_chars if total_chars > 0 else float("nan")
    macro_cer = kept_df["CER"].mean()

    print(f"Total GT characters (kept samples): {total_chars}")
    print(f"Total edit operations (kept samples): {total_edits}")
    print(f"Overall CER excl. failures (micro-avg): {overall_cer:.4f} ({overall_cer*100:.2f}%)")
    print(f"Mean per-image CER excl. failures (macro-avg): {macro_cer:.4f} ({macro_cer*100:.2f}%)")

    return overall_cer


import pandas as pd
import ast

def save_samples_with_cer_below_threshold(
    csv_path="ocr_eval_results_test.csv",
    output_path="samples_cer_below_2_percent.csv",
    cer_threshold=0.02,
):
    """
    Save file_name values for all samples with CER below the given threshold.

    Args:
        csv_path: Input OCR evaluation CSV.
        output_path: Output CSV file containing file_name column.
        cer_threshold: CER threshold (default: 2%).
    """

    df = pd.read_csv(csv_path)

    def parse(x):
        if not isinstance(x, str):
            return x
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x

    # Compute CER if not already available
    if "CER" not in df.columns:
        df["CER"] = df.apply(
            lambda row: helpers.compute_cer(
                parse(row["target"]),
                parse(row["prediction"])
            ),
            axis=1,
        )

    # Filter samples
    filtered_df = df[df["CER"] < cer_threshold]

    print(
        f"Found {len(filtered_df)} samples with CER < {cer_threshold*100:.1f}% "
        f"out of {len(df)} total samples."
    )

    # Save only file names
    output_df = filtered_df[["file_name"]]
    # make csv to actually be imaege_path column
    # output_df.rename(columns={"file_name": "image_path"}, inplace=True)
    output_df.to_csv(output_path, index=False)

    print(f"Saved file names to: {output_path}")

    return output_df
    


if __name__ == "__main__":
    model_type = "nutris-slim"
    # model_type = "sroie"
    version = "checkpoint-24000"
    paths = [
        r"C:\Users\Jakob\Downloads\trainer_state_1.5B.json",
        r"C:\Users\Jakob\Downloads\trainer_state_3B.json",
        r"C:\Users\Jakob\Downloads\trainer_state_7B.json",
        r"C:\Users\Jakob\Downloads\trainer_state_14B.json",
        r"C:\Users\Jakob\Downloads\trainer_state_32B.json",
        ]

    #path = r"C:\Users\Jakob\Downloads\trainer_state.json"
    #plot_loss(model_type, version, path)
    #plot_learning_rate(model_type, version, path)
    # path = r"C:\Users\Jakob\Downloads\nutris_eval_results.csv"
    path = r"C:\Users\Jakob\Downloads\final_test_eval_qwen_results.csv"
    #path = r"C:\Users\Jakob\Downloads\nutris-flat-original-size_eval_results.csv"

    #path = r"C:\Users\Jakob\Downloads\results_7B_finetuned_new_lora_params.csv"
    
    plot_cer_and_wer_histogram(path, True)
    #save_samples_with_cer_below_threshold(path, output_path="samples_cer_below_6_percent_qwen_final.csv", cer_threshold=0.06)
    # compute_overall_cer(path)
    # plot_fer_histogram()
