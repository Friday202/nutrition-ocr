import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import common.helpers as helpers


def plot_loss(model_type, version=''):
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


def plot_learning_rate(model_type, version=''):
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


import pandas as pd
import matplotlib.pyplot as plt


def plot_cer_and_wer_histogram(csv_path="ocr_eval_results_test.csv", bins=50, clip_max=1.0):
    """
    Plot histograms of per-sample CER and WER using only pandas + matplotlib,
    with horizontal lines at 2% and 5% for CER to indicate 'good' and 'acceptable' thresholds.
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Calculate CER and WER if not already present
    if "CER" not in df.columns:
        df["CER"] = df.apply(lambda row: helpers.compute_cer(eval(row["target"]), eval(row["prediction"])), axis=1)
    if "WER" not in df.columns:
        df["WER"] = df.apply(lambda row: helpers.compute_wer(eval(row["target"]), eval(row["prediction"])), axis=1)

    CER_MIN = 0.5
    CER_MAX = 0.55

    mid_cer_df = df[(df["CER"] >= CER_MIN) & (df["CER"] < CER_MAX)]

    print(f"Found {len(mid_cer_df)} samples with CER between {CER_MIN*100} and {CER_MAX*100}%\n")

    for idx, row in mid_cer_df.iterrows():
        gt = eval(row["target"])
        pred = eval(row["prediction"])
        cer = row["CER"]
        wer = row["WER"]

        print(f"Sample index: {idx}")
        print(f"GT   : {helpers.get_normalized_text(gt)}")
        print(f"Pred : {helpers.get_normalized_text(pred)}")
        print(f"CER  : {cer:.4f}, WER: {wer:.4f}")
        print("-" * 50)

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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # CER histogram
    counts_cer, bins_cer, patches_cer = axes[0].hist(cer, bins=bins, color="steelblue", edgecolor="black")
    axes[0].set_title("Per-image CER distribution")
    axes[0].set_xlabel("CER")
    axes[0].set_ylabel("Number of images")

    # Add vertical lines at 2% and 6%
    axes[0].axvline(0.02, color="red", linestyle="--", linewidth=2, label="2% threshold")
    axes[0].axvline(0.06, color="orange", linestyle="--", linewidth=2, label="6% threshold")
    axes[0].legend()

    # WER histogram
    axes[1].hist(wer, bins=bins, color="darkorange", edgecolor="black")
    axes[1].set_title("Per-image WER distribution")
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


if __name__ == "__main__":
    model_type = "nutris-slim"
    # model_type = "sroie"
    version = "checkpoint-24000"

    # plot_loss(model_type, version)
    # plot_learning_rate(model_type, version)
    plot_cer_and_wer_histogram()
    # plot_fer_histogram()
