import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import common.helpers as helpers

def plot_cer_and_wer_histogram(csv_path="ocr_eval_results_test.csv", show_imgs=False, bins=50, clip_max=1.0):    
 
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

    CER_MAX = 0.6
    CER_MIN = 0.5

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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

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

import ast
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# assumes `helpers.compute_cer` is available, same as in your original code


def _parse(x):
    if not isinstance(x, str):
        return x
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x


def _render_page(
    df,
    bins,
    save_path,
    images_per_bin=1,
    seed=42,
    dpi=300,
    rotate_to_portrait=True,
    show_cer=True,
    label_col_width=0.5,
    max_pred_lines=12,
):
    n_rows = len(bins)
    n_cols = images_per_bin

    # A4 portrait in inches
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = fig.add_gridspec(
        n_rows, n_cols + 1,
        width_ratios=[label_col_width] + [1] * n_cols,   # first column = row label
        left=0.03, right=0.98, top=0.985, bottom=0.015,
        wspace=0.06, hspace=0.18,
    )

    # rough char-wrap width for the side text panel (narrower than a full-width caption)
    wrap_width = 34

    for r, (label, mask_fn) in enumerate(bins):
        subset = df[mask_fn(df["CER"])]

        # label cell
        ax_lab = fig.add_subplot(gs[r, 0])
        ax_lab.axis("off")
        ax_lab.text(0.5, 0.5, f"{label}\n(n={len(subset)})", ha="center", va="center", fontsize=9)

        sample = (
            subset.sample(n=min(images_per_bin, len(subset)), random_state=seed)
            if len(subset)
            else subset
        )
        rows = list(sample.iterrows())

        for c in range(n_cols):
            # split each cell into an image area (left) and a caption area (right)
            cell_gs = gs[r, c + 1].subgridspec(1, 2, width_ratios=[2.1, 1.3], wspace=0.05)
            ax_img = fig.add_subplot(cell_gs[0])
            ax_txt = fig.add_subplot(cell_gs[1])

            ax_img.set_xticks([])
            ax_img.set_yticks([])
            for s in ax_img.spines.values():
                s.set_linewidth(0.5)
            ax_txt.axis("off")

            if c >= len(rows):
                continue  # empty cell (bin has fewer images than requested)

            _, row_data = rows[c]
            try:
                img = Image.open(row_data["file_name"]).convert("RGB")
                if rotate_to_portrait and img.width > img.height:
                    img = img.rotate(90, expand=True)  # force portrait orientation
                img.thumbnail((1600, 1600))  # keep memory / file size reasonable
                ax_img.imshow(img)
                ax_img.set_aspect("equal")

                pred_text = str(_parse(row_data["prediction"]))
                wrapped_lines = textwrap.wrap(pred_text, width=wrap_width) or [pred_text]
                if len(wrapped_lines) > max_pred_lines:
                    wrapped_lines = wrapped_lines[:max_pred_lines]
                    last = wrapped_lines[-1].rstrip()
                    # trim a bit of room so "..." doesn't push the line over wrap_width
                    wrapped_lines[-1] = last[: max(0, wrap_width - 3)].rstrip() + "..."
                pred_wrapped = "\n".join(wrapped_lines)

                if show_cer:
                    caption = f"CER: {row_data['CER'] * 100:.1f}%\n\nPredikcija:\n{pred_wrapped}"
                else:
                    caption = f"Predikcija:\n{pred_wrapped}"

                ax_txt.text(
                    0.0, 0.98, caption,
                    transform=ax_txt.transAxes, ha="left", va="top", fontsize=6.5,
                )
            except Exception as e:
                ax_img.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6)
                print(f"Could not load {row_data['file_name']}: {e}")

    fig.savefig(save_path, dpi=dpi)
    print(f"Saved to {save_path}")
    plt.show()


def plot_cer_bin_collage_two_pages(
    csv_path,
    images_per_bin=1,
    seed=42,
    save_path_page1="cer_bin_collage_a4_page1.png",
    save_path_page2="cer_bin_collage_a4_page2.png",
    dpi=300,
    rotate_to_portrait=True,
    show_cer=True,
    label_col_width=0.22,
    max_pred_lines=4,
):
    """
    Two A4-portrait collages, 6 CER bins each (12 bins total, highest error first):

    Page 1: CER > 100%, 100-90%, 90-80%, 80-70%, 70-60%, 60-50%
    Page 2: 50-40%, 40-30%, 30-20%, 20-10%, 10-5%, CER < 5%

    Each row shows `images_per_bin` randomly sampled images (portrait-oriented),
    with "CER: XX%" and the model's prediction written beside each image
    (long predictions are truncated with "..." after `max_pred_lines` lines).
    Ground truth is intentionally NOT printed - it should be read off the image itself.
    """
    df = pd.read_csv(csv_path)
    df["CER"] = df.apply(
        lambda row: helpers.compute_cer(_parse(row["target"]), _parse(row["prediction"])),
        axis=1,
    )

    bins_page1 = [
        ("CER > 100 %", lambda c: c > 1.0),
        ("100–90 %", lambda c: (c >= 0.9) & (c <= 1.0)),
        ("90–80 %", lambda c: (c >= 0.8) & (c < 0.9)),
        ("80–70 %", lambda c: (c >= 0.7) & (c < 0.8)),
        ("70–60 %", lambda c: (c >= 0.6) & (c < 0.7)),
        ("60–50 %", lambda c: (c >= 0.5) & (c < 0.6)),
    ]
    bins_page2 = [
        ("50–40 %", lambda c: (c >= 0.4) & (c < 0.5)),
        ("40–30 %", lambda c: (c >= 0.3) & (c < 0.4)),
        ("30–20 %", lambda c: (c >= 0.2) & (c < 0.3)),
        ("20–10 %", lambda c: (c >= 0.1) & (c < 0.2)),
        ("10–5 %", lambda c: (c >= 0.05) & (c < 0.1)),
        ("CER < 5 %", lambda c: c < 0.05),
    ]

    _render_page(
        df, bins_page1, save_path_page1,
        images_per_bin=images_per_bin, seed=seed, dpi=dpi,
        rotate_to_portrait=rotate_to_portrait, show_cer=show_cer,
        label_col_width=label_col_width, max_pred_lines=max_pred_lines,
    )
    _render_page(
        df, bins_page2, save_path_page2,
        images_per_bin=images_per_bin, seed=seed, dpi=dpi,
        rotate_to_portrait=rotate_to_portrait, show_cer=show_cer,
        label_col_width=label_col_width, max_pred_lines=max_pred_lines,
    )


path = r"C:\Users\Jakob\Downloads\final_test_eval_donut_results.csv"

plot_cer_bin_collage_two_pages(
    path,
    images_per_bin=2,  # bump to 2 if you want two images side by side per row
    save_path_page1="cer_bin_collage_donut_a4_page1.png",
    save_path_page2="cer_bin_collage_donut_a4_page2.png",
    dpi=300,
    max_pred_lines=12,
    label_col_width=0.1
)