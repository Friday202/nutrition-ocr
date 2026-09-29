import csv
import matplotlib.pyplot as plt


def read_lengths(csv_path):
    lengths = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lengths.append(int(row["n_tokens"]))
    return lengths


def main():
    train_lengths = read_lengths("C:\\Users\\Jakob\\Downloads\\train_prompt_lengths.csv")
    val_lengths = read_lengths("C:\\Users\\Jakob\\Downloads\\val_prompt_lengths.csv")

    plt.figure(figsize=(10, 6))

    plt.hist(
        train_lengths,
        bins=50,
        alpha=0.6,
        label=f"Učna množica",
        color="steelblue",
        edgecolor="black",
    )
    plt.hist(
        val_lengths,
        bins=50,
        alpha=0.6,
        label=f"Validacijska množica",
        color="darkorange",
        edgecolor="black",
    )

    plt.axvline(
        x=1536,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Meja: 1536 žetonov",
    )

    plt.title("Porazdelitev dolžine pozivov (v žetonih)")
    plt.xlabel("Število žetonov")
    plt.ylabel("Število primerov")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("porazdelitev_dolzine_pozivov.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()