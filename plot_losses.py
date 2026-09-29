import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def _load_losses(path):
    """Load train/val loss curves from a single trainer_state.json file."""
    with open(path, 'r') as file:
        trainer_state = json.load(file)

    log_history = trainer_state['log_history']

    epochs = []
    train_loss = []
    validation_loss = []

    for entry in log_history:
        epochs.append(entry['epoch'])
        train_loss.append(entry.get('loss', None))
        validation_loss.append(entry.get('eval_loss', None))

    def filter_valid(x, y):
        pairs = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
        if not pairs:
            return [], []
        xs, ys = zip(*pairs)
        return list(xs), list(ys)

    train_epochs, train_vals = filter_valid(epochs, train_loss)
    val_epochs, val_vals = filter_valid(epochs, validation_loss)

    return train_epochs, train_vals, val_epochs, val_vals


def plot_loss(paths, labels=None, title='Training and Validation Loss Over Epochs',
              verbose=True):
    """
    Plot training/validation loss curves for one or more runs on a single figure.

    Parameters
    ----------
    paths : str or list of str
        Path(s) to trainer_state.json file(s).
    labels : list of str, optional
        Labels to use per run (defaults to the path itself).
    title : str, optional
        Plot title.
    verbose : bool, optional
        If True, print loss values for each run (same behavior as before).
    """
    if isinstance(paths, str):
        paths = [paths]

    if labels is None:
        labels = paths
    if len(labels) != len(paths):
        raise ValueError("labels must be the same length as paths")

    plt.figure(figsize=(10, 6))

    colors = cm.tab10.colors  # up to 10 distinct colors, cycles if more runs

    for i, (path, label) in enumerate(zip(paths, labels)):
        train_epochs, train_vals, val_epochs, val_vals = _load_losses(path)
        color = colors[i % len(colors)]

        if verbose:
            print(f"\n=== {label} ===")
            print("Validation Loss values:")
            for epoch, val in zip(val_epochs, val_vals):
                print(f"  Epoch {epoch:.2f}: {val:.4f}")

            print("Training Loss values (every 5th):")
            for amount, (epoch, val) in enumerate(zip(train_epochs, train_vals), start=1):
                if amount % 5 == 0:
                    print(f"  Epoch {epoch:.2f}: {val:.4f}")

        #if train_epochs:
        #    plt.plot(train_epochs, train_vals,
        #              label=f'{label} - Train', color=color, marker='o', linestyle='-')
        if val_epochs:
            plt.plot(val_epochs, val_vals,
                      label=f'{label}', color=color, marker='x', linestyle='--')

    plt.title(title)
    plt.xlabel('Epoha')
    plt.ylabel('Validacijska izguba')
    #plt.xscale('log')  # Log scale for better visibility of loss values
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage:
    #paths = [
    #    r"C:\Users\Jakob\Downloads\trainer_state_1.5B.json",
    #    r"C:\Users\Jakob\Downloads\trainer_state_3B.json",
    #    r"C:\Users\Jakob\Downloads\trainer_state_7B.json",
    #    r"C:\Users\Jakob\Downloads\trainer_state_14B.json",
    #    r"C:\Users\Jakob\Downloads\trainer_state_32B.json",
    #    ]
    # labels = ["1.5B", "3B", "7B", "14B", "32B"]
    paths = [
        r"C:\Users\Jakob\Downloads\trainer_state_1k.json",
        r"C:\Users\Jakob\Downloads\trainer_state_3k.json",
        r"C:\Users\Jakob\Downloads\trainer_state_5k.json",
        r"C:\Users\Jakob\Downloads\trainer_state_10k.json",
        r"C:\Users\Jakob\Downloads\trainer_state_20k.json",
        r"C:\Users\Jakob\Downloads\trainer_state_20k_O.json",
        r"C:\Users\Jakob\Downloads\trainer_state_20k_O_F.json",
    ]
    labels = ["1k", "3k", "5k", "10k", "18k", "18k_O", "18k_O_F"]

    # V slovenščini naj bo naslo
    plot_loss(paths, labels=labels, title="Validacijska izguba po epohah")