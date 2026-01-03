import matplotlib.pyplot as plt
import os
import json


def plot(model_type, version=''):
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

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot training loss (always connected)
    plt.plot(train_epochs, train_vals, label='Training Loss', color='blue', marker='o')

    # Plot validation loss (will handle gaps automatically with NaN)
    plt.plot(val_epochs, val_vals, label='Validation Loss', color='orange', marker='x')

    # Adding title and labels
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    model_type = "nutris-slim"
    # model_type = "sroie"
    version = ""

    plot(model_type, version)
