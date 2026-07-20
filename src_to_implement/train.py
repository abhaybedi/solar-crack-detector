from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from sklearn.model_selection import train_test_split

import model
from data import ChallengeDataset
from trainer import Trainer

CSV_FILE = "data.csv"
VALIDATION_SIZE = 0.2
RANDOM_SEED = 42

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 0


def find_label_columns(dataframe):
    lower_to_original = {
        str(column).lower(): column for column in dataframe.columns
    }

    crack_column = None
    inactive_column = None

    for name, original_name in lower_to_original.items():
        if "crack" in name:
            crack_column = original_name
        if "inactive" in name:
            inactive_column = original_name

    if crack_column is not None and inactive_column is not None:
        return crack_column, inactive_column

    usable_columns = [
        column
        for column in dataframe.columns
        if not str(column).lower().startswith("unnamed")
    ]

    if len(usable_columns) < 3:
        return None

    return usable_columns[-2], usable_columns[-1]


def create_stratification_labels(dataframe):
    label_columns = find_label_columns(dataframe)

    if label_columns is None:
        return None

    first_label = dataframe[label_columns[0]].astype(int)
    second_label = dataframe[label_columns[1]].astype(int)

    combined_labels = (
        first_label.astype(str) + "_" + second_label.astype(str)
    )

    label_counts = combined_labels.value_counts()

    if len(label_counts) < 2 or label_counts.min() < 2:
        return None

    validation_samples = int(
        np.ceil(len(dataframe) * VALIDATION_SIZE)
    )
    training_samples = len(dataframe) - validation_samples

    if (
        validation_samples < len(label_counts)
        or training_samples < len(label_counts)
    ):
        return None

    return combined_labels


def main():
    t.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if t.cuda.is_available():
        t.cuda.manual_seed_all(RANDOM_SEED)
        t.backends.cudnn.benchmark = True

    csv_path = Path(CSV_FILE)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Could not find {CSV_FILE!r} in {Path.cwd()}"
        )

    dataframe = pd.read_csv(csv_path)

    if len(dataframe) < 2:
        raise ValueError(
            "The dataset must contain at least two samples"
        )

    stratification_labels = create_stratification_labels(dataframe)

    train_data, validation_data = train_test_split(
        dataframe,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=stratification_labels,
    )

    train_dataset = ChallengeDataset(
        train_data,
        mode="train",
    )
    validation_dataset = ChallengeDataset(
        validation_data,
        mode="val",
    )

    use_cuda = t.cuda.is_available()

    train_loader = t.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_cuda,
        drop_last=False,
    )

    validation_loader = t.utils.data.DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_cuda,
        drop_last=False,
    )

    network = model.ResNet()

    # The network already applies sigmoid. Therefore, use BCELoss rather
    # than BCEWithLogitsLoss.
    criterion = t.nn.BCELoss()

    optimizer = t.optim.Adam(
        network.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    trainer = Trainer(
        model=network,
        crit=criterion,
        optim=optimizer,
        train_dl=train_loader,
        val_test_dl=validation_loader,
        cuda=use_cuda,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )

    results = trainer.fit(epochs=MAX_EPOCHS)

    train_losses, validation_losses = results

    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, validation_losses, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("losses.png", dpi=200)
    plt.close()

    trainer.save_onnx("model.onnx")

    print("\nTraining complete.")
    print("Loss plot written to: losses.png")
    print("Best model written to: model.onnx")


if __name__ == "__main__":
    main()