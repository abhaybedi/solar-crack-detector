from pathlib import Path

import numpy as np
import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:
    def __init__(
        self,
        model,
        crit,
        optim=None,
        train_dl=None,
        val_test_dl=None,
        cuda=True,
        early_stopping_patience=-1,
    ):
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = bool(cuda and t.cuda.is_available())
        self._early_stopping_patience = early_stopping_patience

        self._device = t.device(
            "cuda" if self._cuda else "cpu"
        )

        self._model = self._model.to(self._device)
        self._crit = self._crit.to(self._device)

    def save_checkpoint(self, epoch):
        checkpoint_directory = Path("checkpoints")
        checkpoint_directory.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_directory / (
            f"checkpoint_{epoch:03d}.ckp"
        )

        checkpoint = {
            "epoch": epoch,
            "state_dict": self._model.state_dict(),
        }

        if self._optim is not None:
            checkpoint["optimizer"] = self._optim.state_dict()

        t.save(checkpoint, checkpoint_path)

    def restore_checkpoint(self, epoch_n):
        checkpoint_path = Path("checkpoints") / (
            f"checkpoint_{epoch_n:03d}.ckp"
        )

        checkpoint = t.load(
            checkpoint_path,
            map_location=self._device,
        )

        self._model.load_state_dict(checkpoint["state_dict"])

        if self._optim is not None and "optimizer" in checkpoint:
            self._optim.load_state_dict(checkpoint["optimizer"])

        self._model.to(self._device)

    def save_onnx(self, fn):
        original_device = next(self._model.parameters()).device
        original_training_state = self._model.training

        self._model.cpu()
        self._model.eval()

        example_input = t.randn(1, 3, 300, 300)

        t.onnx.export(
            self._model,
            example_input,
            fn,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        self._model.to(original_device)
        self._model.train(original_training_state)

    def train_step(self, x, y):
        if self._optim is None:
            raise RuntimeError(
                "An optimizer is required to perform a training step"
            )

        self._optim.zero_grad()

        predictions = self._model(x)
        loss = self._crit(predictions, y)

        loss.backward()
        self._optim.step()

        return loss

    def val_test_step(self, x, y):
        predictions = self._model(x)
        loss = self._crit(predictions, y)

        return loss, predictions

    def train_epoch(self):
        if self._train_dl is None:
            raise RuntimeError("No training DataLoader was provided")

        self._model.train()

        total_loss = 0.0
        number_of_samples = 0

        progress = tqdm(
            self._train_dl,
            desc="Training",
            leave=False,
        )

        for x, y in progress:
            x = x.to(self._device, non_blocking=self._cuda)
            y = y.to(
                self._device,
                dtype=t.float32,
                non_blocking=self._cuda,
            )

            loss = self.train_step(x, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            number_of_samples += batch_size

            progress.set_postfix(loss=f"{loss.item():.6f}")

        if number_of_samples == 0:
            raise RuntimeError("The training DataLoader is empty")

        return total_loss / number_of_samples

    @staticmethod
    def _calculate_f1(labels, predictions):
        binary_predictions = (predictions >= 0.5).astype(np.int64)
        binary_labels = labels.astype(np.int64)

        label_f1_scores = f1_score(
            binary_labels,
            binary_predictions,
            average=None,
            zero_division=0,
        )

        mean_f1 = float(np.mean(label_f1_scores))

        return mean_f1, label_f1_scores

    def val_test(self):
        if self._val_test_dl is None:
            raise RuntimeError("No validation/test DataLoader was provided")

        self._model.eval()

        total_loss = 0.0
        number_of_samples = 0
        all_predictions = []
        all_labels = []

        progress = tqdm(
            self._val_test_dl,
            desc="Validation",
            leave=False,
        )

        with t.no_grad():
            for x, y in progress:
                x = x.to(self._device, non_blocking=self._cuda)
                y = y.to(
                    self._device,
                    dtype=t.float32,
                    non_blocking=self._cuda,
                )

                loss, predictions = self.val_test_step(x, y)

                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                number_of_samples += batch_size

                all_predictions.append(predictions.cpu())
                all_labels.append(y.cpu())

                progress.set_postfix(loss=f"{loss.item():.6f}")

        if number_of_samples == 0:
            raise RuntimeError("The validation DataLoader is empty")

        predictions = t.cat(all_predictions, dim=0).numpy()
        labels = t.cat(all_labels, dim=0).numpy()

        average_loss = total_loss / number_of_samples
        mean_f1, label_f1_scores = self._calculate_f1(
            labels,
            predictions,
        )

        crack_f1 = float(label_f1_scores[0])
        inactive_f1 = float(label_f1_scores[1])

        print(
            f"Validation loss: {average_loss:.6f} | "
            f"crack F1: {crack_f1:.4f} | "
            f"inactive F1: {inactive_f1:.4f} | "
            f"mean F1: {mean_f1:.4f}"
        )

        return average_loss

    def fit(self, epochs=-1):
        if self._train_dl is None:
            raise RuntimeError("No training DataLoader was provided")

        if self._val_test_dl is None:
            raise RuntimeError("No validation DataLoader was provided")

        if self._optim is None:
            raise RuntimeError("No optimizer was provided")

        assert self._early_stopping_patience > 0 or epochs > 0

        train_losses = []
        validation_losses = []

        epoch = 0
        best_validation_loss = float("inf")
        best_epoch = None
        epochs_without_improvement = 0

        while True:
            if epochs > 0 and epoch >= epochs:
                break

            print(f"\nEpoch {epoch + 1}")

            train_loss = self.train_epoch()
            validation_loss = self.val_test()

            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

            print(
                f"Train loss: {train_loss:.6f} | "
                f"Validation loss: {validation_loss:.6f}"
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                self.save_checkpoint(epoch)

                print(
                    f"Saved improved checkpoint for epoch {epoch + 1}"
                )
            else:
                epochs_without_improvement += 1

            epoch += 1

            if (
                self._early_stopping_patience > 0
                and epochs_without_improvement
                >= self._early_stopping_patience
            ):
                print(
                    "Early stopping: validation loss did not improve for "
                    f"{self._early_stopping_patience} epochs."
                )
                break

        if best_epoch is not None:
            self.restore_checkpoint(best_epoch)
            print(
                f"Restored best checkpoint from epoch {best_epoch + 1}."
            )

        return train_losses, validation_losses