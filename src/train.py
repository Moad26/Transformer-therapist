from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "input" / "empatheticdialogues"
MODEL_DIR = ROOT_DIR / "model"


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        verbose: bool = False,
        path: str = os.path.join(MODEL_DIR, "checkpoint.pt"),
    ) -> None:
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.verbose: bool = verbose
        self.path: str = path
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min = np.inf

    def save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        model: torch.nn.Module,
        optimizer: Adam,
    ):
        try:
            if self.verbose:
                print(
                    f"Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model..."
                )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                self.path,
            )
            self.val_loss_min = val_loss
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def __call__(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        model: torch.nn.Module,
        optimizer: Adam,
    ) -> bool:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, train_loss, val_loss, model, optimizer)

        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, train_loss, val_loss, model, optimizer)
            self.counter = 0

        return self.early_stop


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 10,
    learning_rate: float = 1e-4,
    num_epochs: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_path: Path = MODEL_DIR,
):
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if val_dataset
        else None
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print(f"Training on {device}")

    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_steps = 0
        avg_val_loss = np.inf
        avg_train_loss = np.inf
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")

        for batch in train_pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Handle the new return format from the model
            model_output = model(input_ids, labels)
            if isinstance(model_output, tuple):
                logits, adjusted_labels = model_output
            else:
                logits = model_output
                adjusted_labels = labels

            loss = criterion(
                logits.reshape(-1, logits.size(-1)), adjusted_labels.reshape(-1)
            )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_steps += 1
            train_pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / train_steps

        if val_loader:
            model.eval()
            val_loss = 0
            val_steps = 0

            with torch.no_grad():
                val_pbar = tqdm(
                    val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
                )

                for batch in val_pbar:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)

                    # Handle the new return format from the model
                    model_output = model(input_ids, labels)
                    if isinstance(model_output, tuple):
                        logits, adjusted_labels = model_output
                    else:
                        logits = model_output
                        adjusted_labels = labels

                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)), adjusted_labels.reshape(-1)
                    )
                    val_loss += loss.item()
                    val_steps += 1

                    val_pbar.set_postfix({"val_loss": loss.item()})

            avg_val_loss = val_loss / val_steps

            print(
                f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
            )

            if early_stopping(epoch, avg_train_loss, avg_val_loss, model, optimizer):
                break
        else:
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
            save_path = model_path / "model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                },
                save_path,
            )

    print("Training completed!")
    return model
