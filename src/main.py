import argparse
import random
import numpy as np
import os
import torch
from pathlib import Path

from data import EmpatheticConv
from train import train_model
from model import Transformer

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Transformer for Empathetic Dialogues"
    )

    # Data parameters
    parser.add_argument(
        "--max_seq_len", type=int, default=128, help="Maximum sequence length"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="facebook/blenderbot-400M-distill",
        help="Tokenizer to use",
    )

    # Model parameters
    parser.add_argument(
        "--embed_dim", type=int, default=512, help="Embedding dimension"
    )
    parser.add_argument(
        "--num_head", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs"
    )

    # System parameters
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Evaluation
    parser.add_argument(
        "--eval_only", action="store_true", help="Only run evaluation, don't train"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint for evaluation",
    )

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_device(device_arg: str):
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_datasets(args):
    train_dataset = EmpatheticConv(
        split="train", max_seq_len=args.max_seq_len, tokenizer_name=args.tokenizer_name
    )
    try:
        val_dataset = EmpatheticConv(
            split="valid",
            max_seq_len=args.max_seq_len,
            tokenizer_name=args.tokenizer_name,
        )
    except FileNotFoundError:
        print("Validation dataset not found, using training data split")
        val_dataset = None
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset


def create_model(args, vocab_size: int):
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        seq_len=args.max_seq_len,
        num_head=args.num_head,
        dropout=args.dropout,
        num_layers=args.num_layers,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def load_checkpoint(model, optimizer, checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    train_loss = checkpoint.get("train_loss", 0)
    val_loss = checkpoint.get("val_loss", 0)

    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

    return model, optimizer, epoch


def evaluate_model(model, dataset, tokenizer, device: str, num_samples: int = 5):
    model.eval()
    model = model.to(device)
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)

        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        with torch.no_grad():
            start_token = (
                tokenizer.convert_tokens_to_ids("<start>")
                if "<start>" in tokenizer.get_vocab()
                else 0
            )
            end_token = (
                tokenizer.convert_tokens_to_ids("<end>")
                if "<end>" in tokenizer.get_vocab()
                else tokenizer.eos_token_id or 1
            )

            generated = model.generate(
                input_ids, max_len=50, start_token=start_token, end_token=end_token
            )

            response = tokenizer.decode(generated[0], skip_special_tokens=True)

        print(f"Sample {i+1}:")
        print(f"Input: {input_text}")
        print(f"Generated Response: {response}")
        print("-" * 40)


def main():
    args = parse_arguments()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    train_dataset, val_dataset = load_datasets(args)

    vocab_size = train_dataset.tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    model = create_model(args, vocab_size)

    if args.model_checkpoint:
        model, _, _ = load_checkpoint(model, None, args.model_checkpoint, device)
    if args.eval_only:
        if not args.model_checkpoint:
            checkpoint_path = MODEL_DIR / "checkpoint.pt"
            if checkpoint_path.exists():
                model, _, _ = load_checkpoint(model, None, str(checkpoint_path), device)
            else:
                print("No checkpoint found for evaluation!")
                return

        evaluate_model(model, train_dataset, train_dataset.tokenizer, device)
        return
    print("\nStarting training...")
    print(f"Training configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Max sequence length: {args.max_seq_len}")

    trained_model = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=device,
        model_path=MODEL_DIR,
    )

    final_model_path = MODEL_DIR / "final_model.pt"
    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "args": vars(args),
            "vocab_size": vocab_size,
        },
        final_model_path,
    )

    print(f"Final model saved to {final_model_path}")

    evaluate_model(trained_model, train_dataset, train_dataset.tokenizer, device)


if __name__ == "__main__":
    main()
