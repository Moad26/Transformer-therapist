# Transformer for Empathetic Dialogue

A from-scratch PyTorch implementation of a transformer model trained on the EmpatheticDialogues dataset. This is an educational project demonstrating sequence-to-sequence learning for conversational AI.

## What This Is

This project implements a custom transformer architecture (encoder-decoder) trained to generate empathetic responses in conversations. The model learns from the EmpatheticDialogues dataset, where one speaker shares an emotional situation and another provides supportive responses.

**Important**: This is a learning project and research experiment, not a production system or therapeutic tool. The dataset contains crowdsourced empathetic conversations, not actual therapeutic dialogues.

## Project Structure

```
.
├── src/
│   ├── data.py            # Dataset loading and preprocessing
│   ├── main.py            # Training script with CLI
│   ├── model.py           # Transformer implementation
│   └── train.py           # Training loop
├── input/
│   └── empatheticdialogues/  # Dataset files (train/valid/test.csv)
├── model/                 # Saved checkpoints
├── setup.sh              # Automated setup script
├── pyproject.toml        # Dependencies (uv)
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager (installed automatically by setup script)

### Installation

Run the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

This will:

1. Install uv if not present
2. Create virtual environment and install dependencies
3. Download and extract the EmpatheticDialogues dataset
4. Create necessary directories

Activate the environment:

```bash
source .venv/bin/activate
```

## Usage

### Training

Basic training with defaults:

```bash
cd src
python main.py
```

Custom configuration:

```bash
python main.py \
  --batch_size 32 \
  --learning_rate 5e-5 \
  --num_epochs 100 \
  --embed_dim 768 \
  --num_head 8 \
  --num_layers 6
```

### Arguments

**Data Parameters:**

- `--max_seq_len`: Maximum sequence length (default: 128)
- `--tokenizer_name`: HuggingFace tokenizer (default: facebook/blenderbot-400M-distill)

**Model Parameters:**

- `--embed_dim`: Embedding dimension (default: 512)
- `--num_head`: Number of attention heads (default: 8)
- `--num_layers`: Transformer layers (default: 6)
- `--dropout`: Dropout rate (default: 0.1)

**Training Parameters:**

- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_epochs`: Training epochs (default: 50)

**System:**

- `--device`: Device selection: auto/cuda/cpu (default: auto)
- `--seed`: Random seed (default: 42)

### Evaluation

Evaluate a trained model:

```bash
python main.py --eval_only --model_checkpoint ../model/final_model.pt
```

Or use the latest checkpoint:

```bash
python main.py --eval_only
```

## Architecture

### Custom Transformer Implementation

This is a **from-scratch implementation** built for educational purposes, not using pre-built transformer layers.

**Key Components:**

- **Encoder**: Multi-head self-attention + feed-forward networks with layer normalization
- **Decoder**: Masked multi-head attention + cross-attention + feed-forward networks
- **Positional Encoding**: Sinusoidal position embeddings
- **Multi-Head Attention**: Custom implementation that uses independent attention modules per head (differs from standard implementation which splits dimensions)

**Architecture Decision**: The multi-head attention uses separate attention modules rather than dimension-splitting. This increases parameters but allows each head to operate over the full embedding space, potentially capturing richer representations at the cost of efficiency.

### Training Details

- **Loss**: Cross-entropy with padding token masking (-100)
- **Optimizer**: Adam
- **Early Stopping**: Monitors validation loss with configurable patience
- **Checkpointing**: Saves best model based on validation performance

## Dataset

The [EmpatheticDialogues dataset](https://github.com/facebookresearch/EmpatheticDialogues) contains ~25k conversations where:

- One person (speaker) describes an emotional situation
- Another person (listener) responds empathetically

The dataset covers 32 emotion categories. This is **not** clinical therapeutic data - it's crowdsourced conversations with emotional context.

### Data Processing

1. Conversations are split into alternating speaker-listener pairs
2. Speaker utterances become model inputs
3. Listener responses become target outputs
4. Tokenization uses BlenderBot tokenizer
5. Sequences are padded/truncated to max_seq_len

## Limitations

**This is a learning project with significant limitations:**

- Trained from scratch on limited data (~25k examples)
- No comparison against pre-trained baselines
- Evaluation only shows sample outputs (no quantitative metrics like BLEU/perplexity)
- Simple greedy decoding (no beam search or sampling)
- Custom architecture is less efficient than standard implementations
- No safety filtering or content moderation
- Not suitable for real-world deployment

## Development

### Dependencies

Managed via `pyproject.toml` with uv:

```toml
requires-python = ">=3.10"
dependencies = [
  "torch>=2.8.0",
  "transformers>=4.56.2",
  "pandas>=2.3.2",
  "numpy>=2.2.6",
  "tqdm>=4.67.1",
  ...
]
```

### Adding Dependencies

```bash
uv add <package-name>
```

## Acknowledgments

- EmpatheticDialogues dataset by Facebook AI Research
- Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017)
