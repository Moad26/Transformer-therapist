# Transformer-Based Therapeutic Dialogue System

A PyTorch implementation of a transformer model designed for empathetic dialogue generation, specifically trained on therapeutic conversations to provide supportive and empathetic responses.

## Project Overview

This project implements a sequence-to-sequence transformer model that learns to generate empathetic responses similar to a therapist. The model is trained on the EmpatheticDialogues dataset, which contains conversations grounded in emotional situations.

### Key Features

- Custom Transformer Architecture: Built from scratch with encoder-decoder structure
- Empathetic Response Generation: Trained on therapeutic dialogue patterns
- Flexible Training Pipeline: Configurable hyperparameters and training options
- Cross-platform Support: Compatible with Windows, Linux, and macOS
- Comprehensive Evaluation: Built-in model evaluation and sample generation

## Project Structure

```
transformer-therapist/
├── src/                    # Source code
│   ├── data.py            # Dataset loading and preprocessing
│   ├── main.py            # Main training script with CLI
│   ├── model.py           # Transformer model implementation
│   ├── train.py           # Training loop and utilities
│   └── test.ipynb         # Jupyter notebook for testing
├── input/                 # Data directory
│   └── empatheticdialogues/
│       ├── train.csv      # Training data
│       ├── valid.csv      # Validation data
│       └── test.csv       # Test data
├── model/                 # Model checkpoints and saved models
├── info/                  # Documentation and analysis
├── visualisation/         # Training visualizations and plots
├── make.py               # Project setup and management script
└── requirements.txt      # Python dependencies
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Transformers library
- CUDA (optional, for GPU training)

### Setup Process

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/Moad26/Transformer-therapist.git
cd transformer-therapist
```

2. Set up the project environment:

```bash
python make.py setup
```

This command creates necessary directories, sets up a virtual environment, and installs all dependencies from requirements.txt.

3. Download the dataset:

```bash
python make.py dataset
```

Downloads and extracts the EmpatheticDialogues dataset automatically.

4. Activate the virtual environment:

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

## Usage

### Training the Model

Basic training with default parameters:

```bash
cd src
python main.py
```

Training with custom parameters:

```bash
python main.py --batch_size 32 --learning_rate 5e-5 --num_epochs 100 --embed_dim 768
```

### Training Parameters

- `--batch_size`: Training batch size (default: 16)
- `--learning_rate`: Learning rate for optimizer (default: 1e-4)
- `--num_epochs`: Number of training epochs (default: 50)
- `--embed_dim`: Model embedding dimension (default: 512)
- `--num_head`: Number of attention heads (default: 8)
- `--num_layers`: Number of transformer layers (default: 6)
- `--max_seq_len`: Maximum sequence length (default: 128)
- `--device`: Computing device (auto, cuda, cpu)
- `--seed`: Random seed for reproducibility (default: 42)

### Model Evaluation

Evaluate a trained model:

```bash
python main.py --eval_only --model_checkpoint ../model/final_model.pt
```

Evaluate using the latest checkpoint:

```bash
python main.py --eval_only
```

## Architecture

### Model Components

The transformer architecture consists of:

- **Encoder**: Processes input patient statements using multi-head attention and feed-forward layers
- **Decoder**: Generates empathetic therapist responses with causal masking for autoregressive generation
- **Multi-Head Attention**: Captures different aspects of emotional context across multiple attention heads
- **Positional Encoding**: Maintains sequence order information using sinusoidal encoding
- **Feed-Forward Networks**: Applies non-linear transformations with ReLU activation

### Technical Details

- Layer normalization and residual connections for stable training
- Configurable number of layers and attention heads
- Dropout regularization to prevent overfitting
- Cross-entropy loss with label smoothing
- Adam optimizer with configurable learning rate

### Dataset Processing

The EmpatheticDialogues dataset contains over 25,000 conversations grounded in emotional situations. The preprocessing pipeline:

- Extracts patient-therapist dialogue pairs from conversation data
- Applies tokenization using the BlenderBot tokenizer
- Handles special tokens for sequence boundaries
- Implements padding and truncation for uniform sequence lengths

## Training Process

### Training Configuration

- **Loss Function**: Cross-entropy loss with ignored padding tokens
- **Optimizer**: Adam optimizer with configurable learning rate
- **Early Stopping**: Prevents overfitting using validation loss monitoring
- **Checkpointing**: Automatic model saving at best validation performance

### Monitoring and Evaluation

- Real-time training and validation loss tracking
- Progress bars showing training metrics
- Sample generation during evaluation phases
- Automatic checkpoint management

## Development Tools

### Make Script Commands

```bash
python make.py setup      # Complete project setup
python make.py install    # Install dependencies only
python make.py dataset    # Download dataset
python make.py clean      # Remove build artifacts
python make.py help       # Display help information
```

### Project Management

The make script handles:

- Virtual environment creation and management
- Directory structure initialization
- Dataset download and extraction
- Dependency installation and verification

## Customization

### Model Configuration

Modify the model architecture by adjusting parameters:

- Embedding dimensions for representation capacity
- Number of attention heads for multi-aspect learning
- Transformer layer depth for model complexity
- Sequence length limits for memory efficiency

### Training Configuration

Customize training behavior:

- Batch sizes for memory and convergence trade-offs
- Learning rates for optimization stability
- Early stopping patience for overfitting prevention
- Device selection for computational resources

### Data Processing

Extend the data processing pipeline:

- Alternative tokenization strategies
- Custom preprocessing methods
- Different conversation formatting approaches
- Additional data augmentation techniques

## Expected Results

The model generates responses designed to be:

- **Empathetic**: Demonstrating understanding of emotional context
- **Supportive**: Providing helpful and constructive guidance
- **Contextually Appropriate**: Matching the tone and content of conversations
- **Therapeutically Informed**: Following principles of supportive dialogue

### Example Interaction

```
Input: "I've been feeling really anxious about work lately"
Generated Response: "It sounds like work has been weighing heavily on your mind. Can you tell me more about what specifically is making you feel anxious?"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with appropriate tests
4. Ensure code follows existing style conventions
5. Submit a pull request with detailed description

## Technical Requirements

- Python 3.8+
- PyTorch 1.8+
- Transformers library
- Additional dependencies listed in requirements.txt

## Acknowledgments

This project builds upon:

- The EmpatheticDialogues dataset from Facebook AI Research
- The Transformer architecture from "Attention Is All You Need"
- PyTorch deep learning framework
- Hugging Face transformers library
