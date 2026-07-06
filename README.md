# ContinualFormer

O(1) Transformer with **Neuron Freezing** and **Dynamic Expansion** for continual learning.

~80K parameters. Zero forgetting across tasks. No growing memory replay buffer.

## Architecture

```
ContinualFormer/
├── cli.py                 # Root CLI entry point (thin dispatcher)
├── src/
│   ├── model/
│   │   ├── config.py      # Constants & hyperparameters
│   │   ├── tokenizer.py   # Text normalization & vocab building
│   │   ├── attention.py   # LinearAttention (O(1) via kernelized attention)
│   │   ├── ffn.py         # FreezableFFN (neuron-level freezing)
│   │   ├── layer.py       # TransformerLayer (attention + FFN + norms)
│   │   └── model.py       # ContinualFormer (links everything, train/eval/save/load)
│   ├── data/
│   │   └── loader.py      # CSV, TXT, HuggingFace dataset loading + split
│   ├── cli/
│   │   ├── train.py       # train, predict, eval, demo, info, save commands
│   │   ├── bench.py       # GSM8k benchmark + matplotlib comparison plots
│   │   └── chat.py        # Interactive chat / prediction mode
│   └── tests/
│       └── test_model.py  # Neuron freezing & expansion tests
├── data/                  # HF dataset cache + local data files
├── plots/                 # Benchmark comparison charts (.png)
└── checkpoint/            # Model .pt files
```

## Key Features

- **Linear Attention**: O(1) inference cost via kernelized attention (ELU + 1)
- **Neuron Freezing**: After each task, important FFN neurons are frozen to prevent forgetting
- **Dynamic Expansion**: When too many neurons are frozen, FFN expands with new neurons
- **Per-task Heads**: Each task gets its own classification head
- **Continual Vocab**: Vocabulary grows incrementally across tasks

## Installation

```bash
pip install torch numpy
# Optional: for HuggingFace datasets
pip install datasets
# Optional: for benchmark plots
pip install matplotlib
```

## Usage

### Train on a new task

```bash
# From CSV
python cli.py train --data dataset.csv --task 0

# From HuggingFace Hub
python cli.py train --hf user/dataset_name --task 0

# Continue from existing checkpoint
python cli.py train --data dataset.csv --task 1 --model checkpoint/model.pt
```

### Predict

```bash
python cli.py predict --text "hello world" --task 0 --model checkpoint/model.pt
python cli.py predict --file input.txt --task 0 --model checkpoint/model.pt
```

### Evaluate

```bash
python cli.py eval --data test.csv --task 0 --model checkpoint/model.pt
python cli.py eval --hf user/dataset_name --task 0 --model checkpoint/model.pt
```

### Interactive chat

```bash
python cli.py chat chat --task 0 --model checkpoint/model.pt
```

### Benchmark (GSM8k)

```bash
# Evaluate on GSM8k
python cli.py bench gsm8k --model checkpoint/model.pt --max-samples 500

# Train then evaluate
python cli.py bench gsm8k --train --max-samples 1000

# Generate comparison plots
python cli.py bench plot

# List all results
python cli.py bench list
```

### Demo

```bash
python cli.py demo
```

### Tests

```bash
python cli.py test
```

### Model info

```bash
python cli.py info --model checkpoint/model.pt
```

## CSV Format

```csv
text,label
"This is some text",0
"Another example",1
```

## TXT Format

```
text||label
"This is some text"||0
```

## Config

Hyperparameters are defined in `src/model/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DIM` | 64 | Embedding / hidden dim |
| `FFN_HIDDEN` | 128 | FFN hidden dim (initial) |
| `HEADS` | 2 | Attention heads |
| `LAYERS` | 2 | Transformer layers |
| `CLASSES_PER_TASK` | 10 | Output classes per task |
| `VOCAB_SIZE` | 250 | Max vocabulary size |
| `MAX_LEN` | 48 | Max sequence length |
| `FREEZE_RATIO` | 0.3 | Fraction of neurons frozen after each task |
| `EXPAND_THRESHOLD` | 0.2 | Active ratio below which FFN expands |
| `EXPAND_RATIO` | 0.2 | Fraction of new neurons added on expansion |
| `EPOCHS` | 50 | Training epochs per task |
| `LR` | 0.001 | Learning rate |
| `BATCH_SIZE` | 64 | Batch size |
| `DROPOUT` | 0.1 | Dropout rate |
