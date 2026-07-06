"""
Data loading utilities for the Continual Text Model.
Supports CSV, TXT, and HuggingFace Hub datasets.
"""

import sys
import os
import csv


# Root directory of the project (parent of src/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')


def load_hf_dataset(dataset_name, split='train', text_col=None,
                     subset=None, max_samples=None, limit=None):
    """Load a dataset from HuggingFace Hub (text only, for language modeling).

    Args:
        dataset_name: HF dataset name in 'user/dataset_name' format (required)
        split: Which split to load ('train', 'test', 'validation')
        text_col: Column name containing text (auto-detect if None)
        subset: Subset/config name for datasets with multiple configs
        max_samples: Limit number of samples loaded
        limit: Max number of samples to keep

    Returns:
        list of text strings
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed. Install with: pip install datasets")
        sys.exit(1)

    print(f"Loading HF dataset: {dataset_name} (split={split})")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Auto-detect config name if not provided
    if not subset:
        try:
            from datasets import get_dataset_config_names
            configs = get_dataset_config_names(dataset_name, cache_dir=DATA_DIR)
            if configs and configs != ['default']:
                subset = 'main' if 'main' in configs else configs[0]
                print(f"  Auto-selected config: '{subset}' (available: {configs})")
        except Exception:
            pass

    # Load with split fallback
    try:
        if subset:
            ds = load_dataset(dataset_name, subset, split=split, cache_dir=DATA_DIR)
        else:
            ds = load_dataset(dataset_name, split=split, cache_dir=DATA_DIR)
    except ValueError as e:
        if 'Unknown split' in str(e) or 'Should be one of' in str(e):
            if subset:
                all_ds = load_dataset(dataset_name, subset, cache_dir=DATA_DIR)
            else:
                all_ds = load_dataset(dataset_name, cache_dir=DATA_DIR)
            available = list(all_ds.keys())
            print(f"  Split '{split}' not available. Using '{available[0]}' (available: {available})")
            ds = all_ds[available[0]]
        else:
            raise

    print(f"  Loaded {len(ds)} rows. Columns: {ds.column_names}")

    # Auto-detect text column
    if text_col is None:
        for candidate in ['text', 'sentence', 'review', 'content', 'tweet', 'prompt', 'question', 'body', 'answer', 'definition', 'word']:
            if candidate in ds.column_names:
                text_col = candidate
                break
        if text_col is None:
            for col in ds.column_names:
                if isinstance(ds[0][col], str):
                    text_col = col
                    break
        if text_col is None:
            print(f"Error: Could not auto-detect text column. Available: {ds.column_names}")
            print("Specify with --text-col <column_name>")
            sys.exit(1)

    print(f"  Text column: '{text_col}'")

    all_texts = [str(t) for t in ds[text_col]]

    import random as _rng
    _rng.shuffle(all_texts)
    if max_samples:
        all_texts = all_texts[:max_samples]
    if limit:
        all_texts = all_texts[:limit]

    print(f"  Extracted {len(all_texts)} samples.")

    return all_texts


def load_csv(path):
    """Load a CSV file — returns list of text strings."""
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text', row.get('Text', row.get('sentence', '')))
            texts.append(text)
    return texts


def load_txt(path):
    """Load a text file — one text per line."""
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(line)
    return texts


def load_data(path):
    """Auto-detect format and load from file path."""
    if path.endswith('.csv'):
        return load_csv(path)
    else:
        return load_txt(path)


def load_data_smart(args):
    """Load data from either file path or HuggingFace Hub.

    Args must have one of: --data (file path) or --hf (HF dataset name)
    Optional: --text-col, --split, --subset, --max-samples, --limit
    """
    if hasattr(args, 'hf') and args.hf:
        return load_hf_dataset(
            dataset_name=args.hf,
            split=getattr(args, 'split', 'train'),
            text_col=getattr(args, 'text_col', None),
            subset=getattr(args, 'subset', None),
            max_samples=getattr(args, 'max_samples', None),
            limit=getattr(args, 'limit', None),
        )
    elif hasattr(args, 'data') and args.data:
        return load_data(args.data)
    else:
        print("Error: must specify either --data <file> or --hf <dataset_name>")
        sys.exit(1)


def split_texts(texts, val_ratio=0.2):
    """Split texts into train/val."""
    import random
    indices = list(range(len(texts)))
    random.shuffle(indices)
    n_val = int(len(texts) * val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_t = [texts[i] for i in train_idx]
    val_t = [texts[i] for i in val_idx]
    return train_t, val_t
