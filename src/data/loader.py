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


def load_hf_dataset(dataset_name, split='train', text_col=None, label_col=None,
                     subset=None, max_samples=None, limit=None):
    """Load a dataset from HuggingFace Hub.

    Args:
        dataset_name: HF dataset name in 'user/dataset_name' format (required)
        split: Which split to load ('train', 'test', 'validation')
        text_col: Column name containing text (auto-detect if None)
        label_col: Column name containing labels (auto-detect if None)
        subset: Subset/config name for datasets with multiple configs
        max_samples: Limit number of samples loaded

    Returns:
        (texts, labels) lists
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
        for candidate in ['text', 'sentence', 'review', 'content', 'tweet', 'prompt', 'question', 'body']:
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

    # Auto-detect label column
    if label_col is None:
        for candidate in ['label', 'labels', 'category', 'class', 'target', 'rating', 'sentiment']:
            if candidate in ds.column_names:
                label_col = candidate
                break
        if label_col is None:
            for col in ds.column_names:
                if col != text_col and not isinstance(ds[0][col], str):
                    label_col = col
                    break
        if label_col is None:
            for col in ds.column_names:
                if col != text_col:
                    label_col = col
                    break
        if label_col is None:
            print(f"Error: Could not auto-detect label column. Available: {ds.column_names}")
            print("Specify with --label-col <column_name>")
            sys.exit(1)

    print(f"  Text column: '{text_col}', Label column: '{label_col}'")

    import random as _rng
    all_indices = list(range(len(ds)))
    _rng.shuffle(all_indices)
    if max_samples:
        all_indices = all_indices[:max_samples * 3]

    texts = []
    labels = []
    for idx in all_indices:
        if max_samples and len(texts) >= max_samples:
            break
        row = ds[idx]
        text = str(row[text_col])
        label = row[label_col]
        if hasattr(label, '__class__') and 'ClassLabel' in str(type(label)):
            label = int(label)
        elif isinstance(label, str):
            try:
                label = int(label)
            except ValueError:
                if not hasattr(load_hf_dataset, '_label_map'):
                    load_hf_dataset._label_map = {}
                    load_hf_dataset._next_id = 0
                if label not in load_hf_dataset._label_map:
                    load_hf_dataset._label_map[label] = load_hf_dataset._next_id
                    load_hf_dataset._next_id += 1
                label = load_hf_dataset._label_map[label]
        else:
            label = int(label)

        if limit is not None and (label < 0 or label >= limit):
            continue

        texts.append(text)
        labels.append(label)

    unique_labels = sorted(set(labels))
    if len(unique_labels) <= 20:
        print(f"  Extracted {len(texts)} samples. Labels: {unique_labels}")
    else:
        print(f"  Extracted {len(texts)} samples. {len(unique_labels)} labels: {unique_labels[:5]}...{unique_labels[-5:]}")
    if hasattr(load_hf_dataset, '_label_map') and load_hf_dataset._label_map:
        lm = load_hf_dataset._label_map
        preview = dict(list(lm.items())[:5])
        print(f"  Label mapping: {len(lm)} classes. First 5: {preview}")

    return texts, labels


def load_csv(path):
    """Load a CSV file with 'text' and 'label' columns."""
    texts, labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text', row.get('Text', row.get('sentence', '')))
            label = int(row.get('label', row.get('Label', row.get('category', 0))))
            texts.append(text)
            labels.append(label)
    return texts, labels


def load_txt(path):
    """Load a text file with 'text||label' format."""
    texts, labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '||' not in line:
                continue
            parts = line.rsplit('||', 1)
            text = parts[0].strip().strip('"')
            label = int(parts[1].strip())
            texts.append(text)
            labels.append(label)
    return texts, labels


def load_data(path):
    """Auto-detect format and load from file path."""
    if path.endswith('.csv'):
        return load_csv(path)
    else:
        return load_txt(path)


def load_data_smart(args):
    """Load data from either file path or HuggingFace Hub.

    Args must have one of: --data (file path) or --hf (HF dataset name)
    Optional: --text-col, --label-col, --split, --subset, --max-samples
    """
    if hasattr(args, 'hf') and args.hf:
        return load_hf_dataset(
            dataset_name=args.hf,
            split=getattr(args, 'split', 'train'),
            text_col=getattr(args, 'text_col', None),
            label_col=getattr(args, 'label_col', None),
            subset=getattr(args, 'subset', None),
            max_samples=getattr(args, 'max_samples', None),
            limit=getattr(args, 'limit', None),
        )
    elif hasattr(args, 'data') and args.data:
        return load_data(args.data)
    else:
        print("Error: must specify either --data <file> or --hf <dataset_name>")
        sys.exit(1)


def split_data(texts, labels, val_ratio=0.2):
    """Split into train/val."""
    import random
    indices = list(range(len(texts)))
    random.shuffle(indices)
    n_val = int(len(texts) * val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_t = [texts[i] for i in train_idx]
    train_l = [labels[i] for i in train_idx]
    val_t = [texts[i] for i in val_idx]
    val_l = [labels[i] for i in val_idx]
    return train_t, train_l, val_t, val_l
