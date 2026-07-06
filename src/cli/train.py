"""
Training and demo commands for the Continual Text Model.
"""

import os
import sys

from src.model.model import ContinualFormer
from src.data.loader import load_data_smart, split_texts

# Default checkpoint path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, 'checkpoint')
DEFAULT_MODEL = os.path.join(CHECKPOINT_DIR, 'model.pt')


def _resolve_model_path(path):
    if path:
        return path
    return DEFAULT_MODEL


def cmd_info(args):
    model_path = _resolve_model_path(args.model)
    model = ContinualFormer.load(model_path) if os.path.exists(model_path) else ContinualFormer()
    model.info()


def cmd_train(args):
    texts = load_data_smart(args)
    import random
    random.shuffle(texts)

    print(f"Loaded {len(texts)} samples")

    model_path = _resolve_model_path(args.model)
    model = ContinualFormer.load(model_path) if os.path.exists(model_path) else ContinualFormer()

    train_t, val_t = split_texts(texts)
    print(f"Train: {len(train_t)}, Val: {len(val_t)}")

    model.train(train_t, task_id=args.task, val_texts=val_t, checkpoint_path=model_path)

    if val_t:
        val_loss = model.evaluate(val_t)
        print(f"\nFinal validation loss: {val_loss:.4f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.save(model_path)


def cmd_predict(args):
    model_path = _resolve_model_path(args.model)
    model = ContinualFormer.load(model_path) if os.path.exists(model_path) else ContinualFormer()

    if args.text:
        output = model.generate(args.text, max_new_tokens=getattr(args, 'max_tokens', 50),
                                temperature=getattr(args, 'temperature', 1.0))
        print(f"Input: {args.text}")
        print(f"Generated: {output}")
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        for t in texts:
            output = model.generate(t, max_new_tokens=getattr(args, 'max_tokens', 50),
                                    temperature=getattr(args, 'temperature', 1.0))
            print(f"  {output}")


def cmd_eval(args):
    texts = load_data_smart(args)
    model_path = _resolve_model_path(args.model)
    model = ContinualFormer.load(model_path) if os.path.exists(model_path) else ContinualFormer()
    val_loss = model.evaluate(texts)
    src = args.hf if hasattr(args, 'hf') and args.hf else args.data
    print(f"Val loss on {src} (task {args.task}): {val_loss:.4f} ({len(texts)} samples)")


def cmd_save(args):
    if not args.model:
        print("Error: --model path required")
        sys.exit(1)
    print(f"Model is auto-saved during training to {args.model}")


def cmd_demo(args):
    """Run a quick demo: train on 2 text tasks, show no forgetting."""
    print("=" * 60)
    print("CONTINUALFORMER — GENERATIVE DEMO")
    print("=" * 60)

    model = ContinualFormer()
    print(f"\nParameters: {model.param_count()}")
    print(f"Device: {model.device}")

    task0_texts = [
        "I love this product it is amazing",
        "This is the best thing ever",
        "I really enjoy using this",
        "Great quality and fast delivery",
        "This is wonderful and beautiful",
        "I am very happy with this purchase",
        "Excellent service and great value",
        "This works perfectly and looks great",
        "I am impressed by the quality",
        "Fantastic experience highly recommend",
        "I hate this it is terrible",
        "This is the worst purchase ever",
        "I dislike this product completely",
        "Bad quality and slow shipping",
        "This is awful and disgusting",
        "I am very disappointed with this",
        "Terrible service and poor value",
        "This broke after one day of use",
        "I regret buying this useless thing",
        "Horrible experience would not recommend",
    ] * 10

    task1_texts = [
        "The computer runs on new software",
        "Python programming is fun and useful",
        "The laptop has a fast processor",
        "Software engineering requires patience",
        "The algorithm processes data quickly",
        "New technology drives innovation forward",
        "The smartphone has a great camera",
        "Programming languages evolve over time",
        "The database stores millions of records",
        "Machine learning models predict outcomes",
    ] * 10

    train0, val0 = split_texts(task0_texts, val_ratio=0.15)
    train1, val1 = split_texts(task1_texts, val_ratio=0.15)

    print(f"\n--- Training Task 0 (sentiment text) ---")
    print(f"Train: {len(train0)}, Val: {len(val0)}")
    model.train(train0, task_id=0, val_texts=val0, verbose=True)

    loss0_before = model.evaluate(task0_texts)
    print(f"Task 0 val loss after training: {loss0_before:.4f}")

    print(f"\n--- Training Task 1 (tech text) ---")
    print(f"Train: {len(train1)}, Val: {len(val1)}")
    model.train(train1, task_id=1, val_texts=val1, verbose=True)

    loss1 = model.evaluate(task1_texts)
    loss0_after = model.evaluate(task0_texts)
    print(f"\n--- Results ---")
    print(f"Task 0 val loss: {loss0_before:.4f} -> {loss0_after:.4f} (forgetting: {loss0_after - loss0_before:.4f})")
    print(f"Task 1 val loss: {loss1:.4f}")

    print(f"\n--- Generation ---")
    for prompt in ["I love", "This is terrible", "The computer", "Python is"]:
        output = model.generate(prompt, max_new_tokens=20, temperature=0.7)
        print(f"  '{prompt}' -> {output}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    demo_path = os.path.join(CHECKPOINT_DIR, 'demo_model.pt')
    model.save(demo_path)
    print(f"\nDemo model saved to {demo_path}")
