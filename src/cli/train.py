"""
Training and demo commands for the Continual Text Model.
"""

import os
import sys

from src.model.model import ContinualFormer
from src.data.loader import load_data_smart, split_data

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
    texts, labels = load_data_smart(args)
    combined = list(zip(texts, labels))
    import random
    random.shuffle(combined)
    texts = [t for t, l in combined]
    labels = [l for t, l in combined]

    print(f"Loaded {len(texts)} samples")
    print(f"Labels: {sorted(set(labels))}")

    model_path = _resolve_model_path(args.model)
    model = ContinualFormer.load(model_path) if os.path.exists(model_path) else ContinualFormer()

    train_t, train_l, val_t, val_l = split_data(texts, labels)
    print(f"Train: {len(train_t)}, Val: {len(val_t)}")

    model.train(train_t, train_l, task_id=args.task, val_texts=val_t, val_labels=val_l, checkpoint_path=model_path)

    if val_t:
        acc = model.evaluate(val_t, val_l, args.task)
        print(f"\nFinal validation accuracy: {acc:.4f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.save(model_path)


def cmd_predict(args):
    model_path = _resolve_model_path(args.model)
    model = ContinualFormer.load(model_path) if os.path.exists(model_path) else ContinualFormer()

    if args.text:
        pred = model.predict(args.text, task_id=args.task)
        probs = model.predict_proba(args.text, task_id=args.task)
        print(f"Input: {args.text}")
        print(f"Predicted label: {pred}")
        print(f"Probabilities: {[f'{p:.3f}' for p in probs]}")
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        preds = model.predict(texts, task_id=args.task)
        for t, p in zip(texts, preds):
            print(f"  {p}  {t[:80]}")


def cmd_eval(args):
    texts, labels = load_data_smart(args)
    model_path = _resolve_model_path(args.model)
    model = ContinualFormer.load(model_path) if os.path.exists(model_path) else ContinualFormer()
    acc = model.evaluate(texts, labels, args.task)
    src = args.hf if hasattr(args, 'hf') and args.hf else args.data
    print(f"Accuracy on {src} (task {args.task}): {acc:.4f} ({len(texts)} samples)")


def cmd_save(args):
    if not args.model:
        print("Error: --model path required")
        sys.exit(1)
    print(f"Model is auto-saved during training to {args.model}")


def cmd_demo(args):
    """Run a quick demo: train on 2 text tasks, show no forgetting."""
    print("=" * 60)
    print("CONTINUAL TEXT MODEL — DEMO")
    print("=" * 60)

    model = ContinualFormer()
    print(f"\nParameters: {model.param_count()} (grows to ~81K as tasks are added)")
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

    task0_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 10

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
        "The pizza tastes delicious and cheesy",
        "I love eating sushi and ramen",
        "The cake is sweet and fluffy",
        "The restaurant serves great burgers",
        "Cooking pasta is easy and fun",
        "Fresh ingredients make better meals",
        "The bakery sells amazing croissants",
        "I enjoy trying new food recipes",
        "The coffee shop has great pastries",
        "Dinner at that place was wonderful",
        "The team won the football match",
        "Basketball is an exciting sport",
        "The runner broke the world record",
        "Tennis requires skill and focus",
        "Swimming is good for your health",
        "The soccer game was intense today",
        "He trained hard for the marathon",
        "The boxer knocked out his opponent",
        "Yoga improves flexibility and strength",
        "The hockey team made the playoffs",
    ] * 5

    task1_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] * 5

    train0_t, train0_l, val0_t, val0_l = split_data(task0_texts, task0_labels, val_ratio=0.15)
    train1_t, train1_l, val1_t, val1_l = split_data(task1_texts, task1_labels, val_ratio=0.15)

    print(f"\n--- Training Task 0 (sentiment) ---")
    print(f"Train: {len(train0_t)}, Val: {len(val0_t)}")
    model.train(train0_t, train0_l, task_id=0, val_texts=val0_t, val_labels=val0_l, verbose=True)

    acc0_before = model.evaluate(task0_texts, task0_labels, task_id=0)
    print(f"Task 0 accuracy after training: {acc0_before:.4f}")

    print(f"\n--- Training Task 1 (topics) ---")
    print(f"Train: {len(train1_t)}, Val: {len(val1_t)}")
    model.train(train1_t, train1_l, task_id=1, val_texts=val1_t, val_labels=val1_l, verbose=True)

    acc1 = model.evaluate(task1_texts, task1_labels, task_id=1)
    acc0_after = model.evaluate(task0_texts, task0_labels, task_id=0)
    print(f"\n--- Results ---")
    print(f"Task 0 accuracy: {acc0_before:.4f} → {acc0_after:.4f} (forgetting: {acc0_before - acc0_after:.4f})")
    print(f"Task 1 accuracy: {acc1:.4f}")

    print(f"\n--- Predictions ---")
    test1 = "I really love this amazing product"
    test2 = "This is terrible and I hate it"
    test3 = "The computer runs Python fast"
    test4 = "The pizza is delicious and cheesy"

    print(f"  '{test1}' → sentiment: {model.predict(test1, task_id=0)}")
    print(f"  '{test2}' → sentiment: {model.predict(test2, task_id=0)}")
    print(f"  '{test3}' → topic: {model.predict(test3, task_id=1)}")
    print(f"  '{test4}' → topic: {model.predict(test4, task_id=1)}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    demo_path = os.path.join(CHECKPOINT_DIR, 'demo_model.pt')
    model.save(demo_path)
    print(f"\nDemo model saved to {demo_path}")
