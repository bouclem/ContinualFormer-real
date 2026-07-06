"""
Chat / Prediction commands for the Continual Text Model
========================================================
Usage:
  python cli.py chat predict --text "hello world" --task 0 --model mymodel.pt
  python cli.py chat predict --file input.txt --task 0 --model mymodel.pt
  python cli.py chat chat --task 0 --model mymodel.pt
"""

import sys
import os

from src.model.model import ContinualFormer

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, 'checkpoint')
DEFAULT_MODEL = os.path.join(CHECKPOINT_DIR, 'model.pt')


def _load_model(args):
    model_path = args.model if args.model else DEFAULT_MODEL
    if os.path.exists(model_path):
        return ContinualFormer.load(model_path)
    return ContinualFormer()


def cmd_predict(args):
    model = _load_model(args)

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


def cmd_chat(args):
    """Interactive chat mode — type text and get predictions in real time."""
    model = _load_model(args)

    print("=" * 60)
    print("CONTINUAL TEXT MODEL -- CHAT MODE")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Model: {args.model or '(untrained)'}")
    print(f"Classes per task: {model.CLASSES_PER_TASK}")
    print()
    print("Type text to classify. Commands:")
    print("  :quit   -- exit chat")
    print("  :task N -- switch to task N")
    print("  :probs  -- toggle showing probabilities")
    print("  :info   -- show model info")
    print()

    show_probs = True
    task_id = args.task

    while True:
        try:
            text = input(f"[task {task_id}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not text:
            continue

        if text.startswith(':'):
            cmd = text.lower()
            if cmd == ':quit' or cmd == ':q' or cmd == ':exit':
                print("Bye!")
                break
            elif cmd.startswith(':task'):
                parts = cmd.split()
                if len(parts) >= 2:
                    try:
                        task_id = int(parts[1])
                        print(f"Switched to task {task_id}")
                    except ValueError:
                        print("Invalid task ID")
                else:
                    print(f"Current task: {task_id}")
            elif cmd == ':probs':
                show_probs = not show_probs
                print(f"Probabilities: {'ON' if show_probs else 'OFF'}")
            elif cmd == ':info':
                model.info()
            else:
                print(f"Unknown command: {text}")
            continue

        pred = model.predict(text, task_id=task_id)
        print(f"  -> Label: {pred}")

        if show_probs:
            probs = model.predict_proba(text, task_id=task_id)
            prob_str = "  ".join(f"[{i}]={p:.3f}" for i, p in enumerate(probs))
            print(f"     {prob_str}")
