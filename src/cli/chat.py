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


def cmd_chat(args):
    """Interactive chat mode — type text and get generations in real time."""
    model = _load_model(args)

    print("=" * 60)
    print("CONTINUALFORMER — CHAT MODE")
    print("=" * 60)
    print(f"Model: {args.model or '(untrained)'}")
    print(f"Params: {model.param_count()}")
    print()
    print("Type a prompt to generate text. Commands:")
    print("  :quit   -- exit chat")
    print("  :info   -- show model info")
    print("  :temp T -- set temperature (0 = greedy, 1 = creative)")
    print()

    temperature = getattr(args, 'temperature', 1.0)
    max_tokens = getattr(args, 'max_tokens', 50)

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not text:
            continue

        if text.startswith(':'):
            cmd = text.lower()
            if cmd in (':quit', ':q', ':exit'):
                print("Bye!")
                break
            elif cmd.startswith(':temp'):
                parts = cmd.split()
                if len(parts) >= 2:
                    try:
                        temperature = float(parts[1])
                        print(f"Temperature set to {temperature}")
                    except ValueError:
                        print("Invalid temperature")
                else:
                    print(f"Current temperature: {temperature}")
            elif cmd == ':info':
                model.info()
            else:
                print(f"Unknown command: {text}")
            continue

        output = model.generate(text, max_new_tokens=max_tokens, temperature=temperature)
        print(f"  {output}")
