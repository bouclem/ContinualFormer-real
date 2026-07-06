"""
CLI for the Continual Text Model
=================================
Usage:
  python cli.py info
  python cli.py train --data dataset.csv --task 0
  python cli.py train --hf user/dataset_name --task 0
  python cli.py train --hf user/dataset_name --text-col sentence --label-col label --task 0
  python cli.py train --data dataset.csv --task 0 --model checkpoint/mymodel.pt
  python cli.py predict --text "hello world" --task 0 --model checkpoint/mymodel.pt
  python cli.py predict --file input.txt --task 0 --model checkpoint/mymodel.pt
  python cli.py eval --data test.csv --task 0 --model checkpoint/mymodel.pt
  python cli.py eval --hf user/dataset_name --task 0 --model checkpoint/mymodel.pt
  python cli.py save --model checkpoint/mymodel.pt
  python cli.py demo
  python cli.py bench gsm8k --model checkpoint/mymodel.pt --max-samples 500
  python cli.py bench gsm8k --train --max-samples 1000
  python cli.py bench plot
  python cli.py bench list
  python cli.py chat predict --text "hello world" --task 0 --model checkpoint/mymodel.pt
  python cli.py chat chat --task 0 --model checkpoint/mymodel.pt
  python cli.py test

CSV format:
  text,label
  "This is some text",0
  "Another example",1

Or plain text file (one per line):
  text||label
  "This is some text"||0

HuggingFace format:
  --hf user/dataset_name          (loads from HuggingFace Hub)
  --hf user/dataset_name --split train
  --hf user/dataset_name --text-col text --label-col label
"""

import argparse
import sys
import os

# Ensure project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="ContinualFormer — O(1) Transformer with Neuron Freezing")
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # info
    p_info = subparsers.add_parser('info', help='Show model info')
    p_info.add_argument('--model', default=None, help='Path to .pt file')

    # train
    p_train = subparsers.add_parser('train', help='Train on a new task')
    p_train.add_argument('--data', default=None, help='Path to CSV or TXT dataset')
    p_train.add_argument('--hf', default=None, help='HuggingFace dataset name (e.g. user/dataset_name)')
    p_train.add_argument('--task', type=int, required=True, help='Task ID (0-indexed)')
    p_train.add_argument('--model', default=None, help='Path to existing .pt to continue from')
    p_train.add_argument('--text-col', default=None, help='Text column name (auto-detect if omitted)')
    p_train.add_argument('--label-col', default=None, help='Label column name (auto-detect if omitted)')
    p_train.add_argument('--split', default='train', help='HF dataset split (train, test, validation)')
    p_train.add_argument('--subset', default=None, help='HF dataset subset/config name')
    p_train.add_argument('--max-samples', type=int, default=None, help='Limit number of samples')
    p_train.add_argument('--limit', type=int, default=None, help='Max label classes to keep (default: no limit)')

    # predict
    p_pred = subparsers.add_parser('predict', help='Predict labels')
    p_pred.add_argument('--text', default=None, help='Text to classify')
    p_pred.add_argument('--file', default=None, help='File with one text per line')
    p_pred.add_argument('--task', type=int, required=True, help='Task ID')
    p_pred.add_argument('--model', default=None, help='Path to .pt file')

    # eval
    p_eval = subparsers.add_parser('eval', help='Evaluate accuracy')
    p_eval.add_argument('--data', default=None, help='Path to CSV or TXT dataset')
    p_eval.add_argument('--hf', default=None, help='HuggingFace dataset name (e.g. user/dataset_name)')
    p_eval.add_argument('--task', type=int, required=True, help='Task ID')
    p_eval.add_argument('--model', default=None, help='Path to .pt file')
    p_eval.add_argument('--text-col', default=None, help='Text column name (auto-detect if omitted)')
    p_eval.add_argument('--label-col', default=None, help='Label column name (auto-detect if omitted)')
    p_eval.add_argument('--split', default='test', help='HF dataset split')
    p_eval.add_argument('--subset', default=None, help='HF dataset subset/config name')
    p_eval.add_argument('--max-samples', type=int, default=None, help='Limit number of samples')
    p_eval.add_argument('--limit', type=int, default=None, help='Max label classes to keep (default: no limit)')

    # save
    p_save = subparsers.add_parser('save', help='Save model')
    p_save.add_argument('--model', default=None, help='Path to .pt file')

    # demo
    subparsers.add_parser('demo', help='Run a quick demo')

    # bench
    p_bench = subparsers.add_parser('bench', help='Benchmark evaluation')
    p_bench.add_argument('bench_cmd', help='Benchmark command (gsm8k, plot, list)')
    p_bench.add_argument('--model', default=None, help='Path to .pt model file')
    p_bench.add_argument('--task', type=int, default=0, help='Task ID for evaluation')
    p_bench.add_argument('--max-samples', type=int, default=500, help='Max samples to evaluate')
    p_bench.add_argument('--train', action='store_true', help='Train on benchmark before evaluating')
    p_bench.add_argument('--name', default=None, help='Name for this model in the results')

    # chat
    p_chat = subparsers.add_parser('chat', help='Chat / interactive prediction')
    p_chat.add_argument('chat_cmd', help='Chat command (predict, chat)')
    p_chat.add_argument('--text', default=None, help='Text to classify')
    p_chat.add_argument('--file', default=None, help='File with one text per line')
    p_chat.add_argument('--task', type=int, default=0, help='Task ID')
    p_chat.add_argument('--model', default=None, help='Path to .pt file')

    # test
    subparsers.add_parser('test', help='Run model tests')

    args = parser.parse_args()

    if args.command == 'info':
        from src.cli.train import cmd_info
        cmd_info(args)
    elif args.command == 'train':
        from src.cli.train import cmd_train
        cmd_train(args)
    elif args.command == 'predict':
        from src.cli.train import cmd_predict
        cmd_predict(args)
    elif args.command == 'eval':
        from src.cli.train import cmd_eval
        cmd_eval(args)
    elif args.command == 'save':
        from src.cli.train import cmd_save
        cmd_save(args)
    elif args.command == 'demo':
        from src.cli.train import cmd_demo
        cmd_demo(args)
    elif args.command == 'bench':
        from src.cli.bench import eval_gsm8k, plot_all, list_results
        if args.bench_cmd == 'gsm8k':
            eval_gsm8k(args)
        elif args.bench_cmd == 'plot':
            plot_all()
        elif args.bench_cmd == 'list':
            list_results(args)
        else:
            print(f"Unknown bench command: {args.bench_cmd}")
            print("Available: gsm8k, plot, list")
            sys.exit(1)
    elif args.command == 'chat':
        from src.cli.chat import cmd_predict, cmd_chat
        if args.chat_cmd == 'predict':
            cmd_predict(args)
        elif args.chat_cmd == 'chat':
            cmd_chat(args)
        else:
            print(f"Unknown chat command: {args.chat_cmd}")
            print("Available: predict, chat")
            sys.exit(1)
    elif args.command == 'test':
        from src.tests.test_model import run_test
        run_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
