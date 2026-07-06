"""
Test ContinualFormer: zero forgetting guarantee via neuron freezing.
Tests on synthetic tasks + checks expansion mechanism.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.model import ContinualFormer
import random


def make_task(n_samples, vocab_words, seed=42):
    """Generate texts using vocab_words."""
    random.seed(seed)
    texts = []
    for _ in range(n_samples):
        n_words = random.randint(5, 15)
        words = [random.choice(vocab_words) for _ in range(n_words)]
        random.shuffle(words)
        texts.append(' '.join(words))
    return texts


def run_test():
    print("=" * 60)
    print("ContinualFormer Test — Neuron Freezing + Expansion")
    print("=" * 60)

    model = ContinualFormer()
    print(f"\nInitial params: {model.param_count()}")
    model.info()

    # Task 0: sentiment words
    t0_words = ['good', 'great', 'amazing', 'bad', 'terrible', 'awful', 'okay', 'fine', 'worst', 'best']
    t0_texts = make_task(400, t0_words, seed=1)
    t0_val = make_task(100, t0_words, seed=99)

    # Task 1: tech words
    t1_words = ['tech', 'computer', 'software', 'sport', 'game', 'team', 'code', 'player', 'digital', 'field']
    t1_texts = make_task(400, t1_words, seed=2)
    t1_val = make_task(100, t1_words, seed=98)

    # Task 2: food words
    t2_words = ['food', 'recipe', 'cook', 'travel', 'trip', 'journey', 'taste', 'explore', 'dish', 'route']
    t2_texts = make_task(400, t2_words, seed=3)
    t2_val = make_task(100, t2_words, seed=97)

    print("\n--- Task 0: Sentiment ---")
    model.train(t0_texts, task_id=0, val_texts=t0_val)
    loss0_after = model.evaluate(t0_val)
    print(f"  Task 0 val loss: {loss0_after:.4f}")

    print("\n--- Task 1: Tech ---")
    model.train(t1_texts, task_id=1, val_texts=t1_val)
    loss0_after_t1 = model.evaluate(t0_val)
    loss1_after = model.evaluate(t1_val)
    print(f"  Task 0 val loss after task 1: {loss0_after_t1:.4f} (forgetting: {loss0_after_t1 - loss0_after:.4f})")
    print(f"  Task 1 val loss: {loss1_after:.4f}")

    print("\n--- Task 2: Food ---")
    model.train(t2_texts, task_id=2, val_texts=t2_val)
    loss0_after_t2 = model.evaluate(t0_val)
    loss1_after_t2 = model.evaluate(t1_val)
    loss2_after = model.evaluate(t2_val)
    print(f"  Task 0 val loss after task 2: {loss0_after_t2:.4f} (forgetting: {loss0_after_t2 - loss0_after:.4f})")
    print(f"  Task 1 val loss after task 2: {loss1_after_t2:.4f} (forgetting: {loss1_after_t2 - loss1_after:.4f})")
    print(f"  Task 2 val loss: {loss2_after:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    model.info()
    print(f"\n  Task 0: {loss0_after:.4f} -> {loss0_after_t2:.4f} (forgetting: {loss0_after_t2 - loss0_after:.4f})")
    print(f"  Task 1: {loss1_after:.4f} -> {loss1_after_t2:.4f} (forgetting: {loss1_after_t2 - loss1_after:.4f})")
    print(f"  Task 2: {loss2_after:.4f}")
    avg_f = ((loss0_after_t2 - loss0_after) + (loss1_after_t2 - loss1_after)) / 2
    print(f"\n  Average forgetting: {avg_f:.4f}")
    print(f"  Expected: ~0.0 (frozen neurons preserve old knowledge)")

    # Generation test
    print("\n--- Generation test ---")
    output = model.generate("good great amazing", max_new_tokens=10, temperature=0.7)
    print(f"  'good great amazing' -> {output}")

    # Save and reload test
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    test_path = os.path.join(checkpoint_dir, 'continualformer_test.pt')
    model.save(test_path)
    model2 = ContinualFormer.load(test_path)
    loss0_loaded = model2.evaluate(t0_val)
    print(f"\n  After save/load — Task 0 val loss: {loss0_loaded:.4f}")
    model2.info()


if __name__ == '__main__':
    run_test()
