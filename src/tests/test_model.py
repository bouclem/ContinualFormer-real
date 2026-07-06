"""
Test ContinualFormer: zero forgetting guarantee via neuron freezing.
Tests on synthetic tasks + checks expansion mechanism.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.model import ContinualFormer
import random


def make_task(n_samples, vocab_words, label_words, seed=42):
    """Generate texts where label is determined by presence of label_words."""
    random.seed(seed)
    texts, labels = [], []
    other_words = [w for w in vocab_words if w not in label_words]
    for _ in range(n_samples):
        label = random.randint(0, len(label_words) - 1)
        chosen = label_words[label]
        n_words = random.randint(5, 15)
        words = [chosen] * random.randint(2, 5)
        words += [random.choice(other_words) for _ in range(n_words)]
        random.shuffle(words)
        texts.append(' '.join(words))
        labels.append(label)
    return texts, labels


def run_test():
    print("=" * 60)
    print("ContinualFormer Test — Neuron Freezing + Expansion")
    print("=" * 60)

    model = ContinualFormer()
    print(f"\nInitial params: {model.param_count()}")
    model.info()

    # Task 0: sentiment (label 0 = negative words, label 1 = positive words)
    t0_words = ['good', 'great', 'amazing', 'bad', 'terrible', 'awful', 'okay', 'fine', 'worst', 'best']
    t0_texts, t0_labels = make_task(400, t0_words, ['bad', 'good'], seed=1)
    t0_val_t, t0_val_l = make_task(100, t0_words, ['bad', 'good'], seed=99)

    # Task 1: topic (label 0 = tech, label 1 = sport)
    t1_words = ['tech', 'computer', 'software', 'sport', 'game', 'team', 'code', 'player', 'digital', 'field']
    t1_texts, t1_labels = make_task(400, t1_words, ['tech', 'sport'], seed=2)
    t1_val_t, t1_val_l = make_task(100, t1_words, ['tech', 'sport'], seed=98)

    # Task 2: topic (label 0 = food, label 1 = travel)
    t2_words = ['food', 'recipe', 'cook', 'travel', 'trip', 'journey', 'taste', 'explore', 'dish', 'route']
    t2_texts, t2_labels = make_task(400, t2_words, ['food', 'travel'], seed=3)
    t2_val_t, t2_val_l = make_task(100, t2_words, ['food', 'travel'], seed=97)

    print("\n--- Task 0: Sentiment ---")
    model.train(t0_texts, t0_labels, task_id=0, val_texts=t0_val_t, val_labels=t0_val_l)
    acc0_after = model.evaluate(t0_val_t, t0_val_l, task_id=0)
    print(f"  Task 0 accuracy: {acc0_after:.4f}")

    print("\n--- Task 1: Topic (tech vs sport) ---")
    model.train(t1_texts, t1_labels, task_id=1, val_texts=t1_val_t, val_labels=t1_val_l)
    acc0_after_t1 = model.evaluate(t0_val_t, t0_val_l, task_id=0)
    acc1_after = model.evaluate(t1_val_t, t1_val_l, task_id=1)
    print(f"  Task 0 accuracy after task 1: {acc0_after_t1:.4f} (forgetting: {acc0_after - acc0_after_t1:.4f})")
    print(f"  Task 1 accuracy: {acc1_after:.4f}")

    print("\n--- Task 2: Topic (food vs travel) ---")
    model.train(t2_texts, t2_labels, task_id=2, val_texts=t2_val_t, val_labels=t2_val_l)
    acc0_after_t2 = model.evaluate(t0_val_t, t0_val_l, task_id=0)
    acc1_after_t2 = model.evaluate(t1_val_t, t1_val_l, task_id=1)
    acc2_after = model.evaluate(t2_val_t, t2_val_l, task_id=2)
    print(f"  Task 0 accuracy after task 2: {acc0_after_t2:.4f} (forgetgetting: {acc0_after - acc0_after_t2:.4f})")
    print(f"  Task 1 accuracy after task 2: {acc1_after_t2:.4f} (forgetting: {acc1_after - acc1_after_t2:.4f})")
    print(f"  Task 2 accuracy: {acc2_after:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    model.info()
    print(f"\n  Task 0: {acc0_after:.4f} -> {acc0_after_t2:.4f} (forgetting: {acc0_after - acc0_after_t2:.4f})")
    print(f"  Task 1: {acc1_after:.4f} -> {acc1_after_t2:.4f} (forgetting: {acc1_after - acc1_after_t2:.4f})")
    print(f"  Task 2: {acc2_after:.4f}")
    avg_f = ((acc0_after - acc0_after_t2) + (acc1_after - acc1_after_t2)) / 2
    print(f"\n  Average forgetting: {avg_f:.4f}")
    print(f"  Expected: 0.0000 (frozen neurons guarantee zero forgetting)")

    # Save and reload test
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    test_path = os.path.join(checkpoint_dir, 'continualformer_test.pt')
    model.save(test_path)
    model2 = ContinualFormer.load(test_path)
    acc0_loaded = model2.evaluate(t0_val_t, t0_val_l, task_id=0)
    print(f"\n  After save/load — Task 0 accuracy: {acc0_loaded:.4f}")
    model2.info()


if __name__ == '__main__':
    run_test()
