import re
from collections import Counter

from .config import PAD_ID, UNK_ID, MAX_VOCAB


def normalize_text(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def build_vocab(texts, max_vocab=MAX_VOCAB, existing_vocab=None):
    if existing_vocab:
        vocab = dict(existing_vocab)
        next_id = max(vocab.values()) + 1
    else:
        vocab = {'<PAD>': PAD_ID, '<UNK>': UNK_ID}
        next_id = 2
    wc = Counter()
    for t in texts:
        wc.update(normalize_text(t))
    for w, _ in wc.most_common():
        if w in vocab or next_id >= max_vocab:
            continue
        vocab[w] = next_id
        next_id += 1
    return vocab
