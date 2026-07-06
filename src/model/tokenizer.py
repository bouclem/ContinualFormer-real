import re
from collections import Counter

from .config import PAD_ID, UNK_ID, BOS_ID, EOS_ID, MAX_VOCAB


def normalize_text(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def build_vocab(texts, max_vocab=MAX_VOCAB, existing_vocab=None):
    if existing_vocab:
        vocab = dict(existing_vocab)
        next_id = max(vocab.values()) + 1
    else:
        vocab = {'<PAD>': PAD_ID, '<UNK>': UNK_ID, '<BOS>': BOS_ID, '<EOS>': EOS_ID}
        next_id = 4
    wc = Counter()
    for t in texts:
        wc.update(normalize_text(t))
    for w, _ in wc.most_common():
        if w in vocab or next_id >= max_vocab:
            continue
        vocab[w] = next_id
        next_id += 1
    return vocab


def encode(text, vocab, max_len=None):
    tokens = normalize_text(text)
    ids = [BOS_ID] + [vocab.get(w, UNK_ID) for w in tokens] + [EOS_ID]
    if max_len:
        ids = ids[:max_len]
    return ids


def decode(ids, vocab):
    inv = {v: k for k, v in vocab.items()}
    tokens = []
    for i in ids:
        if i in (PAD_ID, BOS_ID, EOS_ID):
            continue
        tokens.append(inv.get(i, '<UNK>'))
    return ' '.join(tokens)
