"""
ContinualFormer — O(1) Transformer with Neuron Freezing + Dynamic Expansion
Links all model components together.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PAD_ID, UNK_ID, MAX_VOCAB, ContinualConfig
from .tokenizer import normalize_text, build_vocab
from .layer import TransformerLayer


class ContinualFormer(ContinualConfig):
    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vocab = {'<PAD>': PAD_ID, '<UNK>': UNK_ID}
        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.DIM).to(self.device)
        with torch.no_grad():
            self.embedding.weight[PAD_ID].zero_()
        self.pos_emb = nn.Embedding(self.MAX_LEN, self.DIM).to(self.device)
        self.layers = nn.ModuleList([
            TransformerLayer(self.DIM, self.FFN_HIDDEN, self.HEADS, self.DROPOUT)
            for _ in range(self.LAYERS)
        ]).to(self.device)
        self.heads = {}
        self.tasks_learned = []

    def _get_or_create_head(self, task_id, num_classes=None):
        if num_classes is None:
            num_classes = self.CLASSES_PER_TASK
        if task_id not in self.heads:
            self.heads[task_id] = nn.Linear(self.DIM, num_classes).to(self.device)
        elif self.heads[task_id].out_features < num_classes:
            old_head = self.heads[task_id]
            new_head = nn.Linear(self.DIM, num_classes).to(self.device)
            with torch.no_grad():
                new_head.weight[:old_head.out_features] = old_head.weight.data
                new_head.bias[:old_head.out_features] = old_head.bias.data
            self.heads[task_id] = new_head
        return self.heads[task_id]

    def _all_modules(self):
        yield 'embedding', self.embedding
        yield 'pos_emb', self.pos_emb
        for i, l in enumerate(self.layers):
            yield f'layer{i}', l
        for t in sorted(self.heads):
            yield f'head{t}', self.heads[t]

    def _named_params(self):
        for pfx, m in self._all_modules():
            for n, p in m.named_parameters():
                yield f"{pfx}.{n}", p

    def _all_param_list(self):
        return [p for _, p in self._named_params() if p.requires_grad]

    def param_count(self):
        return sum(p.numel() for _, p in self._named_params())

    def trainable_param_count(self):
        return sum(p.numel() for _, p in self._named_params() if p.requires_grad)

    def _encode(self, texts):
        B = len(texts)
        ids_list = []
        mask_list = []
        for t in texts:
            words = normalize_text(t)[:self.MAX_LEN]
            row_ids = [self.vocab.get(w, UNK_ID) for w in words]
            row_ids += [PAD_ID] * (self.MAX_LEN - len(row_ids))
            ids_list.append(row_ids[:self.MAX_LEN])
            m = [1.0] * min(len(words), self.MAX_LEN) + [0.0] * (self.MAX_LEN - min(len(words), self.MAX_LEN))
            mask_list.append(m)
        ids = torch.tensor(ids_list, dtype=torch.long, device=self.device)
        mask = torch.tensor(mask_list, dtype=torch.float, device=self.device)
        return ids, mask

    def _forward(self, texts, task_id):
        ids, mask = self._encode(texts)
        return self._forward_ids(ids, mask, task_id)

    def _forward_ids(self, ids, mask, task_id):
        pos = torch.arange(self.MAX_LEN, device=self.device).unsqueeze(0).expand(ids.size(0), -1)
        x = self.embedding(ids) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x, mask)
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self._get_or_create_head(task_id)(pooled)

    def _freeze_neurons_ids(self, ids, mask, task_id):
        pos = torch.arange(self.MAX_LEN, device=self.device).unsqueeze(0).expand(ids.size(0), -1)
        x = self.embedding(ids) + self.pos_emb(pos)
        frozen_count = 0
        for layer in self.layers:
            h = layer.norm2(x)
            importance = layer.ffn.get_importance(h, mask).cpu()
            unfrozen = ~layer.ffn.frozen_mask
            if unfrozen.sum() == 0:
                x = layer(x, mask)
                continue
            unfrozen_imp = importance[unfrozen]
            n_freeze = max(1, int(unfrozen.sum().item() * self.FREEZE_RATIO))
            _, top_local = unfrozen_imp.topk(min(n_freeze, len(unfrozen_imp)))
            unfrozen_idx = torch.where(unfrozen)[0]
            freeze_idx = unfrozen_idx[top_local]
            layer.ffn.freeze_neurons(freeze_idx)
            frozen_count += len(freeze_idx)
            x = layer(x, mask)
        return frozen_count

    def _freeze_neurons(self, texts, task_id):
        ids, mask = self._encode(texts)
        return self._freeze_neurons_ids(ids, mask, task_id)

    def _check_and_expand(self):
        expanded = False
        for layer in self.layers:
            active = layer.ffn.active_count()
            total = layer.ffn.hidden_dim
            if active / total < self.EXPAND_THRESHOLD:
                self._expand_ffn(layer)
                expanded = True
        return expanded

    def _expand_ffn(self, layer):
        old_h = layer.ffn.hidden_dim
        new_n = max(4, int(old_h * self.EXPAND_RATIO))
        new_h = old_h + new_n
        old_fc1_w = layer.ffn.fc1.weight.data.clone()
        old_fc1_b = layer.ffn.fc1.bias.data.clone()
        old_fc2_w = layer.ffn.fc2.weight.data.clone()
        old_frozen = layer.ffn.frozen_mask.clone()
        new_fc1 = nn.Linear(layer.ffn.dim, new_h).to(self.device)
        new_fc2 = nn.Linear(new_h, layer.ffn.dim).to(self.device)
        with torch.no_grad():
            new_fc1.weight[:old_h] = old_fc1_w
            new_fc1.bias[:old_h] = old_fc1_b
            new_fc2.weight[:, :old_h] = old_fc2_w
            nn.init.xavier_uniform_(new_fc1.weight[old_h:])
            nn.init.zeros_(new_fc1.bias[old_h:])
            nn.init.xavier_uniform_(new_fc2.weight[:, old_h:])
        layer.ffn.fc1 = new_fc1
        layer.ffn.fc2 = new_fc2
        layer.ffn.hidden_dim = new_h
        layer.ffn.frozen_mask = torch.zeros(new_h, dtype=torch.bool, device=self.device)
        layer.ffn.frozen_mask[:old_h] = old_frozen.to(self.device)
        old_frozen_idx = torch.where(old_frozen)[0]
        if len(old_frozen_idx) > 0:
            layer.ffn.freeze_neurons(old_frozen_idx)
        print(f"  Expanded FFN: {old_h} -> {new_h} neurons")

    def train(self, texts, labels, task_id, val_texts=None, val_labels=None, verbose=True, checkpoint_path=None):
        old_vs = len(self.vocab)
        self.vocab = build_vocab(texts, max_vocab=self.VOCAB_SIZE, existing_vocab=self.vocab)
        if verbose:
            print(f"  Vocab: {old_vs} -> {len(self.vocab)} words")
        num_classes = max(max(labels) + 1, self.CLASSES_PER_TASK)
        self._get_or_create_head(task_id, num_classes)
        optimizer = torch.optim.Adam(self._all_param_list(), lr=self.LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.EPOCHS, eta_min=self.LR * 0.1)
        y = torch.tensor(labels, dtype=torch.long, device=self.device)
        n = len(texts)

        all_ids, all_mask = self._encode(texts)

        val_ids = None
        val_mask = None
        if val_texts:
            val_ids, val_mask = self._encode(val_texts)

        for epoch in range(self.EPOCHS):
            self.embedding.train()
            self.pos_emb.train()
            for l in self.layers:
                l.train()
            for h in self.heads.values():
                h.train()
            perm = torch.randperm(n)
            total_loss = 0.0
            nb = 0
            for i in range(0, n, self.BATCH_SIZE):
                bi = perm[i:i+self.BATCH_SIZE]
                bx_ids = all_ids[bi]
                bx_mask = all_mask[bi]
                by = y[bi]
                optimizer.zero_grad()
                logits = self._forward_ids(bx_ids, bx_mask, task_id)
                loss = F.cross_entropy(logits, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                nb += 1
            scheduler.step()
            va = 0.0
            if val_ids is not None:
                va = self._evaluate_ids(val_ids, val_mask, val_labels, task_id)
            if verbose and (epoch % 10 == 0 or epoch == self.EPOCHS - 1):
                tp = self.trainable_param_count()
                print(f"  Task {task_id} Epoch {epoch:3d} | loss={total_loss/max(nb,1):.4f} | val_acc={va:.4f} | trainable={tp}")
            if checkpoint_path and (epoch + 1) % 10 == 0:
                self.save(checkpoint_path)
        frozen = self._freeze_neurons_ids(all_ids, all_mask, task_id)
        if verbose:
            print(f"  Frozen {frozen} neurons after task {task_id}")
        if self._check_and_expand() and verbose:
            print(f"  Model expanded. New param count: {self.param_count()}")
        if task_id not in self.tasks_learned:
            self.tasks_learned.append(task_id)

    def _evaluate_ids(self, ids, mask, labels, task_id):
        self.embedding.eval()
        self.pos_emb.eval()
        for l in self.layers:
            l.eval()
        for h in self.heads.values():
            h.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(labels), 128):
                bi = slice(i, i + 128)
                bx_ids = ids[bi]
                bx_mask = mask[bi]
                by = torch.tensor(labels[i:i+128], dtype=torch.long, device=self.device)
                pred = self._forward_ids(bx_ids, bx_mask, task_id).argmax(1)
                correct += (pred == by).sum().item()
                total += len(by)
        self.embedding.train()
        self.pos_emb.train()
        for l in self.layers:
            l.train()
        for h in self.heads.values():
            h.train()
        return correct / max(total, 1)

    def evaluate(self, texts, labels, task_id):
        self.embedding.eval()
        self.pos_emb.eval()
        for l in self.layers:
            l.eval()
        for h in self.heads.values():
            h.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(texts), 128):
                bt = texts[i:i+128]
                by = torch.tensor(labels[i:i+128], dtype=torch.long, device=self.device)
                pred = self._forward(bt, task_id).argmax(1)
                correct += (pred == by).sum().item()
                total += len(bt)
        self.embedding.train()
        self.pos_emb.train()
        for l in self.layers:
            l.train()
        for h in self.heads.values():
            h.train()
        return correct / max(total, 1)

    def predict(self, texts, task_id):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        self.embedding.eval()
        self.pos_emb.eval()
        for l in self.layers:
            l.eval()
        for h in self.heads.values():
            h.eval()
        with torch.no_grad():
            preds = self._forward(texts, task_id).argmax(1).cpu().numpy()
        self.embedding.train()
        self.pos_emb.train()
        for l in self.layers:
            l.train()
        for h in self.heads.values():
            h.train()
        return int(preds[0]) if single else preds.tolist()

    def predict_proba(self, texts, task_id):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        self.embedding.eval()
        self.pos_emb.eval()
        for l in self.layers:
            l.eval()
        for h in self.heads.values():
            h.eval()
        with torch.no_grad():
            logits = self._forward(texts, task_id)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        self.embedding.train()
        self.pos_emb.train()
        for l in self.layers:
            l.train()
        for h in self.heads.values():
            h.train()
        return probs[0].tolist() if single else probs.tolist()

    @staticmethod
    def _checkpoint_exists(path):
        return os.path.exists(path)

    def save(self, path):
        state = {
            'embedding': self.embedding.state_dict(),
            'pos_emb': self.pos_emb.state_dict(),
            'layers': [{n: m.state_dict() for n, m in l.named_children()} for l in self.layers],
            'heads': {t: h.state_dict() for t, h in self.heads.items()},
            'tasks_learned': self.tasks_learned,
            'vocab': self.vocab,
            'frozen_masks': [l.ffn.frozen_mask.clone() for l in self.layers],
            'ffn_hidden_dims': [l.ffn.hidden_dim for l in self.layers],
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(state, path)
        print(f"Saved to {path} ({self.param_count()} params, {len(self.tasks_learned)} tasks)")

    @classmethod
    def load(cls, path, device=None):
        model = cls(device=device)
        state = torch.load(path, map_location=model.device)
        model.embedding.load_state_dict(state['embedding'])
        model.pos_emb.load_state_dict(state['pos_emb'])
        for i, ls in enumerate(state['layers']):
            fh = state['ffn_hidden_dims'][i]
            if fh != model.FFN_HIDDEN:
                model.layers[i] = TransformerLayer(model.DIM, fh, model.HEADS, model.DROPOUT).to(model.device)
            for n, sd in ls.items():
                dict(model.layers[i].named_children())[n].load_state_dict(sd)
        for t, hs in state['heads'].items():
            model._get_or_create_head(t)
            model.heads[t].load_state_dict(hs)
        model.tasks_learned = state['tasks_learned']
        model.vocab = state.get('vocab', {'<PAD>': PAD_ID, '<UNK>': UNK_ID})
        for i, fm in enumerate(state['frozen_masks']):
            model.layers[i].ffn.frozen_mask = fm.to(model.device)
            frozen_indices = torch.where(fm)[0]
            if len(frozen_indices) > 0:
                model.layers[i].ffn.freeze_neurons(frozen_indices)
        return model

    def info(self):
        print(f"ContinualFormer")
        print(f"  Params: {self.param_count()} (trainable: {self.trainable_param_count()})")
        print(f"  Tasks learned: {len(self.tasks_learned)} {self.tasks_learned}")
        print(f"  Vocab: {len(self.vocab)} words")
        for i, l in enumerate(self.layers):
            a = l.ffn.active_count()
            t = l.ffn.hidden_dim
            print(f"  Layer {i} FFN: {a}/{t} active ({t-a} frozen)")
