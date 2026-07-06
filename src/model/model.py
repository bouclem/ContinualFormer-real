"""
ContinualFormer — O(1) Transformer with Neuron Freezing + Dynamic Expansion
Links all model components together.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PAD_ID, UNK_ID, BOS_ID, EOS_ID, MAX_VOCAB, ContinualConfig
from .tokenizer import normalize_text, build_vocab, encode as tok_encode, decode as tok_decode
from .layer import TransformerLayer


class ContinualFormer(ContinualConfig):
    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vocab = {'<PAD>': PAD_ID, '<UNK>': UNK_ID, '<BOS>': BOS_ID, '<EOS>': EOS_ID}
        self.embedding = nn.Embedding(self.INIT_VOCAB, self.DIM).to(self.device)
        with torch.no_grad():
            self.embedding.weight[PAD_ID].zero_()
        self.pos_emb = nn.Embedding(self.MAX_LEN, self.DIM).to(self.device)
        self.layers = nn.ModuleList([
            TransformerLayer(self.DIM, self.FFN_HIDDEN, self.HEADS, self.DROPOUT)
            for _ in range(self.LAYERS)
        ]).to(self.device)
        self.lm_head = nn.Linear(self.DIM, self.INIT_VOCAB).to(self.device)
        self.tasks_learned = []

    def _all_modules(self):
        yield 'embedding', self.embedding
        yield 'pos_emb', self.pos_emb
        for i, l in enumerate(self.layers):
            yield f'layer{i}', l
        yield 'lm_head', self.lm_head

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
        ids_list = []
        mask_list = []
        for t in texts:
            ids = tok_encode(t, self.vocab, max_len=self.MAX_LEN)
            length = len(ids)
            ids = ids + [PAD_ID] * (self.MAX_LEN - length)
            ids_list.append(ids[:self.MAX_LEN])
            m = [1.0] * min(length, self.MAX_LEN) + [0.0] * (self.MAX_LEN - min(length, self.MAX_LEN))
            mask_list.append(m)
        ids = torch.tensor(ids_list, dtype=torch.long, device=self.device)
        mask = torch.tensor(mask_list, dtype=torch.float, device=self.device)
        return ids, mask

    def _forward_ids(self, ids, mask):
        pos = torch.arange(ids.size(1), device=self.device).unsqueeze(0).expand(ids.size(0), -1)
        x = self.embedding(ids) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x, mask, causal=True)
        return self.lm_head(x)

    def _freeze_neurons_ids(self, ids, mask):
        pos = torch.arange(ids.size(1), device=self.device).unsqueeze(0).expand(ids.size(0), -1)
        x = self.embedding(ids) + self.pos_emb(pos)
        frozen_count = 0
        for layer in self.layers:
            h = layer.norm2(x)
            importance = layer.ffn.get_importance(h, mask).cpu()
            unfrozen = ~layer.ffn.frozen_mask
            if unfrozen.sum() == 0:
                x = layer(x, mask, causal=True)
                continue
            unfrozen_imp = importance[unfrozen]
            n_freeze = max(1, int(unfrozen.sum().item() * self.FREEZE_RATIO))
            _, top_local = unfrozen_imp.topk(min(n_freeze, len(unfrozen_imp)))
            unfrozen_idx = torch.where(unfrozen)[0]
            freeze_idx = unfrozen_idx[top_local]
            layer.ffn.freeze_neurons(freeze_idx)
            frozen_count += len(freeze_idx)
            x = layer(x, mask, causal=True)
        return frozen_count

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

    def _resize_vocab(self, new_size):
        old_size = self.embedding.num_embeddings
        if new_size == old_size:
            return
        old_emb = self.embedding.weight.data.clone()
        old_head_w = self.lm_head.weight.data.clone()
        old_head_b = self.lm_head.bias.data.clone()
        self.embedding = nn.Embedding(new_size, self.DIM).to(self.device)
        self.lm_head = nn.Linear(self.DIM, new_size).to(self.device)
        with torch.no_grad():
            copy_n = min(old_size, new_size)
            self.embedding.weight[:copy_n] = old_emb[:copy_n]
            self.lm_head.weight[:copy_n] = old_head_w[:copy_n]
            self.lm_head.bias[:copy_n] = old_head_b[:copy_n]
            if new_size > old_size:
                nn.init.xavier_uniform_(self.embedding.weight[old_size:])
                nn.init.zeros_(self.lm_head.bias[old_size:])
            self.embedding.weight[PAD_ID].zero_()

    def train(self, texts, task_id, val_texts=None, verbose=True, checkpoint_path=None):
        old_vs = len(self.vocab)
        self.vocab = build_vocab(texts, max_vocab=self.VOCAB_SIZE, existing_vocab=self.vocab)
        if verbose:
            print(f"  Vocab: {old_vs} -> {len(self.vocab)} words")
        if len(self.vocab) > self.embedding.num_embeddings:
            self._resize_vocab(len(self.vocab))
        optimizer = torch.optim.Adam(self._all_param_list(), lr=self.LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.EPOCHS, eta_min=self.LR * 0.1)
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
            self.lm_head.train()
            perm = torch.randperm(n)
            total_loss = 0.0
            nb = 0
            for i in range(0, n, self.BATCH_SIZE):
                bi = perm[i:i+self.BATCH_SIZE]
                bx_ids = all_ids[bi]
                bx_mask = all_mask[bi]
                optimizer.zero_grad()
                logits = self._forward_ids(bx_ids, bx_mask)
                # Next-token prediction: predict token t+1 from token t
                # Shift: input[:-1] -> target[1:]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = bx_ids[:, 1:].contiguous()
                # Only compute loss on non-pad positions
                shift_mask = bx_mask[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    reduction='none'
                ).view(shift_targets.shape)
                loss = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                nb += 1
            scheduler.step()
            va = 0.0
            if val_ids is not None:
                va = self._evaluate_ids(val_ids, val_mask)
            if verbose and (epoch % 10 == 0 or epoch == self.EPOCHS - 1):
                tp = self.trainable_param_count()
                print(f"  Task {task_id} Epoch {epoch:3d} | loss={total_loss/max(nb,1):.4f} | val_loss={va:.4f} | trainable={tp}")
            if checkpoint_path and (epoch + 1) % 10 == 0:
                self.save(checkpoint_path)
        frozen = self._freeze_neurons_ids(all_ids, all_mask)
        if verbose:
            print(f"  Frozen {frozen} neurons after task {task_id}")
        if self._check_and_expand() and verbose:
            print(f"  Model expanded. New param count: {self.param_count()}")
        if task_id not in self.tasks_learned:
            self.tasks_learned.append(task_id)

    def _evaluate_ids(self, ids, mask):
        self.embedding.eval()
        self.pos_emb.eval()
        for l in self.layers:
            l.eval()
        self.lm_head.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for i in range(0, ids.size(0), 128):
                bi = slice(i, i + 128)
                bx_ids = ids[bi]
                bx_mask = mask[bi]
                logits = self._forward_ids(bx_ids, bx_mask)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = bx_ids[:, 1:].contiguous()
                shift_mask = bx_mask[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    reduction='none'
                ).view(shift_targets.shape)
                total_loss += (loss * shift_mask).sum().item()
                total_tokens += shift_mask.sum().item()
        self.embedding.train()
        self.pos_emb.train()
        for l in self.layers:
            l.train()
        self.lm_head.train()
        return total_loss / max(total_tokens, 1)

    def evaluate(self, texts):
        ids, mask = self._encode(texts)
        return self._evaluate_ids(ids, mask)

    def generate(self, prompt, max_new_tokens=50, temperature=1.0):
        self.embedding.eval()
        self.pos_emb.eval()
        for l in self.layers:
            l.eval()
        self.lm_head.eval()
        ids = tok_encode(prompt, self.vocab, max_len=self.MAX_LEN)
        if len(ids) >= self.MAX_LEN:
            ids = ids[:self.MAX_LEN - 1]
        with torch.no_grad():
            for _ in range(max_new_tokens):
                x_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                x_mask = torch.ones(1, len(ids), device=self.device)
                logits = self._forward_ids(x_ids, x_mask)
                next_logits = logits[0, -1, :] / max(temperature, 1e-8)
                if temperature > 0:
                    probs = F.softmax(next_logits, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                else:
                    next_id = next_logits.argmax().item()
                if next_id == EOS_ID:
                    break
                ids.append(next_id)
                if len(ids) >= self.MAX_LEN:
                    break
        self.embedding.train()
        self.pos_emb.train()
        for l in self.layers:
            l.train()
        self.lm_head.train()
        return tok_decode(ids, self.vocab)

    @staticmethod
    def _checkpoint_exists(path):
        return os.path.exists(path)

    def save(self, path):
        state = {
            'embedding': self.embedding.state_dict(),
            'pos_emb': self.pos_emb.state_dict(),
            'layers': [{n: m.state_dict() for n, m in l.named_children()} for l in self.layers],
            'lm_head': self.lm_head.state_dict(),
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
        ckpt_vocab_size = state['embedding']['weight'].shape[0]
        model._resize_vocab(ckpt_vocab_size)
        model.embedding.load_state_dict(state['embedding'])
        model.pos_emb.load_state_dict(state['pos_emb'])
        for i, ls in enumerate(state['layers']):
            fh = state['ffn_hidden_dims'][i]
            if fh != model.FFN_HIDDEN:
                model.layers[i] = TransformerLayer(model.DIM, fh, model.HEADS, model.DROPOUT).to(model.device)
            for n, sd in ls.items():
                dict(model.layers[i].named_children())[n].load_state_dict(sd)
        if 'lm_head' in state:
            model.lm_head.load_state_dict(state['lm_head'])
        elif 'heads' in state:
            # Legacy checkpoint with classification heads — ignore
            pass
        model.tasks_learned = state.get('tasks_learned', [])
        model.vocab = state.get('vocab', {'<PAD>': PAD_ID, '<UNK>': UNK_ID, '<BOS>': BOS_ID, '<EOS>': EOS_ID})
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
