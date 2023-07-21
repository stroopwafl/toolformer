# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/06_finetune.ipynb.

# %% ../nbs/06_finetune.ipynb 2
from __future__ import annotations
import math, random, torch, matplotlib.pyplot as plt, numpy as np, matplotlib as mpl, shutil, os, gzip, pickle, re, copy, time
from pathlib import Path
from functools import partial
import fastcore.all as fc
from glob import glob

from torch import tensor, nn, optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, default_collate
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence

# %% auto 0
__all__ = ['get_gen', 'set_grads', 'save_model_weights', 'load_lora_weights', 'finetune', 'FinetuneDS']

# %% ../nbs/06_finetune.ipynb 3
def get_gen(l):
    for i in l: yield i

# %% ../nbs/06_finetune.ipynb 4
def set_grads(model, set_grads_to=False, lora=False):
    if lora:
        for name, mod in model.named_modules():
            for p in mod.parameters():
                p.requires_grad = set_grads_to if 'lora' in name else False
    else:
        for p in model.parameters(): p.requires_grad = set_grads_to
    return

# %% ../nbs/06_finetune.ipynb 5
def save_model_weights(save_path, model, lora=False):
    if lora:
        keys = [k for k in model.state_dict() if 'lora' in k]
        params = [model.state_dict()[key] for key in keys]
        d = {k:p for k,p in zip(keys, params)}
        torch.save(d, save_path)
    else: torch.save(model.state_dict(), save_path)
    print(f'Saved model weights at {save_path}')

# %% ../nbs/06_finetune.ipynb 6
def load_lora_weights(path, model):
    weights = torch.load(path)
    model.load_state_dict(weights, strict=False)
    return

# %% ../nbs/06_finetune.ipynb 7
def finetune(model, dataset, save_path, lr=1e-5, epochs=10, bs=1, opt_func=optim.Adam, lora=True, device='cuda'):
    assert len(dataset) > 0
    dl = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=partial(pad_sequence, batch_first=True), num_workers=4)

    model.train()
    set_grads(model, set_grads_to=True, lora=lora)
    opt = opt_func([p for p in model.parameters() if p.requires_grad], lr)

    for epoch in progress_bar(range(epochs), comment='finetuning...'):
        for i, batch in enumerate(progress_bar(dl, leave=False)):
            inp, label = to_device(batch[:,:-1]), to_device(batch[:,1:])
            logits = self.model(inp, 0)
            logits = rearrange(logits, 'b s v -> b v s')
            loss = F.cross_entropy(logits, label, ignore_index=self.pad_id)
            loss.backward()
            opt.step()
            opt.zero_grad()

    set_grads(self.model, set_grads_to=False, lora=lora)
    save_model_weights(save_path, self.model, lora=lora)

# %% ../nbs/06_finetune.ipynb 8
class FinetuneDS:
    def __init__(self, prompts:List[str]): fc.store_attr()
    def __len__(self): return len(self.prompts)
    def __getitem__(self, i): 
        prompt = self.prompts[i]
        tokens = tokenizer.encode(prompt, bos=True, eos=True)
        return torch.tensor(tokens)