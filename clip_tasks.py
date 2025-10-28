
import os
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Optional

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_clip(model_name: str = "ViT-B/32", device: Optional[torch.device] = None):
    import clip  # pip install git+https://github.com/openai/CLIP.git
    device = device or get_device()
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device

def load_pairs_from_csv(csv_path: str, image_col: str = "path", text_col: str = "caption") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if image_col not in df.columns:
        raise ValueError(f"Column '{image_col}' not found in {csv_path}. Columns: {df.columns.tolist()}")
    if text_col not in df.columns:
        for alt in ["caption_concise", "text", "report"]:
            if alt in df.columns:
                text_col = alt
                break
        else:
            raise ValueError(f"No text column found. Tried '{text_col}', 'caption_concise', 'text', 'report'.")
    df = df[df[image_col].notna() & df[text_col].notna()].copy()
    df["image_path"] = df[image_col].astype(str)
    df["text_clean"] = df[text_col].astype(str)
    return df[["image_path", "text_clean"] + [c for c in df.columns if c not in ("image_path", "text_clean")]]

@torch.no_grad()
def encode_images(model, preprocess, paths: Iterable[str], device: torch.device, batch_size: int = 64) -> torch.Tensor:
    feats = []
    paths = list(paths)
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
        images = torch.stack(imgs, dim=0).to(device)
        imf = model.encode_image(images)
        imf = imf / imf.norm(dim=-1, keepdim=True)
        feats.append(imf)
    return torch.cat(feats, dim=0)

@torch.no_grad()
def encode_texts(model, prompts: Iterable[str], device: torch.device, batch_size: int = 256) -> torch.Tensor:
    import clip
    prompts = list(prompts)
    feats = []
    for i in range(0, len(prompts), batch_size):
        toks = clip.tokenize(prompts[i:i+batch_size], truncate=True).to(device)
        tf = model.encode_text(toks)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        feats.append(tf)
    return torch.cat(feats, dim=0)

@torch.no_grad()
def percent_across_images(text_emb: torch.Tensor, image_embs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    sims = (image_embs @ text_emb.T).squeeze(1)
    perc = torch.softmax(sims / temperature, dim=0) * 100.0
    return perc

@torch.no_grad()
def percent_across_images_scaled(model, text_emb: torch.Tensor, image_embs: torch.Tensor) -> torch.Tensor:
    sims = (image_embs @ text_emb.T).squeeze(1)
    scale = model.logit_scale.exp()
    perc = torch.softmax(sims * scale, dim = 0) * 100.0
    return perc

@torch.no_grad()
def percent_across_prompts(image_emb: torch.Tensor, text_embs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    sims = (text_embs @ image_emb.T).squeeze(1)
    perc = torch.softmax(sims / temperature, dim=0) * 100.0
    return perc

@torch.no_grad()
def retrieval_at_k(image_embs: torch.Tensor, text_embs: torch.Tensor, ks: Tuple[int, ...] = (1, 5, 10)) -> Dict[str, float]:
    sim = image_embs @ text_embs.T
    N = sim.size(0)
    ranks_i2t = torch.argsort(sim, dim=1, descending=True)
    ranks_t2i = torch.argsort(sim.T, dim=1, descending=True)
    metrics = {}
    for k in ks:
        r_i2t = (ranks_i2t[:, :k] == torch.arange(N, device=sim.device).unsqueeze(1)).any(dim=1).float().mean().item()
        r_t2i = (ranks_t2i[:, :k] == torch.arange(N, device=sim.device).unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f"R@{k}"] = (r_i2t + r_t2i) / 2.0
    return metrics

def make_prompt_ensemble(concept: str, templates: Optional[list] = None) -> list:
    if templates is None:
        templates = [
            "a chest x-ray showing {}",
            "radiographic evidence of {}",
            "{} present on chest radiograph",
            "frontal chest x-ray with {}",
            "an image consistent with {} on CXR",
        ]
    return [t.format(concept) for t in templates]

@torch.no_grad()
def encode_prompt_ensemble(model, concept: str, device: torch.device, templates: Optional[list] = None) -> torch.Tensor:
    prompts = make_prompt_ensemble(concept, templates)
    emb = encode_texts(model, prompts, device)
    emb = emb.mean(dim=0, keepdim=True)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb

def save_features_npz(path: str, image_paths: list, image_embs: torch.Tensor, text_list: Optional[list] = None, text_embs: Optional[torch.Tensor] = None):
    np.savez_compressed(
        path,
        image_paths=np.array(image_paths),
        image_embs=image_embs.detach().cpu().numpy(),
        texts=np.array(text_list) if text_list is not None else np.array([]),
        text_embs=text_embs.detach().cpu().numpy() if text_embs is not None else np.array([]),
    )

def load_features_npz(path: str):
    data = np.load(path, allow_pickle=True)
    out = {
        "image_paths": data["image_paths"].tolist(),
        "image_embs": torch.tensor(data["image_embs"]),
    }
    if "texts" in data.files and data["texts"].size > 0:
        out["texts"] = data["texts"].tolist()
    if "text_embs" in data.files and data["text_embs"].size > 0:
        out["text_embs"] = torch.tensor(data["text_embs"])
    return out

@torch.no_grad()
def build_index_from_csv(csv_path: str, model_name: str = "ViT-B/32", image_col: str = "path", text_col: str = "caption",
                         batch_size: int = 64, save_npz: Optional[str] = None):
    df = load_pairs_from_csv(csv_path, image_col=image_col, text_col=text_col)
    model, preprocess, device = load_clip(model_name)
    image_embs = encode_images(model, preprocess, df["image_path"].tolist(), device, batch_size=batch_size)
    text_embs  = encode_texts(model, df["text_clean"].tolist(), device, batch_size=256)
    if save_npz:
        save_features_npz(save_npz, df["image_path"].tolist(), image_embs, df["text_clean"].tolist(), text_embs)
    return df, image_embs, text_embs, model, preprocess, device

@torch.no_grad()
def rank_images_by_prompt(model, image_embs: torch.Tensor, device: torch.device, prompt: str, topk: int = 10):
    txt = encode_texts(model, [prompt], device)
    sims = (image_embs @ txt.T).squeeze(1)
    scale = model.logit_scale.exp()
    perc = torch.softmax(sims * scale, dim = 0) * 100.0
    top = torch.topk(perc, k=min(topk, perc.numel()))
    return [(int(i), float(p), float(sims[i].item())) for i, p in zip(top.indices, top.values)]
