#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Textual-Inversion (single token) + Dual Contrastive Learning for FLUX.1-dev

功能要点：
1. 双向 InfoNCE：Text->Image 与 Image->Text，两者损失相加。
2. 两阶段训练：warmup 阶段(弱对比/无bank) → main 阶段(强对比/有bank、hard negative、margin)。
3. 采样保存：每 sample_every 步，固定若干 prompt 调用 FLUX 生成图像保存。
4. fp16/bf16 主干，logits/相似度部分用 fp32。
5. 可微替换 hidden-state，梯度真正流入占位符向量；日志里缓存反传前梯度，保证看到非0。
6. Projector 可训练（单独 lr）。

依赖：diffusers>=0.27，transformers，tqdm，Pillow，torch
"""

import os, json, random, argparse, collections
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from diffusers import FluxPipeline
from transformers import CLIPModel, CLIPProcessor


# ---------------- Dataset ----------------
class TIPairedDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: str = "", placeholder: str = "<durian_leaf>"):
        self.samples = [json.loads(l) for l in open(jsonl_path, 'r', encoding='utf-8') if l.strip()]
        self.root = Path(image_root) if image_root else Path(".")
        self.placeholder = placeholder

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        prompt = rec["prompt"]
        if self.placeholder not in prompt:
            prompt = prompt.strip() + " " + self.placeholder
        img = Image.open(self.root / rec["image"]).convert("RGB")
        return {"image": img, "prompt": prompt}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"image": [b["image"] for b in batch],
            "prompt": [b["prompt"] for b in batch]}


# ---------------- Utils ----------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def mean_pool(hidden: torch.Tensor, mask: torch.Tensor):
    mask = mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)

@torch.no_grad()
def clamp_and_clean_(tensor, mn=-2.0, mx=2.0):
    tensor.data.copy_(torch.nan_to_num(tensor.data, nan=0.0, posinf=0.0, neginf=0.0))
    tensor.data.clamp_(mn, mx)


def normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def info_nce(q, k_pos, k_neg, tau):
    """
    q: (B, D)  queries
    k_pos: (B, D) positives
    k_neg: (N, D) negatives  (can be None -> use only k_pos as negatives)
    return: loss, diag_sim, logits (B, P+N)
    """
    qn = normalize(q.float())
    kp = normalize(k_pos.float())
    pos_logits = (qn * kp).sum(-1, keepdim=True)  # (B,1)

    if k_neg is not None and k_neg.numel() > 0:
        kn = normalize(k_neg.float())
        neg_logits = qn @ kn.t()  # (B, N)
        all_logits = torch.cat([pos_logits, neg_logits], dim=1) / tau
    else:
        all_logits = pos_logits / tau

    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)  # 正样本在列0
    loss = nn.CrossEntropyLoss()(torch.clamp(all_logits, -60, 60), labels)
    diag_sim = (qn * kp).sum(-1).mean().item()
    return loss, diag_sim, all_logits


def topk_negatives(q, bank, k):
    if bank is None or bank.numel() == 0 or k <= 0:
        return bank
    with torch.no_grad():
        qn = normalize(q.float())
        bn = normalize(bank.float())
        sims = qn @ bn.t()  # (B, Nbank)
        idxs = torch.topk(sims, k=min(k, bn.size(0)), dim=1, largest=True).indices
    gathered = [bank[idxs[i]] for i in range(q.size(0))]
    return torch.cat(gathered, dim=0)


# ---------------- Config ----------------
@dataclass
class TrainCfg:
    data_jsonl: str
    image_root: str
    out_dir: str = "outputs/ti_cl"
    placeholder: str = "<durian_leaf>"

    # Optim
    lr: float = 5e-4
    proj_lr: float = 2e-4
    batch_size: int = 4
    max_steps: int = 1200
    warmup_steps: int = 200
    grad_accum: int = 1

    # Logging / saving / sampling
    log_every: int = 50
    save_every: int = 200
    sample_every: int = 200
    sample_prompts: str = ""  # 用;分隔： "a photo of <durian_leaf>;macro shot of <durian_leaf>"
    num_sample_images: int = 1
    num_inference_steps: int = 28
    guidance_scale: float = 3.5

    # Loss weights
    lambda_con: float = 1.0
    lambda_diff: float = 0.0
    lambda_l2: float = 1e-3
    cos_w: float = 0.2
    margin_w: float = 0.3
    margin_m: float = 0.15

    # tau schedule
    tau_warm: float = 0.2
    tau_main: float = 0.08

    # bank / neg
    bank_size: int = 256
    topk_neg_main: int = 8

    # misc
    seed: int = 42
    device: str = "cuda"
    dtype: str = "fp16"      # "bf16"/"fp16"/"fp32"
    num_workers: int = 0
    max_length: int = 256
    max_grad_norm: float = 1.0
    min_batch_for_nce: int = 2


# ---------------- Optional diffusion loss (0 by default) ----------------
def diffusion_mse_loss(pipe: FluxPipeline, images, prompts, device, model_dtype):
    if not hasattr(pipe.scheduler, "add_noise"):
        return torch.zeros((), device=device, dtype=model_dtype)
    with torch.no_grad():
        vae = pipe.vae
        proc = torch.cat([pipe.image_processor.preprocess(im).to(device, dtype=model_dtype) for im in images], 0)
        lat = vae.encode(proc).latent_dist.sample().to(model_dtype) * 0.18215
    noise = torch.randn_like(lat, dtype=model_dtype)
    t = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (lat.size(0),), device=device).long()
    noisy = pipe.scheduler.add_noise(lat, noise, t)
    tok = pipe.tokenizer(prompts, padding="max_length",
                         max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        hs = pipe.text_encoder(**tok)[0]
    with torch.autocast(device_type=device.type, dtype=model_dtype):
        pred = pipe.unet(noisy, t, encoder_hidden_states=hs).sample
    return nn.MSELoss()(pred.to(model_dtype), noise)


# ---------------- Training ----------------
def train(cfg: TrainCfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(Path(cfg.out_dir) / "samples", exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.dtype, torch.float32)

    # Load FLUX
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch_dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None

    model_dtype = pipe.text_encoder.dtype
    tokenizer, text_encoder = pipe.tokenizer, pipe.text_encoder
    emb_layer = text_encoder.get_input_embeddings()

    # add token
    tokenizer.add_tokens([cfg.placeholder])
    text_encoder.resize_token_embeddings(len(tokenizer))
    tok_id = tokenizer.convert_tokens_to_ids(cfg.placeholder)

    for p in text_encoder.parameters():
        p.requires_grad = False

    # token param
    init_vec_fp32 = emb_layer.weight.data[tok_id].clone().float().cpu()
    ti_param = nn.Parameter(init_vec_fp32.clone(), requires_grad=True)

    # projector
    d_t5 = emb_layer.weight.shape[-1]
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    d_clip = clip_model.config.projection_dim
    projector = nn.Linear(d_t5, d_clip, bias=False).to(device)

    optim = torch.optim.AdamW(
        [{"params": [ti_param], "lr": cfg.lr},
         {"params": projector.parameters(), "lr": cfg.proj_lr}]
    )

    # Data
    ds = TIPairedDataset(cfg.data_jsonl, cfg.image_root, cfg.placeholder)
    eff_bs = min(cfg.batch_size, len(ds))
    dl = DataLoader(ds, batch_size=eff_bs, shuffle=True,
                    num_workers=cfg.num_workers, drop_last=False, collate_fn=collate_fn)
    data_iter = cycle(dl)

    # banks
    img_bank = collections.deque(maxlen=cfg.bank_size)
    txt_bank = collections.deque(maxlen=cfg.bank_size)

    # sampling prompts
    sample_prompts = [p.strip() for p in cfg.sample_prompts.split(";") if p.strip()]
    if not sample_prompts:
        sample_prompts = [f"A macro photo of {cfg.placeholder}", f"botanical illustration of {cfg.placeholder}"]

    total = cfg.max_steps
    pbar = tqdm(total=total, ncols=92, desc="TI+DualCL")
    accum = 0

    last_grad_tok = 0.0
    last_grad_proj = 0.0

    for step in range(1, total + 1):
        # phase params
        warm = step <= cfg.warmup_steps
        tau = cfg.tau_warm if warm else cfg.tau_main
        use_bank = (not warm) and (cfg.bank_size > 0)
        topk_k = 0 if warm else cfg.topk_neg_main
        cos_w = cfg.cos_w
        margin_w = 0.0 if warm else cfg.margin_w

        batch = next(data_iter)
        imgs, prompts = batch["image"], batch["prompt"]
        if len(imgs) < cfg.min_batch_for_nce:
            imgs = imgs * cfg.min_batch_for_nce
            prompts = prompts * cfg.min_batch_for_nce

        # diffusion (optional)
        ld = diffusion_mse_loss(pipe, imgs, prompts, device, model_dtype) if cfg.lambda_diff > 0 else torch.zeros((), device=device)

        # -------- Text branch --------
        tok = tokenizer(prompts, padding="max_length",
                        max_length=min(cfg.max_length, tokenizer.model_max_length),
                        truncation=True, return_tensors="pt").to(device)
        hs = text_encoder(**tok)[0]                   # (B,L,d_t5)
        mask = (tok.input_ids == tok_id)
        mask_f = mask.unsqueeze(-1).to(hs.dtype)
        ti_vec = ti_param.to(device).to(hs.dtype)
        hs = hs * (1 - mask_f) + ti_vec.unsqueeze(0).unsqueeze(0) * mask_f
        ti_feats = mean_pool(hs, mask) if mask.sum() > 0 else hs.mean(1)
        txt_proj = projector(ti_feats)               # (B, d_clip)

        # -------- Image branch --------
        with torch.no_grad():
            clip_in = clip_proc(text=prompts, images=imgs, return_tensors="pt", padding=True)
            clip_in = {k: v.to(device) for k, v in clip_in.items()}
            img_emb = clip_model(**clip_in).image_embeds  # (B, d_clip)

            # push banks
            if use_bank:
                img_bank.extend([e.detach().to(dtype=txt_proj.dtype) for e in img_emb])
                txt_bank.extend([e.detach().to(dtype=txt_proj.dtype) for e in txt_proj])

        # build negatives
        img_bank_tensor = torch.stack(list(img_bank), dim=0).to(device) if (use_bank and len(img_bank) > 0) else None
        txt_bank_tensor = torch.stack(list(txt_bank), dim=0).to(device) if (use_bank and len(txt_bank) > 0) else None
        if use_bank and topk_k > 0:
            img_bank_tensor = topk_negatives(txt_proj, img_bank_tensor, topk_k)
            txt_bank_tensor = topk_negatives(img_emb, txt_bank_tensor, topk_k)

        # --- Dual InfoNCE ---
        # Text->Image
        loss_t2i, sim_t2i, logits_t2i = info_nce(txt_proj, img_emb, img_bank_tensor, tau)
        # Image->Text
        loss_i2t, sim_i2t, logits_i2t = info_nce(img_emb, txt_proj, txt_bank_tensor, tau)

        # Cosine positives（两向平均）
        cos_pos_t2i = 1 - torch.nn.functional.cosine_similarity(txt_proj.float(), img_emb.float(), dim=-1).mean()
        cos_pos_i2t = cos_pos_t2i  # 对称，值相同
        cos_pos = (cos_pos_t2i + cos_pos_i2t) * 0.5

        # Margin loss（均值版本）
        def margin_tensor(logits):
            # logits: (B, 1+Nneg)  第0列正样本
            pos = logits[:, 0]
            neg_mean_each = logits[:, 1:].mean(dim=1) if logits.size(1) > 1 else pos*0
            return torch.relu(neg_mean_each - pos + cfg.margin_m).mean(), pos.mean().item(), neg_mean_each.mean().item()

        margin_loss_t2i, pos_logit_t2i, neg_mean_t2i = margin_tensor(logits_t2i)
        margin_loss_i2t, pos_logit_i2t, neg_mean_i2t = margin_tensor(logits_i2t)
        margin_loss = (margin_loss_t2i + margin_loss_i2t) * 0.5

        # L2 正则
        l2 = ((ti_param - init_vec_fp32.to(ti_param.device)) ** 2).mean()

        # 总 loss
        loss = cfg.lambda_diff*ld + cfg.lambda_con*(loss_t2i + loss_i2t) \
               + cfg.lambda_l2*l2 + cos_w*cos_pos + margin_w*margin_loss

        if torch.isnan(loss) or torch.isinf(loss):
            pbar.write(f"[WARN] step {step}: NaN/Inf, skip.")
            optim.zero_grad(); pbar.update(1); continue

        (loss / cfg.grad_accum).backward()

        # 缓存梯度
        if ti_param.grad is not None:
            last_grad_tok = ti_param.grad.detach().abs().mean().item()
        gp = [p.grad.detach().abs().mean().item() for p in projector.parameters() if p.grad is not None]
        last_grad_proj = sum(gp)/len(gp) if gp else 0.0

        accum += 1
        if accum % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([ti_param], cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(projector.parameters(), cfg.max_grad_norm)
            optim.step(); optim.zero_grad()
            with torch.no_grad():
                clamp_and_clean_(ti_param, -2.0, 2.0)
            accum = 0

        if step % cfg.log_every == 0:
            pct = 100 * step / total
            tok_delta = (ti_param.detach().cpu() - init_vec_fp32).abs().mean().item()

            # 记录 margin（正 - 负）
            margin_t2i = pos_logit_t2i - neg_mean_t2i
            margin_i2t = pos_logit_i2t - neg_mean_i2t

            bank_sz_i = img_bank_tensor.size(0) if (img_bank_tensor is not None) else 0
            bank_sz_t = txt_bank_tensor.size(0) if (txt_bank_tensor is not None) else 0

            pbar.write(f"[{step}/{total} {pct:.1f}%] warm={warm} diff={ld.item():.4f} "
                       f"con_t2i={loss_t2i.item():.4f} con_i2t={loss_i2t.item():.4f} "
                       f"sim_t2i={sim_t2i:.3f} sim_i2t={sim_i2t:.3f} l2={l2.item():.6f} "
                       f"cos_pos={cos_pos.item():.3f} marginL={margin_loss.item():.3f} total={loss.item():.4f}")
            pbar.write(f"[DEBUG] grad_tok={last_grad_tok:.2e} grad_proj={last_grad_proj:.2e} tok_delta={tok_delta:.2e} "
                       f"margin_t2i={margin_t2i:.3f} margin_i2t={margin_i2t:.3f} "
                       f"bank_i={bank_sz_i} bank_t={bank_sz_t} tau={tau}")

        if step % cfg.save_every == 0:
            vec = ti_param.detach().cpu()
            path = Path(cfg.out_dir) / f"ti_vec_step{step}.pt"
            torch.save({"placeholder": cfg.placeholder, "vec": vec}, path)
            pbar.write(f"[SAVE] {path}")

        if step % cfg.sample_every == 0:
            pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch_dtype):
                for i, sp in enumerate(sample_prompts):
                    images = pipe(prompt=sp, num_inference_steps=cfg.num_inference_steps,
                                  guidance_scale=cfg.guidance_scale).images[:cfg.num_sample_images]
                    for j, im in enumerate(images):
                        im.save(Path(cfg.out_dir)/"samples"/f"step{step:04d}_{i}_{j}.png")

        pbar.update(1)

    # final save
    vec = ti_param.detach().cpu()
    torch.save({"placeholder": cfg.placeholder, "vec": vec}, Path(cfg.out_dir) / "ti_vec_final.pt")
    pbar.write("Done. Final vector saved.")
    pbar.close()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="outputs/ti_cl")
    ap.add_argument("--placeholder", type=str, default="<durian_leaf>")

    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--proj_lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=1200)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--grad_accum", type=int, default=1)

    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--sample_every", type=int, default=200)
    ap.add_argument("--sample_prompts", type=str, default="")
    ap.add_argument("--num_sample_images", type=int, default=1)
    ap.add_argument("--num_inference_steps", type=int, default=28)
    ap.add_argument("--guidance_scale", type=float, default=3.5)

    ap.add_argument("--lambda_con", type=float, default=1.0)
    ap.add_argument("--lambda_diff", type=float, default=0.0)
    ap.add_argument("--lambda_l2", type=float, default=1e-3)
    ap.add_argument("--cos_w", type=float, default=0.2)
    ap.add_argument("--margin_w", type=float, default=0.3)
    ap.add_argument("--margin_m", type=float, default=0.15)

    ap.add_argument("--tau_warm", type=float, default=0.2)
    ap.add_argument("--tau_main", type=float, default=0.08)

    ap.add_argument("--bank_size", type=int, default=256)
    ap.add_argument("--topk_neg_main", type=int, default=8)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--min_batch_for_nce", type=int, default=2)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainCfg(**vars(args))
    train(cfg)

