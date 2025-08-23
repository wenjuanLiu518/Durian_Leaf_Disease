#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUX.1-dev Prompt Adapter 正确训练脚本（多类别）
- 从 train_data.jsonl 每行中动态提取 place_holder token
- 注册所有占位符 token 到 tokenizer
- 使用 MLP adapter 对 text embedding 调节
- 输入 encoder_hidden_states 到 UNet 训练
"""

import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from diffusers import DiffusionPipeline, DDPMScheduler
from transformers import AutoTokenizer
from PIL import Image
from tqdm import tqdm
import random

# ==== 参数配置 ====
DATA_JSONL = "./train_data.jsonl"
IMAGE_ROOT = "./images"
OUTPUT_DIR = "./pa_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 2
LR = 1e-4
SAVE_SAMPLE_STEPS = 100

# ==== 自定义 PromptAdapter MLP ====
class PromptAdapter(nn.Module):
    def __init__(self, dim, hidden_dim=1024):
        super().__init__()
        self.proj1 = nn.Linear(dim, hidden_dim)
        self.act = nn.ReLU()
        self.proj2 = nn.Linear(hidden_dim, dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        adapted = self.proj2(self.act(self.proj1(x)))
        return x + self.alpha * adapted

# ==== 数据集加载 ====
class DurianDataset(Dataset):
    def __init__(self, jsonl_path, image_root, tokenizer, image_processor):
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(os.path.join(self.image_root, item["image"])).convert("RGB")
        text = item["text"].replace("XXX", item["place_holder"])
        image_tensor = self.image_processor(image).unsqueeze(0).squeeze(0)
        input_ids = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.squeeze(0)
        return {
            "pixel_values": image_tensor,
            "input_ids": input_ids,
        }

# ==== 主训练流程 ====
def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("加载 FLUX.1-dev pipeline...")

    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("black-forest-labs/FLUX.1-dev")

    # 自动收集并注册所有占位符 token
    with open(DATA_JSONL, "r", encoding="utf-8") as f:
        placeholder_tokens = sorted(set(json.loads(line)["place_holder"] for line in f))
    tokenizer.add_tokens(placeholder_tokens)
    print(f"注册占位符 token: {placeholder_tokens}")

    pipe.text_encoder.resize_token_embeddings(len(tokenizer))

    # 冻结主模型
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    adapter = PromptAdapter(dim=pipe.text_encoder.config.hidden_size).to(DEVICE)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=LR)

    image_processor = pipe.image_processor
    dataset = DurianDataset(DATA_JSONL, IMAGE_ROOT, tokenizer, image_processor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    step = 0
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS} 开始...")
        for batch in tqdm(loader):
            step += 1
            pixel_values = batch["pixel_values"].to(DEVICE).half()
            input_ids = batch["input_ids"].to(DEVICE)

            noise = torch.randn_like(pixel_values)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (pixel_values.shape[0],), device=DEVICE).long()
            noisy_images = noise_scheduler.add_noise(pixel_values, noise, timesteps)

            with torch.no_grad():
                text_outputs = pipe.text_encoder(input_ids=input_ids)
            text_embeds = text_outputs.last_hidden_state
            adapted_embeds = adapter(text_embeds)

            model_pred = pipe.unet(
                sample=noisy_images,
                timestep=timesteps,
                encoder_hidden_states=adapted_embeds
            ).sample

            loss = nn.functional.mse_loss(model_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"[Epoch {epoch+1}] Step {step}: Loss = {loss.item():.4f}")

            if step % SAVE_SAMPLE_STEPS == 0:
                with torch.no_grad():
                    test_token = random.choice(placeholder_tokens)
                    sample_prompt = f"A {test_token} leaf with disease symptoms"
                    text_input = tokenizer(sample_prompt, return_tensors="pt").input_ids.to(DEVICE)
                    text_feat = adapter(pipe.text_encoder(text_input)[0])
                    image = pipe(prompt_embeds=text_feat).images[0]
                    image.save(os.path.join(OUTPUT_DIR, f"sample_e{epoch+1}_s{step}_{test_token}.png"))

    print("训练完成。建议保存 adapter 参数至 pt 文件。")
    torch.save(adapter.state_dict(), os.path.join(OUTPUT_DIR, "prompt_adapter.pth"))

if __name__ == "__main__":
    train()

