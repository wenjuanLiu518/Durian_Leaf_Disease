#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Textual Inversion + 对比学习训练脚本（多类别）
- 从 train_data.jsonl 动态提取每条样本的 place_holder
- 注册所有占位符 token 到 tokenizer
- 训练 token embedding，用于农业语义图文对齐
"""

import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer
from diffusers import FluxPipeline
from PIL import Image
from tqdm import tqdm

# ==== 用户设置区域 ====
DATA_JSONL = "./datasets/train_data.jsonl"       # 每行格式：{"image": ..., "text": ..., "place_holder": ...}
IMAGE_ROOT = "./datasets"                 # 图像根目录
OUTPUT_DIR = "./ti_output"              # 模型输出保存目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 4
LR = 5e-4

# ==== 数据加载 ====
class DurianDataset(Dataset):
    def __init__(self, jsonl_path, image_root, processor):
        self.image_root = image_root
        self.processor = processor
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_root, item["image"])
        image = Image.open(image_path).convert("RGB")
        
        # 判断是否为空白
        if not item["text"] or item["text"].isspace():
            text_file_path = os.path.splitext(image_path)[0] + ".txt"
            with open(text_file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = item["text"]
        # 替换占位符
        text = text.replace("XXX", item["place_holder"])
        
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
        return {k: v.squeeze(0) for k, v in inputs.items()}, item["place_holder"]

# ==== 对比损失定义 ====
def contrastive_loss(sim_i2t, sim_t2i, temperature=0.07):
    labels = torch.arange(sim_i2t.size(0)).to(sim_i2t.device)
    loss_i2t = nn.CrossEntropyLoss()(sim_i2t / temperature, labels)
    loss_t2i = nn.CrossEntropyLoss()(sim_t2i / temperature, labels)
    return (loss_i2t + loss_t2i) / 2

# ==== 主训练流程 ====
def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("加载模型与 processor...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_model.eval()

    # 自动提取所有 place_holder token
    with open(DATA_JSONL, "r", encoding="utf-8") as f:
        placeholder_tokens = sorted(set(json.loads(line)["place_holder"] for line in f))

    # 初始化 embedding 字典
    projection_dim = clip_model.config.projection_dim
    placeholder_embeddings = {
        token: torch.randn(1, projection_dim, requires_grad=True, device=DEVICE).float()
        for token in placeholder_tokens
    }
    optimizer = torch.optim.Adam(placeholder_embeddings.values(), lr=LR)

    dataset = DurianDataset(DATA_JSONL, IMAGE_ROOT, processor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for batch, tokens in tqdm(loader):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            with torch.no_grad():
                image_features = clip_model.get_image_features(pixel_values)
                text_features = clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

            for i, token in enumerate(tokens):
                text_features[i, 0] = placeholder_embeddings[token]

            sim_i2t = image_features @ text_features.T
            sim_t2i = text_features @ image_features.T
            loss = contrastive_loss(sim_i2t, sim_t2i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Loss: {loss.item():.4f}")

    # 保存所有训练好的 token embedding
    for token, emb in placeholder_embeddings.items():
        torch.save(emb.detach().cpu(), os.path.join(OUTPUT_DIR, f"{token}_embedding.pt"))
    print("训练完成，所有占位符 embedding 已保存。")

if __name__ == "__main__":
    train()
