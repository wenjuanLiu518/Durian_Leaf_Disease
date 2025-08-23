#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Adapter 推理与评估脚本（多类别）
- 加载训练好的 adapter 参数
- 使用 FLUX.1-dev 进行多类别 prompt 图像生成
- 进行 CLIP 相似度评估（与参考图对比）
"""

import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from diffusers import DiffusionPipeline
from tqdm import tqdm
import torch.nn as nn

# ==== 配置 ====
ADAPTER_PATH = "./pa_output/prompt_adapter.pth"
OUTPUT_DIR = "./pa_output/infer_results"
PLACEHOLDER_PROMPTS = {
    "algal_leaf_spot": "A algal_leaf_spot durian leaf with visible symptoms",
    "leaf_blight": "A leaf_blight durian leaf with wilting tissue",
    "leaf_spot": "A leaf_spot durian leaf with small dark lesions",
    "no_disease": "A no_disease durian leaf with uniform color"
}
FLUX_MODEL = "black-forest-labs/FLUX.1-dev"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== Prompt Adapter 定义 ====
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

# ==== CLIP 相似度计算 ====
def clip_similarity(img1, img2, clip_model, processor):
    inputs = processor(images=[img1, img2], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return (feats[0] @ feats[1].T).item()

# ==== 推理流程 ====
def run():
    print("加载模型与 adapter...")
    pipe = DiffusionPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.float16, variant="fp16").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(FLUX_MODEL)
    adapter = PromptAdapter(dim=pipe.text_encoder.config.hidden_size).to(DEVICE)
    adapter.load_state_dict(torch.load(ADAPTER_PATH, map_location=DEVICE))
    adapter.eval()

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

    for token, prompt in tqdm(PLACEHOLDER_PROMPTS.items()):
        # 获取 text embedding 并做适配
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            orig_embed = pipe.text_encoder(**inputs)[0]
            adapted_embed = adapter(orig_embed)
            image = pipe(prompt_embeds=adapted_embed).images[0]

        out_path = os.path.join(OUTPUT_DIR, f"gen_{token}.png")
        image.save(out_path)
        print(f"生成图像：{out_path}")

        ref_path = os.path.join("./eval_targets", f"target_{token}.jpg")
        if os.path.exists(ref_path):
            ref_img = Image.open(ref_path).convert("RGB")
            sim = clip_similarity(image, ref_img, clip_model, clip_processor)
            print(f"[{token}] 相似度 = {sim:.4f}")

if __name__ == "__main__":
    run()
