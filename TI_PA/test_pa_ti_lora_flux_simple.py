#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联合推理测试脚本：Textual Inversion (TI) + Prompt Adapter (PA) + LoRA
- 注入 durian leaf 病害专属 embedding (TI)
- 通过 Prompt Adapter 调节语义向量 (PA)
- 加载 榴莲叶疾病 微调LoRA 权重
- 使用 FLUX.1-dev 生成图像
- 使用 CLIP 相似度进行结果评估
"""

import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from diffusers import DiffusionPipeline
from tqdm import tqdm
import torch.nn as nn

# ==== 参数设置 ====
TI_EMBEDDING_DIR = "./ti_output"
PA_ADAPTER_PATH = "./pa_output/prompt_adapter.pth"
OUTPUT_DIR = "./joint_infer"
PLACEHOLDER_PROMPTS = {
    "algal_leaf_spot": "A algal_leaf_spot durian leaf with visible symptoms",
    "leaf_blight": "A leaf_blight durian leaf with wilting tissue",
    "leaf_spot": "A leaf_spot durian leaf with small dark lesions",
    "no_disease": "A no_disease durian leaf with uniform color"
}
FLUX_MODEL = "black-forest-labs/FLUX.1-dev"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_PATHS = [
    "./lora_output/no_disease_r64.safetensors",
    "./lora_output/algal_leaf_spot_r64.safetensors",
]
LORA_WEIGHT = 0.65  # LoRA 权重缩放
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== PromptAdapter 定义 ====
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

# ==== 注入 Textual Inversion embedding ====
def inject_ti_embedding(pipe, tokenizer, token, embedding_path):
    emb = torch.load(embedding_path).to(DEVICE)
    if token not in tokenizer.get_vocab():
        tokenizer.add_tokens([token])
        pipe.text_encoder.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids(token)
    with torch.no_grad():
        pipe.text_encoder.get_input_embeddings().weight[token_id] = emb

# ==== CLIP 相似度计算 ====
def clip_similarity(img1, img2, clip_model, processor):
    inputs = processor(images=[img1, img2], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return (feats[0] @ feats[1].T).item()

# ==== 联合推理主函数 ====
def run():
    print("加载 FLUX, CLIP, Adapter, Token Embedding...")
    pipe = DiffusionPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.float16, variant="fp16").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(FLUX_MODEL)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

    for lora_path in LORA_PATHS:
        if os.path.exists(lora_path) or '/' in lora_path:
            pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors" if "safetensors" not in lora_path else None)
            print(f"已加载 LoRA: {lora_path}")
        else:
            print(f"跳过 LoRA 路径: {lora_path}，文件不存在。")

    adapter = PromptAdapter(dim=pipe.text_encoder.config.hidden_size).to(DEVICE)
    adapter.load_state_dict(torch.load(PA_ADAPTER_PATH, map_location=DEVICE))
    adapter.eval()

    for token, prompt in tqdm(PLACEHOLDER_PROMPTS.items()):
        emb_path = os.path.join(TI_EMBEDDING_DIR, f"{token}_embedding.pt")
        if not os.path.exists(emb_path):
            print(f"跳过 {token}，未找到 TI embedding。")
            continue

        # 注入 textual inversion embedding
        inject_ti_embedding(pipe, tokenizer, token, emb_path)

        # 生成图像 with prompt → embed → adapter → image
        text_input = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            raw_embed = pipe.text_encoder(**text_input).last_hidden_state
            adapted_embed = adapter(raw_embed)
            image = pipe(prompt_embeds=adapted_embed).images[0]

        save_path = os.path.join(OUTPUT_DIR, f"gen_joint_{token}.png")
        image.save(save_path)
        print(f"图像保存：{save_path}")

        # 相似度评估
        ref_path = os.path.join("eval_targets", f"target_{token}.jpg")
        if os.path.exists(ref_path):
            ref = Image.open(ref_path).convert("RGB")
            sim = clip_similarity(image, ref, clip_model, clip_processor)
            print(f"[{token}] CLIP 相似度 = {sim:.4f}")

if __name__ == "__main__":
    run()

