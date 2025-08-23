#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Textual Inversion 多类别推理与评估脚本（多类别）
- 支持多个 token embedding 加载与注入
- 使用 FLUX.1-dev 模型生成图像
- 使用 CLIP 相似度评估与可视化输出
"""

import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from diffusers import DiffusionPipeline
from tqdm import tqdm

# ==== 参数设置 ====
EMBEDDING_DIR = "./ti_output"             # 保存多个 *_embedding.pt 的目录
OUTPUT_DIR = "./ti_output/infer_results"  # 输出生成图像
PLACEHOLDER_PROMPTS = {
    "algal_leaf_spot": "A algal_leaf_spot durian leaf with visible symptoms",
    "leaf_blight": "A leaf_blight durian leaf with wilting tissue",
    "leaf_spot": "A leaf_spot durian leaf with small dark lesions",
    "no_disease": "A no_disease durian leaf with uniform color"
}
FLUX_MODEL = "black-forest-labs/FLUX.1-dev"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== 注入 embedding ====
def inject_token_embedding(pipe, tokenizer, token: str, emb_path: str):
    emb = torch.load(emb_path).to(DEVICE)
    if token not in tokenizer.get_vocab():
        tokenizer.add_tokens([token])
        pipe.text_encoder.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids(token)
    with torch.no_grad():
        pipe.text_encoder.get_input_embeddings().weight[token_id] = emb
    return token_id

# ==== CLIP 相似度计算 ====
def clip_similarity(img1, img2, clip_model, processor):
    inputs = processor(images=[img1, img2], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return (feats[0] @ feats[1].T).item()

# ==== 推理流程 ====
def run():
    print("加载 FLUX 与 CLIP 模型...")
    pipe = DiffusionPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.float16, variant="fp16").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(FLUX_MODEL)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

    for token, prompt in tqdm(PLACEHOLDER_PROMPTS.items()):
        emb_file = os.path.join(EMBEDDING_DIR, f"{token}_embedding.pt")
        if not os.path.exists(emb_file):
            print(f"[跳过] 未找到 embedding：{token}")
            continue

        # 注入 embedding
        inject_token_embedding(pipe, tokenizer, token, emb_file)

        # 生成图像
        image = pipe(prompt=prompt, num_inference_steps=30).images[0]
        image_path = os.path.join(OUTPUT_DIR, f"gen_{token}.png")
        image.save(image_path)
        print(f"生成图像：{image_path}")

        # 可选：与参考图对比（若存在）
        ref_path = os.path.join("./eval_targets", f"target_{token}.jpg")
        if os.path.exists(ref_path):
            ref_img = Image.open(ref_path).convert("RGB")
            sim = clip_similarity(image, ref_img, clip_model, clip_processor)
            print(f"[{token}] CLIP 相似度：{sim:.4f}")

if __name__ == "__main__":
    run()
