import os
import numpy as np
import torch
from PIL import Image
import warnings
warnings.filterwarnings(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32  # 根据显存调整

# repo_id = "facebook/dinov2-large"
def extract_dino_v2_features(image_dir, repo_id="facebook/dinov2-base"):
    # 加载DINOv2模型和处理器
    dinov2_processor = AutoImageProcessor.from_pretrained(repo_id, use_fast=False)
    dinov2_model = AutoModel.from_pretrained(repo_id).to(device)

    # 提取特征
    features = []
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        
        batch_images = []
        # 加载原始图像
        for file in batch_files:
            img = Image.open(file).convert("RGB")
            batch_images.append(img)

        batch_images = dinov2_processor(images=batch_images, return_tensors="pt")['pixel_values'].to(device)
        # DINOv2特征提取
        with torch.no_grad():
            feat = dinov2_model(batch_images).last_hidden_state.mean(dim=1)  # 全局平均池化
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


# repo_id="openai/clip-vit-base-patch32"
def extract_clip_vit_features(image_dir, repo_id="openai/clip-vit-large-patch14"):
    # 加载CLIP模型和处理器
    clip_preprocess = CLIPProcessor.from_pretrained(repo_id, use_fast=False)
    clip_model = CLIPModel.from_pretrained(repo_id).to(device)

    # 提取特征
    features = []
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        
        batch_images = []
        # 加载原始图像
        for file in batch_files:
            img = Image.open(file).convert("RGB")
            batch_images.append(img)

        batch_images = clip_preprocess(images=batch_images, return_tensors="pt")['pixel_values'].to(device)
        # CLIP特征提取
        with torch.no_grad():
            feat = clip_model.get_image_features(batch_images)
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)

# repo_id="google/vit-base-patch16-224"
def extract_vit_features(image_dir, repo_id="google/vit-base-patch16-384", use_cls_token=False):
    # 初始化ViT模型
    vit_processor = AutoImageProcessor.from_pretrained(repo_id, use_fast=False)
    vit_model = AutoModel.from_pretrained(repo_id).to(device)

    # 提取特征
    features = []
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        
        batch_images = []
        # 加载原始图像
        for file in batch_files:
            img = Image.open(file).convert("RGB")
            batch_images.append(img)
        
        batch_images = vit_processor(images=batch_images, return_tensors="pt")['pixel_values'].to(device)
        # ViT特征提取
        with torch.no_grad():
            if use_cls_token:
                feat = vit_model(batch_images).last_hidden_state[:, 0, :]  # 使用CLS token
            else:
                feat = vit_model(batch_images).last_hidden_state.mean(dim=1)  # 全局平均池化
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


# repo_id="google/siglip2-base-patch16-224"
def extract_siglip2_features(image_dir, repo_id="google/siglip2-base-patch16-384"):
    # 初始化ViT模型
    siglip_processor = AutoImageProcessor.from_pretrained(repo_id, use_fast=False)
    siglip_model = AutoModel.from_pretrained(repo_id).to(device)

    # 提取特征
    features = []
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        
        batch_images = []
        # 加载原始图像
        for file in batch_files:
            img = Image.open(file).convert("RGB")
            batch_images.append(img)
        
        batch_images = siglip_processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        # ViT特征提取
        with torch.no_grad():
            feat = siglip_model.get_image_features(**batch_images)
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


if __name__ == "__main__":
    real_image_dir = "D:\\workspace\\real_images\\Leaf_Blight"
    vision_features = extract_siglip2_features(real_image_dir)
    print(vision_features.shape)