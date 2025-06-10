import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from scipy.linalg import sqrtm
from tqdm import tqdm
from sklearn.utils import resample

# 配置参数
config = {
    "real_data_path": "D:\\workspace\\real_images",  # 按类别存放的原始图像
    "fake_data_path": "D:\\workspace\\generated_images",  # 按类别存放的生成图像
    "batch_size": 16,
    "bootstrap_iters": 100,
    "bootstrap_size": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "feature_layer": "features",  # EfficientNet-B3的特征层名称
}

# 加载预训练模型（建议在植物数据集上微调）
def load_feature_extractor():
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()  # 移除分类头
    return model.to(config['device']).eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((300, 300)),  # 适应植物叶片长宽比
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载图像数据（按类别组织）
def load_images_by_class(data_path):
    classes = sorted(os.listdir(data_path))
    image_dict = {}
    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        images = []
        for img_file in tqdm(os.listdir(cls_path), desc=f"Loading {cls}"):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(cls_path, img_file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(transform(img))
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")
        image_dict[cls] = torch.stack(images)
    return image_dict

# 提取特征
def extract_features(model, images):
    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), config['batch_size'])):
            batch = images[i:i+config['batch_size']].to(config['device'])
            feat = model(batch).cpu().numpy()
            features.append(feat)
    return np.concatenate(features, axis=0)

# 计算FID
def calculate_fid(real_features, fake_features):
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu_real - mu_fake) ** 2)
    covmean = sqrtm(sigma_real.dot(sigma_fake))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

# Bootstrap采样计算
def bootstrap_fid(real, fake, n_iters=100, sample_size=50):
    fids = []
    for _ in tqdm(range(n_iters), desc="Bootstrapping"):
        real_sample = resample(real, n_samples=sample_size)
        fake_sample = resample(fake, n_samples=sample_size)
        fid = calculate_fid(real_sample, fake_sample)
        fids.append(fid)
    return np.mean(fids), np.percentile(fids, [2.5, 97.5])

# 主流程
def main():
    model = load_feature_extractor()
    
    # 加载数据
    real_images = load_images_by_class(config['real_data_path'])
    fake_images = load_images_by_class(config['fake_data_path'])
    
    # 按类别计算
    results = {}
    for cls in real_images.keys():
        print(f"\nProcessing class: {cls}")
        
        # 提取特征
        real_feat = extract_features(model, real_images[cls])
        fake_feat = extract_features(model, fake_images[cls])
        
        # Bootstrap计算
        mean_fid, conf_interval = bootstrap_fid(real_feat, fake_feat, 
                                              config['bootstrap_iters'], config['bootstrap_size'])
        
        results[cls] = {
            "mean_fid": mean_fid,
            "confidence_interval": conf_interval.tolist()
        }
    
    # 打印结果
    print("\nResults:")
    for cls, res in results.items():
        print(f"{cls}: FID = {res['mean_fid']:.1f} (95% CI: {res['confidence_interval'][0]:.1f}-{res['confidence_interval'][1]:.1f})")

if __name__ == "__main__":
    main()