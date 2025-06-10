import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random
from PIL import Image  # 新增 Pillow 的 Image 模块导入
import warnings
warnings.filterwarnings(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)
# 检查 scikit-learn 是否安装
try:
    from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, pairwise_distances
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.manifold import TSNE
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    raise ImportError("Please install scikit-learn first: pip install scikit-learn")

import matplotlib.pyplot as plt
# 设置中文字体（以 "SimHei" 为例，可替换为其他中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 "-" 显示为方块的问题
import seaborn as sns
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
import cv2

# 配置参数
batch_size = 32  # 根据显存调整


def make_abs_path(path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def MSRCR_normalize(pil_image, sigma_list=[15, 80, 250], alpha=125.0):
    """多尺度Retinex色彩恢复"""
    def single_scale_retinex(img, sigma):
        retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0,0), sigma))
        return retinex

    img = np.array(pil_image)[:, :, ::-1]  # PIL转OpenCV BGR
    img = img.astype(np.float32) + 1.0  # 避免log(0)
    img_ret = np.zeros_like(img)
    
    for sigma in sigma_list:
        retinex = single_scale_retinex(img, sigma)
        img_ret += retinex
    
    img_ret /= len(sigma_list)
    
    # 颜色恢复因子
    beta = 46.0
    gain = alpha * np.log10(beta * img + 1.0)
    img_out = gain * img_ret
    
    # 归一化到0-255
    img_out = (img_out - img_out.min()) / (img_out.max() - img_out.min()) * 255
    normalized = img_out.astype(np.uint8)
    return Image.fromarray(normalized[:, :, ::-1])


def clahe_normalize(pil_image, clip_limit=2.0, grid_size=8):
    """对比度受限的自适应直方图均衡化"""
    img = np.array(pil_image.convert('RGB'))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_clahe = clahe.apply(l)
    
    lab_norm = cv2.merge((l_clahe, a, b))
    rgb_norm = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb_norm)


def extract_yolov8_detect_features(image_dir, weight=fr"E:\\yolov8\\runs\\classify\\train\\weights\\best.pt"):
    from ultralytics import YOLO
    # 自定义特征提取
    class ExtractFeatures(nn.Module):
        def __init__(self, model, including_neck=True):
            super().__init__()
            self.backbone = nn.Sequential(
                *list(model.model.model.children())[:10]
            )
            self.including_neck = including_neck
            # neck部分
            if self.including_neck:
                self.features = {}
                def get_hook(name):
                    def hook_feature_map(module, input, output):
                        # print(f"hook_feature_map==>>{name}==>>", output.shape)
                        self.features[name] = output.detach()
                    return hook_feature_map

                self.backbone[4].register_forward_hook(get_hook(4))
                self.backbone[6].register_forward_hook(get_hook(6))
                self.backbone[9].register_forward_hook(get_hook(9))
                self.upsample1 = model.model.model[10]
                self.concat1 = model.model.model[11]
                self.conv1 = model.model.model[12]
                self.conv1.register_forward_hook(get_hook(12))
                self.upsample2 = model.model.model[13]
                self.concat2 = model.model.model[14]
                self.conv2 = model.model.model[15]
                self.downsample3 = model.model.model[16]
                self.concat3 = model.model.model[17]
                self.conv3 = model.model.model[18]
                self.downsample4 = model.model.model[19]
                self.concat4 = model.model.model[20]
                self.conv4 = model.model.model[21]
        
        def forward(self, x):
            # backbone
            x = self.backbone(x)
            # SPPF特征
            # neck
            if self.including_neck:
                x = self.upsample1(x)
                x = self.concat1([x, self.features[6]])
                x = self.conv1(x)
                x = self.upsample2(x)
                x = self.concat2([x, self.features[4]])
                x = self.conv2(x)
                x = self.downsample3(x)
                x = self.concat3([x, self.features[12]])
                x = self.conv3(x)
                x = self.downsample4(x)
                x = self.concat4([x, self.features[9]])
                x = self.conv4(x)
            # 返回特征
            return x
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(640),
        #
        # transforms.RandomHorizontalFlip(p=0.3),       # 叶片左右对称性较高
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),       # 亮度调整
        # transforms.RandomRotation(degrees=15),
        #
        transforms.CenterCrop(640),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model_path = weight
    model = ExtractFeatures(YOLO(model_path).eval(), including_neck=False)
    # backbone到SPPF层(第九层)，接下来是neck层(第二十一层)
    # print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 提取特征
    features = []
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    random.shuffle(image_files)
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        
        for file in batch_files:
            img = Image.open(file).convert("RGB")
            # 对明亮度干扰进行排除
            # img = MSRCR_normalize(img)
            # img = clahe_normalize(img)
            img = transform(img).unsqueeze(0)
            batch_images.append(img)
        
        batch = torch.cat(batch_images, dim=0).to(device)
        with torch.no_grad():
            feat = model(batch)
            feat = torch.nn.functional.adaptive_max_pool2d(feat, (1, 1)).squeeze()
        features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def extract_yolov8_classify_features(image_dir, weight=fr"E:\\yolov8\\runs\\classify\\train\\weights\\best.pt"):
    from ultralytics import YOLO
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(640),
        #
        # transforms.RandomHorizontalFlip(p=0.3),       # 叶片左右对称性较高
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),       # 亮度调整
        # transforms.RandomRotation(degrees=15),
        #
        transforms.CenterCrop(640),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model_path = weight
    model = YOLO(model_path).eval()
    # 特征提取到第八层，第九层是Classify
    model = nn.Sequential(
        *list(model.model.model.children())[:9]
    )
    # print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 提取特征
    features = []
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    random.shuffle(image_files)
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        
        for file in batch_files:
            img = Image.open(file).convert("RGB")
            # 对明亮度干扰进行排除
            # img = MSRCR_normalize(img)
            # img = clahe_normalize(img)
            img = transform(img).unsqueeze(0)
            batch_images.append(img)
        
        batch = torch.cat(batch_images, dim=0).to(device)
        with torch.no_grad():
            feat = model(batch)
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1)).squeeze()
        features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def extract_resnet_features(image_dir):
    from torchvision.models import resnet18
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(512),
        #
        # transforms.RandomHorizontalFlip(p=0.3),       # 叶片左右对称性较高
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),       # 亮度调整
        # transforms.RandomRotation(degrees=15),
        #
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载预训练的
    model = resnet18(pretrained=True)
    # print(model.fc)
    # 移除全连接层，直接输出 pool3 特征
    model.fc = torch.nn.Identity()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 提取特征
    features = []
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    random.shuffle(image_files)

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        
        for file in batch_files:
            img = Image.open(file).convert("RGB")
            img = transform(img).unsqueeze(0)
            batch_images.append(img)
        
        batch = torch.cat(batch_images, dim=0).to(device)
        with torch.no_grad():
            feat = model(batch)
        features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def extract_plantnet_confidence(image_dir):
    import sys
    sys.path.append(os.path.abspath("./PlantNet-300K"))
    from utils import load_model # type: ignore
    from torchvision.models import resnet50
    filename = os.path.abspath("./PlantNet-300K/weights/resnet50_weights_best_acc.tar")
    model = resnet50(num_classes=1081)
    load_model(model, filename=filename, use_gpu=True)
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad_(False)
        
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 植物图像专用预处理
    plantnet_transform = transforms.Compose([
        transforms.Resize(256),
        #
        # transforms.RandomHorizontalFlip(p=0.3),       # 叶片左右对称性较高
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),       # 亮度调整
        # transforms.RandomRotation(degrees=15),
        #
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

        # 提取特征
    confidence = []
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    random.shuffle(image_files)

    # 分批处理
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        
        # 加载并预处理图像
        for file in batch_files:
            img = Image.open(file).convert("RGB")
            img = plantnet_transform(img)
            batch_images.append(img)
        
        # 转换为 Tensor
        batch = torch.stack(batch_images)
        
        # GPU 加速 (可选)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = batch.to(device)
        model = model.to(device)
        
        # 提取特征
        with torch.no_grad():
            feats = model(batch)
            conf = torch.softmax(feats, dim=1).max(dim=1)[0]
        
        confidence.append(conf.cpu().numpy())
    
    return np.concatenate(confidence, axis=0)


def extract_plantnet_features(image_dir):
    import sys
    sys.path.append(os.path.abspath("./PlantNet-300K"))
    from utils import load_model # type: ignore
    from torchvision.models import resnet50
    filename = os.path.abspath("./PlantNet-300K/weights/resnet50_weights_best_acc.tar")
    model = resnet50(num_classes=1081)
    # from timm import create_model
    # filename = os.path.abspath("./PlantNet-300K/weights/vit_base_patch16_224_weights_best_acc.tar")
    # model = create_model("vit_base_patch16_224", pretrained=None, num_classes=1081)
    load_model(model, filename=filename, use_gpu=True)
    
    # print(model.fc)
    # 移除分类头，保留特征提取器
    model.fc = nn.Identity()  # 输出维度 1536
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad_(False)
        
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 植物图像专用预处理
    plantnet_transform = transforms.Compose([
        transforms.Resize(256),
        #
        # transforms.RandomHorizontalFlip(p=0.3),       # 叶片左右对称性较高
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),       # 亮度调整
        # transforms.RandomRotation(degrees=15),
        #
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 提取特征
    features = []
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    random.shuffle(image_files)
    
    # 分批处理
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        
        # 加载并预处理图像
        for file in batch_files:
            img = Image.open(file).convert("RGB")
            img = plantnet_transform(img)
            batch_images.append(img)
        
        # 转换为 Tensor
        batch = torch.stack(batch_images)
        
        # GPU 加速 (可选)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = batch.to(device)
        model = model.to(device)
        
        # 提取特征
        with torch.no_grad():
            feats = model(batch)
        
        features.append(feats.cpu().numpy())
    
    return np.concatenate(features, axis=0)


# 提取图像特征
def extract_inception_v3_features(image_dir):
    from torchvision.models import inception_v3
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(299),
        #
        # transforms.RandomHorizontalFlip(p=0.3),       # 叶片左右对称性较高
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),       # 亮度调整
        # transforms.RandomRotation(degrees=15),
        #
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载预训练的 Inception V3 模型（仅需 pool3 特征）
    model = inception_v3(pretrained=True, transform_input=False)
    # print(model.fc)
    # 移除全连接层，直接输出 pool3 特征
    model.fc = torch.nn.Identity()
    if hasattr(model, 'AuxLogits'):
        model.AuxLogits.fc = torch.nn.Identity()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 提取特征
    features = []
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    random.shuffle(image_files)
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        
        for file in batch_files:
            img = Image.open(file).convert("RGB")
            img = transform(img).unsqueeze(0)
            # print("Min:", img.min().item(), "Max:", img.max().item())  # 应为 [-1, 1] 或 [0, 1]
            batch_images.append(img)
        
        batch = torch.cat(batch_images, dim=0).to(device)
        with torch.no_grad():
            feat = model(batch)
        features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def extract_features(image_dir, model_type="inception_v3"):
    if model_type == "inception_v3":
        return extract_inception_v3_features(image_dir)
    elif model_type == "plantnet":
        return extract_plantnet_features(image_dir)
    elif model_type == "resnet":
        return extract_resnet_features(image_dir)
    elif model_type == "yolov8_classify":
        return extract_yolov8_classify_features(image_dir, weight=fr"E:\\yolov8\\runs\\classify\\train\\weights\\best.pt")
    elif model_type == "yolov8_detect":
        return extract_yolov8_detect_features(image_dir, weight=fr"E:\\yolov8\\runs\\detect\\train\\weights\\best.pt")
    elif model_type == "dino_v2":
        from tools_freatures import extract_dino_v2_features
        return extract_dino_v2_features(image_dir)
    elif model_type == "clip_vit":
        from tools_freatures import extract_clip_vit_features
        return extract_clip_vit_features(image_dir)
    elif model_type == "vit-base":
        from tools_freatures import extract_vit_features
        return extract_vit_features(image_dir)
    elif model_type == "siglip2":
        from tools_freatures import extract_siglip2_features
        return extract_siglip2_features(image_dir)
    else:
        raise ValueError("Unsupported model. Choose 'inception_v3' or 'plantnet'.")

# 计算 FID
def calculate_fid(real_features, fake_features, epsilon=1e-6):
    """
    改进版 FID 计算函数（数值稳定 + 正则化）
    :param real_features: 真实图像特征，形状 [N, D]
    :param fake_features: 生成图像特征，形状 [M, D]
    :param epsilon: 协方差矩阵正则化项
    :return: FID 值（越小越好）
    """
    from scipy.linalg import sqrtm
    # 计算均值与协方差
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # 协方差矩阵正则化
    sigma_real += epsilon * np.eye(sigma_real.shape[0])
    sigma_fake += epsilon * np.eye(sigma_fake.shape[0])

    # 计算矩阵平方根（确保对称性）
    covmean = sqrtm(sigma_real.dot(sigma_fake))
    covmean = (covmean + covmean.T) / 2  # 强制对称
    covmean = covmean.real  # 确保实数

    # 计算 FID
    diff = mu_real - mu_fake
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid


# 计算 KID（无偏估计）
def calculate_kid(real_features, fake_features, block_size=100, kernel='poly', **kernel_kwargs):
    """
    改进版 KID 计算函数（支持分块计算 + 多种核函数）
    :param real_features: 真实图像特征，形状 [N, D]
    :param fake_features: 生成图像特征，形状 [M, D]
    :param block_size: 分块大小（防止内存溢出）
    :param kernel: 核函数类型 ('poly' 或 'rbf')
    :param kernel_kwargs: 核函数参数（如 degree, gamma）
    :return: KID 值（越小越好）
    """
    # 选择核函数
    kernel_func = polynomial_kernel if kernel == 'poly' else rbf_kernel

    # 分块计算核矩阵
    def block_kernel(X, Y):
        k = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(0, X.shape[0], block_size):
            for j in range(0, Y.shape[0], block_size):
                X_block = X[i:i+block_size]
                Y_block = Y[j:j+block_size]
                k[i:i+block_size, j:j+block_size] = kernel_func(X_block, Y_block, **kernel_kwargs)
        return k

    # 计算各子项
    k_rr = block_kernel(real_features, real_features)
    k_gg = block_kernel(fake_features, fake_features)
    k_rg = block_kernel(real_features, fake_features)

    # 无偏估计计算
    m = real_features.shape[0]
    n = fake_features.shape[0]
    
    term1 = (np.sum(k_rr) - np.trace(k_rr)) / (m * (m - 1)) if m > 1 else 0.0
    term2 = (np.sum(k_gg) - np.trace(k_gg)) / (n * (n - 1)) if n > 1 else 0.0
    term3 = 2 * np.mean(k_rg) if (m > 0 and n > 0) else 0.0
    
    kid = term1 + term2 - term3
    return kid

def calculate_robust_kid(real_features, fake_features):
    # 计算三种核函数结果
    kid_linear = calculate_kid(real_features, fake_features, kernel='poly', degree=1)
    kid_sqrt = calculate_kid(real_features, fake_features, kernel='poly', degree=2)
    kid_cubic = calculate_kid(real_features, fake_features, kernel='poly', degree=3)
    kid_rbf_001 = calculate_kid(real_features, fake_features, kernel='rbf', gamma=0.01)
    kid_rbf_01 = calculate_kid(real_features, fake_features, kernel='rbf', gamma=0.1)
    kid_rbf_1 = calculate_kid(real_features, fake_features, kernel='rbf', gamma=1.0)
    kid_rbf_005 = calculate_kid(real_features, fake_features, kernel='rbf', gamma=0.05)
    kid_rbf_05 = calculate_kid(real_features, fake_features, kernel='rbf', gamma=0.5)
    kid_rbf_08 = calculate_kid(real_features, fake_features, kernel='rbf', gamma=0.8)
    
    # 生成报告
    report = f"""
    KID 多核分析报告：
    - POLY核(degree=1)：{kid_linear:.4f}
    - POLY核(degree=2)：{kid_sqrt:.4f}
    - POLY核(degree=3)：{kid_cubic:.4f}
    - RBF核(gamma=0.01)：{kid_rbf_001:.4f}
    - RBF核(gamma=0.1)：{kid_rbf_01:.4f}
    - RBF核(gamma=1.0)：{kid_rbf_1:.4f}
    - RBF核(gamma=0.05)：{kid_rbf_005:.4f}
    - RBF核(gamma=0.5)：{kid_rbf_05:.4f}
    - RBF核(gamma=0.8)：{kid_rbf_08:.4f}
    """
    return report


def calculate_primary_dims(real_features, fake_features, top_k=5, figure=False):
    # 计算特征贡献度（基于 FID 公式）
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # 各维度对 FID 的贡献
    fid_terms = (mu_real - mu_fake)**2 + np.diag(sigma_real + sigma_fake - 2*np.sqrt(sigma_real@sigma_fake))
    top_k_dims = np.argsort(fid_terms)[-top_k:][::-1]
    if figure: print("FID 主要贡献维度:", top_k_dims)

    # 可视化关键维度分布
    if figure:
        plt.figure(figsize=(12,3))
        nrows = int(np.sqrt(top_k)) + 1
        for i, dim in enumerate(top_k_dims):
            plt.subplot(nrows,nrows,i+1)
            plt.hist(real_features[:,dim], bins=50, alpha=0.5, label='Real')
            plt.hist(fake_features[:,dim], bins=50, alpha=0.5, label='Fake')
            plt.title(f"Dim {dim}")
        plt.tight_layout()
        plt.show()
    #
    return top_k_dims


def caculate_tsne_kl_divergence(real_tsne_features, fake_tsne_features):
    # 计算 t-SNE 低维空间的密度分布
    def compute_distribution(data, bins=30):
        hist, _ = np.histogramdd(data, bins=bins, density=True)
        return hist / np.sum(hist)  # 归一化

    # 计算 KL 散度
    def kl_divergence(p, q):
        p = p + 1e-6  # 避免 log(0)
        q = q + 1e-6
        return np.sum(p * np.log(p / q))

    # 计算密度分布
    real_tsne_dist = compute_distribution(real_tsne_features)
    fake_tsne_dist = compute_distribution(fake_tsne_features)
    # 计算 KL 散度
    kl_score = kl_divergence(real_tsne_dist, fake_tsne_dist)
    return kl_score


def caculate_frechet_distance(real_tsne_features, fake_tsne_features, eps=1e-6):
    # 计算 Fréchet Distance
    def frechet_distance(mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        cov_sqrt = sqrtm(sigma1 @ sigma2)  # 计算协方差的平方根
        # 确保平方根为实数（若仍有微小虚部则取模）
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = np.abs(cov_sqrt)
        else:
            cov_sqrt = cov_sqrt.real
        return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * cov_sqrt)

    # 计算真实和生成数据的均值 & 协方差
    mu_real, sigma_real = np.mean(real_tsne_features, axis=0), np.cov(real_tsne_features, rowvar=False) + eps * np.eye(real_tsne_features.shape[1])
    mu_fake, sigma_fake = np.mean(fake_tsne_features, axis=0), np.cov(fake_tsne_features, rowvar=False) + eps * np.eye(fake_tsne_features.shape[1])

    # 计算 Fréchet Distance
    fd_score = frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fd_score

#n个最近邻
def caculate_nnd(real_features, fake_features, k=5):
    nbrs_real = NearestNeighbors(n_neighbors=k).fit(real_features)
    distances_real, _ = nbrs_real.kneighbors(real_features)
    real_nnd = np.mean(distances_real[:, 1:])  # 忽略自身点

    nbrs_fake = NearestNeighbors(n_neighbors=k).fit(real_features)
    distances_fake, _ = nbrs_fake.kneighbors(fake_features)
    fake_nnd = np.mean(distances_fake)

    return real_nnd, fake_nnd


def caculate_knn(real_features, fake_features):
    # 生成标签：真实数据=0，生成数据=1
    X = np.vstack((real_features, fake_features))
    y = np.hstack((np.zeros(len(real_features)), np.ones(len(fake_features))))
    # 训练KNN分类器
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    knn_acc = accuracy_score(y_test, y_pred)
    return knn_acc


import numpy as np
from scipy.spatial.distance import cdist


def rbf_kernel(X, Y, gamma=1.0):
    """ 计算 RBF 核矩阵 """
    sq_dists = cdist(X, Y, 'sqeuclidean')  # 计算欧几里得距离的平方
    return np.exp(-gamma * sq_dists)  # 应用 RBF 核函数


def compute_mmd(real_features, fake_features, gamma=0.01):
    """
    计算 MMD 值
    :param real_features: 真实数据特征 (numpy array, shape: [m, d])
    :param fake_features: 生成数据特征 (numpy array, shape: [n, d])
    :param gamma: RBF 核参数
    :return: MMD 值
    """
    K_XX = rbf_kernel(real_features, real_features, gamma)
    K_YY = rbf_kernel(fake_features, fake_features, gamma)
    K_XY = rbf_kernel(real_features, fake_features, gamma)

    m = real_features.shape[0]
    n = fake_features.shape[0]

    # 计算 MMD 值，确保非负
    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return max(mmd, 0)  # 避免数值误差导致的负值


# 计算 MMD 值--co
def caculate_mmd(real_features, fake_features, sigma=None):
    """
    计算正确的MMD值
    参数：
        real_features: 真实样本特征矩阵 (n_samples, n_features)
        fake_features: 生成样本特征矩阵 (n_samples, n_features)
        sigma: 高斯核带宽，默认为特征维度的中位数
    """
    def gaussian_kernel(x, y, sigma=1.0):
        """高斯核函数计算"""
        pairwise_dists = cdist(x, y, 'sqeuclidean')  # 计算平方欧氏距离
        return np.exp(-pairwise_dists / (2 * sigma ** 2))

    # 自动确定带宽参数
    if sigma is None:
        pairwise_dist = cdist(real_features, fake_features, 'euclidean')
        sigma = np.median(pairwise_dist)
    
    # 计算各个核矩阵
    xx = gaussian_kernel(real_features, real_features, sigma)
    yy = gaussian_kernel(fake_features, fake_features, sigma)
    xy = gaussian_kernel(real_features, fake_features, sigma)
    
    # MMD²计算
    mmd_squared = (xx.mean() + yy.mean() - 2 * xy.mean())
    
    # 确保非负性
    return np.sqrt(max(mmd_squared, 0))


from torch.utils.data import Dataset
class ResizeDataset(Dataset):
    def __init__(self, img_dir, target_size=299):
        self.img_dir = img_dir
        self.target_size = target_size
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.3),       # 叶片左右对称性较高
            transforms.ColorJitter(brightness=0.1),       # 轻微亮度调整
            transforms.CenterCrop(target_size),
            transforms.PILToTensor()
        ])
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img


def umap_v(real_features, fake_features):
    import umap
    scaler = StandardScaler()
    # 合并数据并标准化
    combined = np.concatenate([real_features, fake_features])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined)

    # UMAP参数优化配置
    umap_params = {
        'n_neighbors': 15,        # 控制局部与全局结构的平衡（建议范围10-50）
        'min_dist': 0.1,          # 点间距压缩程度（0.0紧密~0.99松散）
        'n_components': 2,        # 输出维度
        'metric': 'cosine',    # 距离度量（可选euclidean、cosine等）
        'random_state': 42,        # 随机种子
        'transform_seed': 42,
        'verbose': False
    }

    # 创建UMAP模型并拟合
    reducer = umap.UMAP(**umap_params)
    embedding = reducer.fit_transform(scaled_features)

    # 分割结果
    real_emb = embedding[:len(real_features)]
    fake_emb = embedding[len(real_features):]

    # 可视化设置
    plt.figure(figsize=(12, 8))
    plt.scatter(real_emb[:, 0], real_emb[:, 1], 
                c='blue', alpha=0.6, label='Real Data', s=10)
    plt.scatter(fake_emb[:, 0], fake_emb[:, 1], 
                c='red', alpha=0.6, label='Generated Data', s=10)

    plt.title('UMAP Visualization of Feature Space', fontsize=14)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(markerscale=2)
    plt.grid(alpha=0.3)
    plt.show()


def tsne_v(real_features, fake_features, caption=""):
     # 计算 t-SNE
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    all_features = np.vstack([real_features, fake_features])
    tsne_results = tsne.fit_transform(all_features)
    # 画图
    real_tsne_results = tsne_results[:len(real_features)]
    fake_tsne_results = tsne_results[len(real_features):]
    #
    # kl散度
    kl_score = caculate_tsne_kl_divergence(real_tsne_results, fake_tsne_results)
    # FD
    fd_score = caculate_frechet_distance(real_tsne_results, fake_tsne_results)
    print(f"""
    KL散度(真实vs生成): {kl_score:.4f}
    FID(t-SNE): {fd_score:.4f}
    """)
    # t-SNE可视化
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=real_tsne_results[:, 0], y=real_tsne_results[:, 1], label="Real", alpha=0.6)
    sns.scatterplot(x=fake_tsne_results[:, 0], y=fake_tsne_results[:, 1], label="Generated", alpha=0.6)
    plt.title(f"{caption} t-SNE ")
    plt.show()


def select_diverse_samples(features, k):
    """
    选择 k 个多样性样本
    """
    n = features.shape[0]
    if n <= k:
        return features
    
    # 归一化特征
    features_normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
    sim_matrix = features_normalized @ features_normalized.T  # 余弦相似度矩阵
    sim_sums = sim_matrix.sum(axis=1)
    
    # 选择相似度总和最小的 k 个样本，差异越大
    indices = np.argpartition(sim_sums, k)[:k]
    # 选择相似度总和最大的 k 个样本，差异越小
    # indices = np.argpartition(sim_sums, k)[-k:]
    return features[indices]


# 主流程
if __name__ == "__main__":
    #Leaf_Blight
    # real_image_dir = "D:\\workspace\\real_images\\Leaf_Blight"  # 替换为真实图像路径
    # fake_image_dir = "D:\\workspace\\generated_images\\Leaf_Blight"  # 替换为生成图像路径
    #Algal_Leaf_Spot
    # real_image_dir = "D:\\workspace\\real_images\\Algal_Leaf_Spot"  # 替换为真实图像路径
    # fake_image_dir = "D:\\workspace\\generated_images\\Algal_Leaf_Spot"  # 替换为生成图像路径
    #Leaf_Spot
    # real_image_dir = "D:\\workspace\\real_images\\Leaf_Spot"  # 替换为真实图像路径
    # fake_image_dir = "D:\\workspace\\generated_images\\Leaf_Spot"  # 替换为生成图像路径
    #Leaf_Spot
    # real_image_dir = "D:\\workspace\\real_images\\No_Disease"  # 替换为真实图像路径
    # fake_image_dir = "D:\\workspace\\generated_images\\No_Disease"  # 替换为生成图像路径
    # 选择特征提取模型
    # select_model_type = "inception_v3"
    # select_model_type = "plantnet"
    # select_model_type = "resnet"
    select_model_type = "yolov8_classify"
    # select_model_type = "yolov8_detect"
    # select_model_type = "dino_v2"
    # select_model_type = "clip_vit"
    # select_model_type = "vit-base"
    # select_model_type = "siglip2"
    # 特征选择分类列表：0=Leaf_Blight， 1=Algal_Leaf_Spot， 2=Leaf_Spot， 3=No_Disease，全选[0, 1, 2, 3]
    # choice_image_dirs = [0,1,2,3]
    choice_image_dirs = [3]
    # 提取特征
    real_image_dirs = [
        "D:\\workspace\\real_images\\Leaf_Blight",
        "D:\\workspace\\real_images\\Algal_Leaf_Spot",
        "D:\\workspace\\real_images\\Leaf_Spot",
        "D:\\workspace\\real_images\\No_Disease"
    ]
    real_features = []
    for real_image_dir in [real_image_dirs[idx] for idx in choice_image_dirs]:
        real_features.append(extract_features(real_image_dir, model_type=select_model_type))
    real_features = np.concatenate(real_features, axis=0)
    print("real_features.shape===>>>", real_features.shape)
    # 正常情况应呈现自然分布，而非恒定值
    # workdir_root = "D:\\workspace"
    workdir_root = "D:\\workspace_SDXL"
    # workdir_root = "D:\\workspace_SD35"
    fake_image_dirs = [
        # f"{workdir_root}\\generated_images\\Leaf_Blight",
        f"{workdir_root}\\generated_images\\Leaf_Blight",
        # f"{workdir_root}\\generated_images\\Algal_Leaf_Spot",
        # f"{workdir_root}\\generated_images\\Algal_Leaf_Spot_delight",
        f"{workdir_root}\\generated_images\\Algal_Leaf_Spot",
        # f"{workdir_root}\\generated_images\\Leaf_Spot_delight",
        # f"{workdir_root}\\generated_images\\Leaf_Spot",
        f"{workdir_root}\\generated_images\\Leaf_Spot",
        # f"{workdir_root}\\generated_images\\No_Disease",
        f"{workdir_root}\\generated_images\\No_Disease"
    ]
    fake_features = []
    for fake_image_dir in [fake_image_dirs[idx] for idx in choice_image_dirs]:
        fake_features.append(extract_features(fake_image_dir, model_type=select_model_type))
    fake_features = np.concatenate(fake_features, axis=0)
    # print("fake_features.shape===>>>", fake_features.shape)
    fake_features = select_diverse_samples(fake_features, min(int(3*len(real_features)/2), len(fake_features)))
    print("fake_features.shape===>>>", fake_features.shape)
    # 选择tsne的标题
    tsne_caption = os.path.basename(real_image_dirs[choice_image_dirs[0]]) if len(choice_image_dirs) == 1 else "ALL"
    # 选择特征标准化工具
    scaler = StandardScaler()
    scaler.fit(real_features)
    real_features = scaler.transform(real_features)
    fake_features = scaler.transform(fake_features)
    # 
    # tsne_v(real_features, fake_features, caption=tsne_caption)
    # umap_v(real_features, fake_features)
    #
    # 检查特征向量模长
    real_feature_norms = np.linalg.norm(real_features, axis=1)
    fake_feature_norms = np.linalg.norm(fake_features, axis=1)
    # 测试特征提取
    half_len = len(real_features) // 2
    fid_real_vs_real = calculate_fid(real_features[:half_len], real_features[half_len:])

    # 计算生成样本间多样性
    fake_diversity = np.mean(pairwise_distances(fake_features, metric='cosine'))
    real_diversity = np.mean(pairwise_distances(real_features, metric='cosine'))
    fake_real_diversity = np.mean(pairwise_distances(fake_features, real_features, metric='cosine'))
    print(f"""
    真实数据特征模长范围: {np.min(real_feature_norms):.2f} ~ {np.max(real_feature_norms):.2f}
    真实数据自身的FID(真实数据分成两半)：{fid_real_vs_real:.2f}
    生成数据特征模长范围：{np.min(fake_feature_norms):.2f} ~ {np.max(fake_feature_norms):.2f}
    生成多样性: {fake_diversity:.3f} vs 真实多样性: {real_diversity:.3f} (生成多样性 ≈ 真实多样性 ±0.05)
    fake_real_diversity: {fake_real_diversity:.3f}
    """)
    
    # 测试指标
    print(calculate_robust_kid(real_features, fake_features).replace("KID 多核分析报告", "KID 多核分析报告(真实vs生成数据)"))
    top_k_dims = calculate_primary_dims(real_features, fake_features, top_k=64, figure=False)
    top_k_fid = calculate_fid(real_features[:, top_k_dims], fake_features[:, top_k_dims])

    from tools_index import compute_kid
    kid_mean, kid_std = compute_kid(real_features, fake_features, subset_size=64)
    
    # 计算指标
    fid = calculate_fid(real_features, fake_features)
    pca = PCA(n_components=min(len(real_features), len(fake_features))-1, random_state=42)
    pca.fit(real_features)
    pca_fid = calculate_fid(pca.transform(real_features), pca.transform(fake_features))
    pca_whiten = PCA(n_components=min(len(real_features), len(fake_features))-1, random_state=42, whiten=True)
    pca_whiten.fit(real_features)
    pca_whiten_fid = calculate_fid(pca_whiten.transform(real_features), pca_whiten.transform(fake_features))
    print(f"""
    FID(all): {fid:.2f}
    FID(top_k): {top_k_fid:.2f}
    FID(PCA): {pca_fid:.2f}
    FID(PCA_whiten): {pca_whiten_fid:.2f}
    KID(kid_batch): {kid_mean:.2f} ± {kid_std:.2f}
    """)
    print(f"""
    生成多样性: {fake_diversity:.3f} vs 真实多样性: {real_diversity:.3f} (生成多样性 ≈ 真实多样性 ±0.05)
    fake_real_diversity: {fake_real_diversity:.3f}
    """)
    real_nnd, fake_nnd = caculate_nnd(real_features, fake_features)
    knn_acc = caculate_knn(real_features, fake_features)
    mmd_value = compute_mmd(real_features, fake_features, gamma=0.01)
    mmd_score = caculate_mmd(real_features, fake_features)
    print(f"""
    真实数据NND: {real_nnd:.4f}, 生成数据NND: {fake_nnd:.4f}
    KNN真假分类精度: {knn_acc:.4f}
    MMD值wendy: {mmd_value:.4f}
    MMDScore: {mmd_score:.4f}
    """)

    # 计算特征中心
    real_center = real_features.mean(axis=0)
    fake_center = fake_features.mean(axis=0)
    # 计算欧几里得距离
    feature_shift = np.linalg.norm(real_center - fake_center)

    # 计算生成数据 vs 真实数据的最近邻距离
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(real_features)
    distances, _ = nbrs.kneighbors(fake_features)
    # 计算平均最近邻距离
    mean_nnd = np.mean(distances)
    print(f"""
    真假数据特征中心偏移: {feature_shift:.2f}
    平均最近邻(NND): {mean_nnd:.2f}
    """)

    # real_confidence = extract_plantnet_confidence(real_image_dir)
    # fake_confidence = extract_plantnet_confidence(fake_image_dir)
    # # 计算均值和标准差
    # real_mean_conf = np.linalg.norm(real_confidence.mean())
    # fake_mean_conf = np.linalg.norm(fake_confidence.mean())
    # real_std_conf = np.linalg.norm(real_confidence.std())
    # fake_std_conf = np.linalg.norm(fake_confidence.std())
    # print(f"真实数据置信度(plantnet): Mean={real_mean_conf:.4f}, Std={real_std_conf:.4f}")
    # print(f"生成数据置信度(plantnet): Mean={fake_mean_conf:.4f}, Std={fake_std_conf:.4f}")

    # from torch_fidelity import calculate_metrics
    # metrics = calculate_metrics(
    #     input1=ResizeDataset(img_dir=real_image_dir, target_size=299), 
    #     input2=ResizeDataset(img_dir=fake_image_dir, target_size=299), 
    #     cuda=True, kid_subset_size=100, 
    #     isc=True, fid=True, kid=True, prc=True
    # )
    
    tsne_v(real_features, fake_features, caption=tsne_caption)
    # umap_v(real_features, fake_features)
