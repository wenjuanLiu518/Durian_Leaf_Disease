import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
# 设置中文字体（以 "SimHei" 为例，可替换为其他中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 "-" 显示为方块的问题

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def make_abs_path(path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

def heatmap_classfy(image_path):
    from ultralytics import YOLO
    # --------------------- 自定义模型包装类 ---------------------
    class YOLOWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model  # 获取底层PyTorch模型
            
        def forward(self, x):
            with torch.enable_grad():
                return self.model(x)[0]  # 取第一个输出项（分类分数）
    
    model_path = fr"E:\\yolov8\\runs\\classify\\train\\weights\\best.pt"
    model = YOLO(model_path)
    model.eval()
    # --------------------- 正确选择目标层 ---------------------
    target_layer = model.model.model[-2]   # 分类器Classify的前一层
    # 要求grad信息输出，这是热力图的权重值
    for param in target_layer.parameters():
        param.requires_grad = True
    # --------------------- 自定义目标函数 ---------------------
    class YOLOTarget:
        def __init__(self, class_id):
            self.class_id = class_id
            
        def __call__(self, model_output):
            # 从YOLO的输出元组中提取分类分数
            if self.class_id >= 0 and self.class_id < len(model_output):
                return model_output[self.class_id]  # 返回指定类别的分数
            elif self.class_id >= len(model_output):
                raise IndexError("out of range")
            else:
                _, indices = torch.max(model_output, dim=0)
                return model_output[indices]

    # --------------------- 图像预处理 ---------------------
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(rgb_img, (640, 640))  # YOLOv8默认输入尺寸
    input_tensor = torch.from_numpy(img_resized / 255.0).permute(2,0,1).unsqueeze(0).float()
    # --------------------- 初始化Grad-CAM ---------------------
    with torch.enable_grad():
        wrapped_model = YOLOWrapper(model)
        grad_cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
        targets = [YOLOTarget(class_id=-1)]     # -1表示自动选择分数最高的分类
        grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=False)
    # --------------------- 可视化叠加 ---------------------
    # 将热力图缩放到原始图像尺寸
    heatmap = cv2.resize(grayscale_cam[0], (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # 叠加热力图与原始图像 (透明度0.5)
    superimposed_img = cv2.addWeighted(img, 0.5, heatmap_colored, 0.5, 0)
    # 保存结果
    name, suffix = os.path.splitext(os.path.basename(image_path))
    save_image_path = image_path.replace(f"{name}{suffix}", f"{name}_cam_classfy{suffix}")
    cv2.imwrite(save_image_path, superimposed_img)
    # 显示结果
    # rgb_img = rgb_img.astype(np.float32) / 255.0
    # plt.imshow(show_cam_on_image(rgb_img, heatmap))
    # plt.axis('off')
    # plt.show()


def heatmap_detect(image_path):
    from ultralytics import YOLO
    # --------------------- 自定义模型包装类 ---------------------
    class YOLOWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model
            # print(self.model.names)
        
        def forward(self, x):
            # 前向传播至目标层
            with torch.enable_grad():
                return self.model(x)
    
    model_path = fr"E:\\yolov8\\runs\\detect\\train\\weights\\best.pt"
    model = YOLO(model_path)
    model.eval()
    # --------------------- 正确选择目标层 ---------------------
    target_layers = [
        model.model.model[15],      # 80x80特征
        model.model.model[18],      # 40x40特征
        model.model.model[21],      # 20x20特征
    ]  # 最后一个主干特征层
    # 要求grad信息输出，这是热力图的权重值
    for target_layer in target_layers:
        for param in target_layer.parameters():
            param.requires_grad = True
    # --------------------- 自定义目标函数 ---------------------
    class YOLOTarget:
        def __init__(self, class_id=0):
            self.class_id = class_id  # 关注指定类别
            
        def __call__(self, model_output):
            # print("完整输出形状:", model_output.shape)  # 期望 (1, 8, 8400)
            # 提取分类分数（跳过 xywh(4) + obj(1)）
            cls_scores = model_output[0, 4:, :]  # (4, 8400)
            # print("分类分数形状:", cls_scores.shape)  # 期望 (4, 8400)

            if self.class_id < 0:
                return torch.max(cls_scores)
            else:
                return torch.max(cls_scores[self.class_id, :])
    # --------------------- 图像预处理 ---------------------
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(rgb_img, (640, 640))  # YOLOv8默认输入尺寸
    input_tensor = torch.from_numpy(img_resized / 255.0).permute(2,0,1).unsqueeze(0).float()
    # --------------------- 初始化Grad-CAM ---------------------
    with torch.enable_grad():
        wrapped_model = YOLOWrapper(model)
        grad_cam = GradCAM(model=wrapped_model, target_layers=target_layers)
        targets = [YOLOTarget(class_id=-1)]     # -1表示自动选择分数最高的分类
        grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=False)
    # --------------------- 可视化叠加 ---------------------
    # 将热力图缩放到原始图像尺寸
    heatmap = cv2.resize(grayscale_cam[0], (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # 叠加热力图与原始图像 (透明度0.5)
    superimposed_img = cv2.addWeighted(img, 0.5, heatmap_colored, 0.5, 0)
    # 保存结果
    name, suffix = os.path.splitext(os.path.basename(image_path))
    save_image_path = image_path.replace(f"{name}{suffix}", f"{name}_cam_detect{suffix}")
    cv2.imwrite(save_image_path, superimposed_img)
    # 显示结果
    # rgb_img = rgb_img.astype(np.float32) / 255.0
    # plt.imshow(show_cam_on_image(rgb_img, heatmap))
    # plt.axis('off')
    # plt.show()


if __name__ == "__main__":
    image_path = make_abs_path(fr"heatmap_result\\morning_leaf_spot_00043_.png")
    heatmap_classfy(image_path)
    heatmap_detect(image_path)