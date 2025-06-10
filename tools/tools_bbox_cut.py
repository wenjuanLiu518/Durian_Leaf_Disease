import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import numpy as np
import tqdm

def make_abs_path(path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

def load_from(work_dir):
    pair_list = []
    for img_path in [os.path.join(work_dir, f) for f in os.listdir(work_dir) if f.endswith(('.png', '.jpg'))]:
        # 过滤没有打标签的文件
        xml_path = img_path.replace(".jpg", ".xml").replace(".png", ".xml")
        if not os.path.exists(xml_path):
            continue
        # 读取xml标签文件
        pair_list.append(tuple([img_path, xml_path]))
    # 返回jpg, xml数据
    return pair_list

def read_bbox(xml_path):
    bbox_list = []
    # 从文件解析
    root = ET.parse(xml_path).getroot()
    # 获取图片尺寸
    size_node = root.find("size")
    width = int(size_node.find("width").text)
    height = int(size_node.find("height").text)
    # 获取bbox尺寸
    for obj_node in root.findall("object"):
        bndbox_node = obj_node.find("bndbox")
        xmin = int(bndbox_node.find("xmin").text)
        ymin = int(bndbox_node.find("ymin").text)
        xmax = int(bndbox_node.find("xmax").text)
        ymax = int(bndbox_node.find("ymax").text)
        # bbox记录
        bbox_list.append(tuple([xmin, ymin, xmax, ymax, width, height]))
    # 返回数据记录
    return bbox_list

def filter_and_padding(b, w, h, p=64):
    # 过滤太小的bbox宽高
    b_width = (b[2] - b[0]) if (b[2] > b[0]) else 0
    b_height = (b[3] - b[1]) if (b[3] > b[1]) else 0
    if b_width <= 64 or b_height <= 64:
        return None
    # 小尺寸减半padding
    if b_width <= 256 or b_height <= 256:
        p = p / 2
    if b_width >= 512 and b_height >= 512:
        p = p * 2
    if b_width >= 1024 and b_height >= 1024:
        p = p * 4
    # 处理padding
    b_left = max(0, b[0]-p)
    b_top = max(0, b[1]-p)
    b_right = min(w, b[2]+p)
    b_bottom = min(h, b[3]+p)
    # 返回bbox
    return tuple([b_left, b_top, b_right, b_bottom])

# 调整threshold筛选清晰图片
def is_blur_image(cropped_image, threshold=64.0, resize_to=(256, 256)):
    # 转为灰度图并缩小尺寸（加速计算）
    gray = np.array(cropped_image.convert('L').resize(resize_to))
    # 使用 OpenCV 的 Sobel 算子计算梯度
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # x方向梯度
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # y方向梯度
    # 计算梯度幅值的均值
    score = np.mean(np.sqrt(gx**2 + gy**2))
    # 打印结果
    print(f"{score:.2f} < {threshold}")
    return score < threshold  # 值越小越模糊

def bbox_cut(pair_list, cropped_dir):
    cropped_list = []
    for img_path, xml_path in tqdm.tqdm(pair_list, total=len(pair_list), desc="bbox截图"):
        bbox_list = read_bbox(xml_path=xml_path)
        img_src = Image.open(img_path).convert("RGB")
        # 遍历bbox
        basename = os.path.basename(img_path)
        for bbox_idx, bbox_src in enumerate(bbox_list):
            b = bbox_src[:4]
            w, h = bbox_src[-2:]
            b = filter_and_padding(b, w, h)
            if b is not None and not is_blur_image(img_src.crop(b), resize_to=(512, 512), threshold=1.0):
                idx_basename = basename.replace(".jpg", f"_{bbox_idx}.jpg")
                crop = img_src.crop(box=b)
                cropped_path = os.path.join(cropped_dir, idx_basename)
                crop.save(cropped_path)
                # 记录裁剪图片
                cropped_list.append(cropped_path)
        # 处理下一个box
    print(f"共计截图{len(cropped_list)}张")
    # 处理下一张图片
    return cropped_list

if __name__ == "__main__":
    work_dir = "E:\\已完成(H)"
    src_pair_list = load_from(work_dir=work_dir)
    # 清空缓存
    cropped_dir = os.path.join(work_dir, "cropped")
    shutil.rmtree(cropped_dir, ignore_errors=True)
    os.makedirs(cropped_dir, exist_ok=True)
    # bbox裁剪
    bbox_cut(src_pair_list, cropped_dir)
