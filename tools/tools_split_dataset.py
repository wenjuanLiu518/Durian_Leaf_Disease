import os
import glob
import shutil
from PIL import Image
import tqdm

dataset_labels = ['Algal_Leaf_Spot', 'Leaf_Blight', 'Leaf_Spot', 'No_Disease']
dataset_images_dir = "E:\\durian_leaf_disease\\train\\images"
dataset_labels_dir = "E:\\durian_leaf_disease\\train\\labels"
lora_dataset_dir = "E:\\durian_leaf_disease\\lora_train_dataset"

image_list = glob.glob(fr"{dataset_images_dir}\\*.jpg")
for image_index, image_file in tqdm.tqdm(enumerate(image_list), desc="分类裁切分割数据集", total=len(image_list), ncols=128):
    basename, suffix = os.path.splitext(os.path.basename(image_file))
    label_file = os.path.join(dataset_labels_dir, fr"{basename}.txt")
    # 读取图片
    image_src = Image.open(image_file).convert("RGB")
    image_w, image_h = image_src.size
    # 从图片中抠图
    with open(label_file) as label_fd:
        label_lines = label_fd.readlines()
    # 分析lable截图
    for label_index, label_line in enumerate(label_lines):
        label_pattern = label_line.split(" ")
        label_type_id = int(label_pattern[0])
        label_type = dataset_labels[label_type_id]
        label_type_dir = os.path.join(lora_dataset_dir, label_type)
        os.makedirs(label_type_dir, exist_ok=True)
        # cxcywh -> xyxy
        center_x = image_w * float(label_pattern[1])
        center_y = image_h * float(label_pattern[2])
        bbox_w = image_w*float(label_pattern[3])
        bbox_h = image_h*float(label_pattern[4])
        crop_bbox = [int(center_x-bbox_w/2), int(center_y-bbox_h/2), int(center_x+bbox_w/2), int(center_y+bbox_h/2)]
        # 从图片截取叶子
        image_crop = image_src.crop(crop_bbox)
        image_crop_file = os.path.join(label_type_dir, fr"{image_index}_{label_index}.jpg")
        # 存储叶子
        image_crop.save(image_crop_file)

for label_type in dataset_labels:
    image_list = glob.glob(fr"{lora_dataset_dir}\\{label_type}\\*.jpg")
    for image_index, image_file in tqdm.tqdm(enumerate(image_list), desc=f"重命名{label_type}", total=len(image_list), ncols=128):
        basename, suffix = os.path.splitext(os.path.basename(image_file))
        shutil.move(image_file, image_file.replace(basename, f"{image_index}"))

if __name__ == '__main__':
    pass