import os
import os.path as osp
import glob
import shutil
import random
import tqdm
import xml.etree.ElementTree as ET

def make_abs_path(fn):
    return osp.abspath(osp.join(osp.dirname(__file__), fn))

def cmp_dir(ann_dir, img_dir, debug=True):
    miss_img = []
    ann_list = glob.glob(osp.join(ann_dir, "*.xml"))
    assert len(ann_list) > 0
    for ann_path in tqdm.tqdm(ann_list, total=len(ann_list), desc="查找缺失图片标记", disable=not debug):
        name, _ = osp.splitext(osp.basename(ann_path))
        if not osp.exists(osp.join(img_dir, f"{name}.jpg")):
            miss_img.append(ann_path)
    #
    miss_ann = []
    img_list = glob.glob(osp.join(img_dir, "*.jpg"))
    assert len(img_list) > 0
    for img_path in tqdm.tqdm(img_list, total=len(img_list), desc="查找缺失标记图片", disable=not debug):
        name, _ = osp.splitext(osp.basename(img_path))
        if not osp.exists(osp.join(ann_dir, f"{name}.xml")):
            miss_ann.append(img_path)
    #
    return miss_img, miss_ann

def remove_missing(voc_dir):
    ann_dir = osp.join(voc_dir, "Annotations")
    img_dir = osp.join(voc_dir, "JPEGImages")

    miss_img, miss_ann = cmp_dir(ann_dir, img_dir)
    for ann_path in miss_img: os.remove(ann_path)
    for img_path in miss_ann: os.remove(img_path)
    miss_img, miss_ann = cmp_dir(ann_dir, img_dir, debug=False)

    assert len(miss_img) == 0 and len(miss_ann) == 0

def create_sets(voc_dir, yolo_dir, train_ratio=0.8, cls_list=["unhealthy"]):
    # voc
    ann_dir = osp.join(voc_dir, "Annotations")
    img_dir = osp.join(voc_dir, "JPEGImages")
    sets_dir = osp.join(voc_dir, "ImageSets", "Main")

    img_list = glob.glob(osp.join(img_dir, "*.jpg"))
    random.shuffle(img_list)
    img_list_len = len(img_list)
    train_list = img_list[:int(img_list_len*train_ratio)]
    val_list = img_list[len(train_list):]
    if not osp.exists(sets_dir): os.makedirs(sets_dir, exist_ok=True)
    train_path = osp.join(sets_dir, 'train.txt')
    val_path = osp.join(sets_dir, "val.txt")
    with open(train_path, 'w') as f:
        for file in train_list: f.write(f"{osp.splitext(osp.basename(file))[0]}\n")
    with open(val_path, 'w') as f:
        for file in val_list: f.write(f"{osp.splitext(osp.basename(file))[0]}\n")
    # yolo
    yolo_images_dir = osp.join(yolo_dir, "images")
    shutil.rmtree(yolo_images_dir, ignore_errors=True)
    if not osp.exists(yolo_images_dir): os.makedirs(yolo_images_dir, exist_ok=True)
    yolo_labels_dir = osp.join(yolo_dir, "labels")
    shutil.rmtree(yolo_labels_dir, ignore_errors=True)
    if not osp.exists(yolo_labels_dir): os.makedirs(yolo_labels_dir, exist_ok=True)
    # 训练集
    yolo_train_img_dir = osp.join(yolo_images_dir, "train")
    if not osp.exists(yolo_train_img_dir): os.makedirs(yolo_train_img_dir, exist_ok=True)
    yolo_train_lab_dir = osp.join(yolo_labels_dir, "train")
    if not osp.exists(yolo_train_lab_dir): os.makedirs(yolo_train_lab_dir, exist_ok=True)
    import cv2
    for img_path in tqdm.tqdm(train_list, total=len(train_list), desc="转换voc2yolo训练集"):
        name, suffix = osp.splitext(osp.basename(img_path))
        # shutil.copy(img_path, osp.join(yolo_train_img_dir, f"{name}{suffix}"))
        new_img = cv2.imread(img_path)
        cv2.imwrite(osp.join(yolo_train_img_dir, f"{name}{suffix}"), new_img)
        cvt_voc_yolo(osp.join(ann_dir, f"{name}.xml"), osp.join(yolo_train_lab_dir, f"{name}.txt"), cls_list)
    # 验证集
    yolo_val_img_dir = osp.join(yolo_images_dir, "val")
    if not osp.exists(yolo_val_img_dir): os.makedirs(yolo_val_img_dir, exist_ok=True)
    yolo_val_lab_dir = osp.join(yolo_labels_dir, "val")
    if not osp.exists(yolo_val_lab_dir): os.makedirs(yolo_val_lab_dir, exist_ok=True)
    for img_path in tqdm.tqdm(val_list, total=len(val_list), desc="转换voc2yolo验证集"):
        name, suffix = osp.splitext(osp.basename(img_path))
        # shutil.copy(img_path, osp.join(yolo_val_img_dir, f"{name}{suffix}"))
        new_img = cv2.imread(img_path)
        cv2.imwrite(osp.join(yolo_val_img_dir, f"{name}{suffix}"), new_img)
        cvt_voc_yolo(osp.join(ann_dir, f"{name}.xml"), osp.join(yolo_val_lab_dir, f"{name}.txt"), cls_list)

def cvt_voc_yolo(xml_path, txt_path, cls_list):
    def cvt_cxcywh(size, box):
        dw, dh = 1./size[0], 1./size[1]
        cx, cy, w, h = (box[0]+box[1])/2.0 -1, (box[2]+box[3])/2.0 -1, box[1]-box[0], box[3]-box[2]
        return cx*dw, cy*dh, w*dw, h*dh

    xml = ET.parse(xml_path)
    root = xml.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    bbox_lines = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        if cls in cls_list:
            xmlbox = obj.find("bndbox")
            bbox = cvt_cxcywh((w, h), [float(xmlbox.find(x).text) for x in ["xmin", "xmax", "ymin", "ymax"]])
            cls_id = cls_list.index(cls)
            bbox_lines.append(" ".join(str(x) for x in (cls_id, *bbox)))
    
    with open(txt_path, 'w') as f:
        f.write("\n".join(bbox_lines))

if __name__ == '__main__':
    voc_dir = make_abs_path(r"VOCdevkit\VOC2007")
    remove_missing(voc_dir)
    #
    yolo_dir = make_abs_path(r"datasets")
    create_sets(voc_dir, yolo_dir, train_ratio=0.9)
