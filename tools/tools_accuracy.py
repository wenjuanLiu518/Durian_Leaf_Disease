import os
import shutil

def make_abs_path(path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

def calculate_classify_accuracy(src_dir, cls_dir, conf_th=0.30):
    # 记录每个分类的结果
    labels_result = {}
    # 原始图片目录
    required_labels = [
        os.path.basename(f).replace(".jpg", ".txt").replace(".png", ".txt") 
        for f in os.listdir(src_dir) if f.endswith((".png", ".jpg"))
    ]
    # labels结果目录
    labels_dir = os.path.join(cls_dir, "labels")
    # 比对是否符合预期结果
    for label_name in required_labels:
        label_key = str(label_name.split("_", maxsplit=1)[0])
        labels_record = labels_result.get(label_key, [0, 0])
        # 读取分类结果
        label_path = os.path.join(labels_dir, label_name)
        with open(label_path, "r") as f:
            label_line = f.readline().strip()
        # 过滤空行
        if not label_line:
            labels_record[1] += 1
            labels_result[label_key] = labels_record
            continue
        # 计算conf过滤
        split_patterns = label_line.split(" ", maxsplit=1)
        conf_calc = float(split_patterns[0])
        # 过滤conf小于阈值
        if conf_calc < conf_th:
            labels_record[1] += 1
            labels_result[label_key] = labels_record
            continue
        # 比对分类结果
        label_calc = str(split_patterns[1]).lower()
        if label_calc == label_key.lower(): 
            # 分类正确
            labels_record[0] += 1
        else: 
            # 分类错误
            labels_record[1] += 1
        # 记录结果
        labels_result[label_key] = labels_record
    # 返回labels计算结果
    return labels_result

def calculate_detect_accuracy(label_mapping, src_dir, det_dir, conf_th=0.30):
    # 记录每个分类的结果
    labels_result = {}
    # 原始图片目录
    required_labels = [
        os.path.basename(f).replace(".jpg", ".txt").replace(".png", ".txt") 
        for f in os.listdir(src_dir) if f.endswith((".png", ".jpg"))
    ]
    # labels结果目录
    labels_dir = os.path.join(det_dir, "labels")
    # 比对是否符合预期结果
    for label_name in required_labels:
        label_key = str(label_name.split("_", maxsplit=1)[0])
        labels_record = labels_result.get(label_key, [0, 0])
        # 读取分类结果
        label_path = os.path.join(labels_dir, label_name)
        if not os.path.exists(label_path):
            labels_record[1] += 1
            labels_result[label_key] = labels_record
            continue
        with open(label_path, "r") as f:
            label_line = f.readline().strip()
        # 过滤空行
        if not label_line:
            labels_record[1] += 1
            labels_result[label_key] = labels_record
            continue
        # 计算conf过滤
        split_patterns = label_line.split(" ", maxsplit=5)
        conf_calc = float(split_patterns[5])
        # 过滤conf小于阈值
        if conf_calc < conf_th:
            labels_record[1] += 1
            labels_result[label_key] = labels_record
            continue
        # 比对分类结果
        label_calc = label_mapping.get(int(split_patterns[0])).lower()
        if label_calc == label_key.lower(): 
            # 分类正确
            labels_record[0] += 1
        else: 
            # 分类错误
            labels_record[1] += 1
        # 记录结果
        labels_result[label_key] = labels_record
    # 返回labels计算结果
    return labels_result

if __name__ == "__main__":
    # label
    source_dir = fr"E:\验证数据集"
    label_mapping = {0: "Algal Leaf Spot", 1: "Leaf Blight", 2: "Leaf Spot", 3: "No Disease"}
    # 分类结果计算
    # classify_dir = fr"E:\\yolov8\\runs\\classify\\predict_baseline"
    # 分类结果: {'Algal Leaf Spot': [32, 25], 'Leaf Blight': [45, 12], 'Leaf Spot': [10, 54], 'No Disease': [7, 3]}
    # classify_dir = fr"E:\\yolov8\\runs\\classify\\predict_1x"
    # 分类结果: {'Algal Leaf Spot': [48, 9], 'Leaf Blight': [41, 16], 'Leaf Spot': [15, 49], 'No Disease': [7, 3]}
    # classify_dir = fr"E:\\yolov8\\runs\\classify\\predict_2x"
    # 分类结果: {'Algal Leaf Spot': [39, 18], 'Leaf Blight': [46, 11], 'Leaf Spot': [9, 55], 'No Disease': [7, 3]}
    # classify_dir = fr"E:\\yolov8\\runs\\classify\\predict_seg_baseline"
    # # 分类结果: {'Algal Leaf Spot': [10, 47], 'Leaf Blight': [30, 27], 'Leaf Spot': [1, 63], 'No Disease': [4, 6]}
    classify_dir = fr"E:\\yolov8\\runs\\classify\\predict23"
    labels_result = calculate_classify_accuracy(source_dir, classify_dir)
    print(fr"分类结果: {labels_result}")
    # 检测结果计算
    detect_dir = fr"E:\\yolov8\\runs\\detect\\predict_baseline"
    labels_result = calculate_detect_accuracy(label_mapping, source_dir, detect_dir)
    print(fr"检测结果: {labels_result}")
