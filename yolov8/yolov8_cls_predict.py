import os.path as osp
from ultralytics import YOLO

def makd_abs_path(fn):
    return osp.abspath(osp.join(osp.dirname(__file__), fn))

if __name__ == '__main__':
    # run_dir = r"runs\classify\train_baseline_aug_hsv0"
    run_dir = r"runs\classify\train_1x_aug"
    # run_dir = r"runs\classify\train_1x"
    # run_dir = r"runs\classify\train_2x"
    # run_dir = r"runs\classify\train_seg_baseline"
    # 加载模型参数
    # ['Algal Leaf Spot', 'Leaf Blight', 'Leaf Spot', 'No Disease']
    model = YOLO(model=makd_abs_path(fr"{run_dir}\weights\best.pt"))
    # 训练数据
    # base_dir = "D:\\workspace"
    # base_dir = "D:\\workspace_SDXL"
    # base_dir = "D:\\workspace_SD35"
    # image_dir = f"{base_dir}\\Algal_Leaf_Spot_generated"
    # image_dir = f"{base_dir}\\Leaf_Spot_generated"
    # image_dir = f"{base_dir}\\Leaf_Blight_generated"
    # image_dir = f"{base_dir}\\No_Disease_generated"
    # image_dir = f"{base_dir}\\wendytest"
    # 全量数据库
    # image_dir = "E:\\Durian Leaf disease.v10i.yolov8\\train\\images"
    # 迁移数据
    # base_dir = "C:\\Users\\Wendy\\Downloads\\Durian_Leaf_disease.v10i.yolov8\\Durian_Leaf_dataset"
    # image_dir = f"{base_dir}\\Algal_Leaf_Spot"
    # image_dir = f"{base_dir}\\Leaf_Blight"
    # image_dir = f"{base_dir}\\Leaf_Spot"
    # image_dir = f"{base_dir}\\No_Disease"
    # image_dir = "E:\datasets\\Durian_Leaf_dataset3\\Leaf_Blight"
    # 验证集数据
    image_dir = "E:\\验证数据集"
    kwargs = {}
    kwargs['save'] = True
    kwargs['save_json'] = True
    kwargs['save_frames'] = True
    kwargs['save_txt'] = True
    kwargs['save_conf'] = True
    kwargs['show_labels'] = True
    kwargs['show_conf'] = True
    model.predict(source=image_dir, **kwargs)