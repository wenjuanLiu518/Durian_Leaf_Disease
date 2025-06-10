import os
import os.path as osp
import shutil
from ultralytics import YOLO
import torch
torch.backends.cudnn.enabled=False
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
import os
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
from utils import logger, logger_file

if __name__ == '__main__':
    run_dir = r"runs\classify\train"
    # 请注意要不要删除log
    if osp.exists(run_dir): shutil.rmtree(run_dir)  # 删除log
    # if not osp.exists(run_dir): os.makedirs(run_dir)
    # logger_file(osp.join(run_dir, "yolov8_train_log.log"))
    # 获取参数开始训练
    model = YOLO("yolov8m-cls.pt")
    results = model.train(
        data="D:\\workspace\\CAS\\dataset_flux_2x", epochs=100, imgsz=640, batch=32, patience=30,
        hsv_h=0.015,  # 轻微 HSV 增强
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.0,  # 根据需要启用
        erasing=0.4,
        auto_augment="randaugment",  # 使用 RandAugment 增强,
    )
    print(results)