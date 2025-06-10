import os
import os.path as osp
import shutil
from yolov8_tools import model_by, train_by
import torch
torch.backends.cudnn.enabled=False
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
import os
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
from utils import logger, logger_file

def makd_abs_path(fn):
    return osp.abspath(osp.join(osp.dirname(__file__), fn))

TRAIN_ARGS = "m.dld"

def model_args(mt:str=TRAIN_ARGS):
    model_cfg = {
        "l.CBAM": {
            "model_path": makd_abs_path(fr"models/yolov8l.CBAM.yaml"),
            "pretrain_path": makd_abs_path(fr"pretrains/yolov8l.pt"),
            "data_path": makd_abs_path(fr"datasets/coco_durian_leaf.yaml"),
            "kwargs": {
                "batch":16,
                "device": "cuda", 
                "label_smoothing": 0.05,
            }
        },
        "m": {
            "model_path": makd_abs_path(fr"models/yolov8m.yaml"),
            "pretrain_path": makd_abs_path(fr"pretrains/yolov8m.pt"),
            "data_path": makd_abs_path(fr"datasets/coco_durian_leaf.yaml"),
            "kwargs": {
                "batch":32,
                "device": "cuda", 
                "label_smoothing": 0.05,
            }
        },
        "m.CBAM": {
            "model_path": makd_abs_path(fr"models/yolov8m.CBAM.yaml"),
            "pretrain_path": makd_abs_path(fr"pretrains/yolov8m.pt"),
            "data_path": makd_abs_path(fr"datasets/coco_durian_leaf.yaml"),
            "kwargs": {
                "batch":32,
                "device": "cuda", 
                "label_smoothing": 0.05,
            }
        },
        "n": {
            "model_path": makd_abs_path(fr"models/yolov8n.yaml"),
            "pretrain_path": makd_abs_path(fr"pretrains/yolov8n.pt"),
            "data_path": makd_abs_path(fr"datasets/coco_durian_leaf.yaml"),
            "kwargs": {
                "batch":64,
                "device": "cuda", 
                "label_smoothing": 0.05,
            }
        },
        "n.CBAM": {
            "model_path": makd_abs_path(fr"models/yolov8n.CBAM.yaml"),
            "pretrain_path": makd_abs_path(fr"pretrains/yolov8n.pt"),
            "data_path": makd_abs_path(fr"datasets/coco_durian_leaf.yaml"),
            "kwargs": {
                "batch": 64,
                "device": "cuda", 
                "label_smoothing": 0.05,
            }
        },
        "s.CBAM": {
            "model_path": makd_abs_path(fr"models/yolov8s.CBAM.yaml"),
            "pretrain_path": makd_abs_path(fr"pretrains/yolov8s.pt"),
            "data_path": makd_abs_path(fr"datasets/coco_durian_leaf.yaml"),
            "kwargs": {
                "batch": 64,
                "device": "cuda", 
                "label_smoothing": 0.05,
            }
        },
        "s.NECK": {
            "model_path": makd_abs_path(fr"models/yolov8s.NECK.yaml"),
            "pretrain_path": makd_abs_path(fr"pretrains/yolov8s.pt"),
            "data_path": makd_abs_path(fr"datasets/coco_durian_leaf.yaml"),
            "kwargs": {
                "batch": 64,
                "device": "cuda", 
                "label_smoothing": 0.05,
            }
        },
        "l.NECK": {
            "model_path": makd_abs_path(fr"models/yolov8l.NECK.yaml"),
            "pretrain_path": makd_abs_path(fr"pretrains/yolov8l.pt"),
            "data_path": makd_abs_path(fr"datasets/coco_durian_leaf.yaml"),
            "kwargs": {
                "batch": 16,
                "device": "cuda", 
                "label_smoothing": 0.05,
            }
        },
        "m.dld": {
            "model_path": makd_abs_path(fr"models/yolov8m.nc4.yaml"),
            "pretrain_path": makd_abs_path(fr"pretrains/yolov8m.pt"),
            "data_path": "D:\\workspace\\detect_dataset\\data.yaml",
            "kwargs": {
                "batch":32,
                "device": "cuda", 
                "label_smoothing": 0.05,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "fliplr": 0.0,
                "mosaic": 0.0,
                "erasing": 0.0,
                "auto_augment": None,
            }
        },
        "l.dld": {
            "model_path": makd_abs_path(fr"models/yolov8l.nc4.yaml"),
            "pretrain_path": makd_abs_path(fr"pretrains/yolov8l.pt"),
            "data_path": "D:\\workspace\\detect_dataset\\data.yaml",
            "kwargs": {
                "batch":16,
                "device": "cuda", 
                "label_smoothing": 0.05,
            }
        },
    }
    return model_cfg[mt]

if __name__ == '__main__':
    """
    指标评估：https://cloud.tencent.com/developer/article/1624811
    """
    run_dir = r"runs\detect\train"
    # 请注意要不要删除log
    if osp.exists(run_dir): shutil.rmtree(run_dir)  # 删除log
    if not osp.exists(run_dir): os.makedirs(run_dir)
    logger_file(osp.join(run_dir, "yolov8_train_log.log"))
    # 获取参数开始训练
    _model_args = model_args(TRAIN_ARGS)
    logger(fr"model_args={_model_args}")
    _model_ = model_by(model_path=_model_args["model_path"], pretrain_path=_model_args["pretrain_path"], verbose=True)
    result = train_by(_model_, data_path=_model_args["data_path"], epochs=100, patience=30, **_model_args["kwargs"])
    