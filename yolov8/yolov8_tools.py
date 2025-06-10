import os.path as osp
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules import NeckBridge
from ultralytics.nn.tasks import DetectionModel
from attention import CBAM
from utils import logger

def model_by(model_path:str, task="detect", verbose=False, pretrain_path=None):
    """
    :param model_path: *.pt权重文件, *.yaml是模型结构, 参考ultralytics/cfg/models/v8/yolov8.yaml
    """
    _model = YOLO(model=model_path, task=task, verbose=verbose)
    _model.model = config_model(_model.model)
    # pretrain是backbone
    if pretrain_path is not None and osp.exists(pretrain_path):
        _model.load(weights=pretrain_path)
    return _model

def config_model(_model:DetectionModel) -> DetectionModel:
    # 开始更换neck部分
    # for neck_idx in (16, 20, 24): # 参考yaml配置层级
    #     old_neckbridge = _model.model[neck_idx]
    #     if not isinstance(old_neckbridge, NeckBridge): continue
    #     # 方式一：
    #     # _model.model[neck_idx].att_seq.append(
    #     #     CBAM(old_neckbridge.c1, old_neckbridge.kernel_size)
    #     # )
    #     # 方式二：
    #     _model.model[neck_idx].att_seq = nn.Sequential(
    #         CBAM(old_neckbridge.c1, old_neckbridge.kernel_size)
    #     )
    # 写入文件检查model
    # logger(f"{_model}")
    # 更好neck部分完成
    return _model

class SelfDetectionTrainer(DetectionTrainer):
    """
    调整运行时的model结构，增加trainer自定义neck部分（增加注意力机制）
    """
    def get_model(self, cfg=None, weights=None, verbose=True):
        _model = super().get_model(cfg, weights, verbose)
        return config_model(_model)

def train_by(model:YOLO, data_path:str, epochs=100, **kwargs):
    """
    https://docs.ultralytics.com/zh/modes/train/
    """
    if kwargs is None: kwargs = {}
    kwargs['save'] = True
    kwargs['exist_ok'] = True
    # mosaic关闭的时机
    if 'close_mosaic' not in kwargs: 
        kwargs['close_mosaic'] = int(0.1 * epochs)
    # 训练的是head部分
    return model.train(trainer=SelfDetectionTrainer, data=data_path, epochs=epochs, **kwargs)

def predict_by(model:YOLO, source_dir, **kwargs):
    """
    https://docs.ultralytics.com/zh/modes/predict/
    """
    if kwargs is None: kwargs = {}
    kwargs['save'] = True
    kwargs['save_json'] = True
    kwargs['save_frames'] = True
    kwargs['save_txt'] = True
    kwargs['save_conf'] = True
    kwargs['save_crop'] = True
    kwargs['show_labels'] = True
    kwargs['show_conf'] = True
    return model.predict(source=source_dir, **kwargs)
