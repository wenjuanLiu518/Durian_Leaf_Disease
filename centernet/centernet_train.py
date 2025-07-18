import os
import os.path as osp
import shutil
from centernet_tools import model_by, train_by
from voc_dataset import VOC_CLASSES, SZIE
from utils import logger, logger_file
import torch
torch.backends.cudnn.enabled=False
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

def makd_abs_path(fn=""):
    return osp.abspath(osp.join(osp.dirname(__file__), fn))

if __name__ == '__main__':
    run_dir = r"centernet\train"
    # 请注意要不要删除log
    if osp.exists(run_dir): shutil.rmtree(run_dir)  # 删除log
    if not osp.exists(run_dir): os.makedirs(run_dir)
    logger_file(osp.join(run_dir, "centernet_train_log.log"))
    # 配置last_wieght_path就会继续训练
    last_weight_path = osp.join(run_dir, "weights", "last.pth")
    _model_ = model_by(num_classes=1, model_path=last_weight_path)
    logger(_model_)
    train_by(model=_model_, data_path=makd_abs_path(), num_epochs=300, last_epoch=-1)
