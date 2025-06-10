import os.path as osp
from yolov8_tools import model_by, predict_by

def makd_abs_path(fn):
    return osp.abspath(osp.join(osp.dirname(__file__), fn))

if __name__ == '__main__':
    from yolov8_train import model_args
    _model_args = model_args()
    _model_ = model_by(model_path=_model_args["model_path"], pretrain_path=makd_abs_path(fr"runs\\detect\\train\\weights\\best.pt"), verbose=True)
    # ['Algal Leaf Spot', 'Leaf Blight', 'Leaf Spot', 'No Disease']
    # result = predict_by(model=_model_, source_dir="D:\\workspace\\Algal_Leaf_Spot_generated")
    # result = predict_by(model=_model_, source_dir="D:\\workspace\\Leaf_Blight_generated")
    # result = predict_by(model=_model_, source_dir="D:\\workspace\\Leaf_Spot_generated")
    # result = predict_by(model=_model_, source_dir="E:\\durian leaf disease dataset2\\train\ALGAL_LEAF_SPOT")
    # sdxl
    # result = predict_by(model=_model_, source_dir="D:\\workspace_SDXL\\Algal_Leaf_Spot_generated")
    # result = predict_by(model=_model_, source_dir="D:\\workspace_SDXL\\Leaf_Spot_generated")
    # result = predict_by(model=_model_, source_dir="D:\\workspace_SDXL\\Leaf_Blight_generated")
    # result = predict_by(model=_model_, source_dir="D:\\workspace_SDXL\\No_Disease_generated")
    # 测试数据
    # result = predict_by(model=_model_, source_dir="E:\\durian_leaf_disease\\aug_result")
    # sd35
    # result = predict_by(model=_model_, source_dir="D:\\workspace_SD35\\Algal_Leaf_Spot_generated")
    # result = predict_by(model=_model_, source_dir="D:\\workspace_SD35\\Leaf_Spot_generated")
    # result = predict_by(model=_model_, source_dir="D:\\workspace_SD35\\Leaf_Blight_generated")
    # result = predict_by(model=_model_, source_dir="D:\\workspace_SD35\\No_Disease_generated")
    # 验证集数据
    result = predict_by(model=_model_, source_dir="E:\\验证数据集")
    