import os

# variable
repo_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(repo_dir, 'models_encrypted')
if not os.path.exists(model_path):
    model_path = os.path.join(repo_dir, 'models')

# log 显示层级 # logging.DEBUG , logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
log_level = "DEBUG",

interface_version = "v1.1",

det_cfg = dict(model_path=os.path.join(model_path, 'FireDetection_v1.0.0.onnx'), #onnx or torchscript
        conf_thres=0.5,
        iou_thres = 0.5,
        devices="0",
        inference='onnx',
        image_size= 640)