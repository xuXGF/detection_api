import sys, os
import cv2
sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0])

import numpy as np
import torch
from .utils import non_max_suppression, scale_coords, letterbox

class FireDetector:
    def __init__(self, model_path, inference="onnx", devices="cpu", conf_thres=0.4, iou_thres=0.5, image_size=640, anchor = [[10,13, 16,30, 33,23],[30,61, 62,45, 59,119],[116,90, 156,198, 373,326]]):

        self.model_path = model_path
        self.inference = inference
        self.devices = devices
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.img_size = image_size
        self.anchor = anchor
        self.class_name = {0:"fire", 1:"fog"}

        if self.inference == 'onnx':
            import onnxruntime as ort
            self.sess = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider'])
        else:
            import torch

            from .models.experimental import attempt_load

            self.model = attempt_load(self.model_path, map_location= self.devices)  # load FP32 model
            if self.devices != 'cpu':
                self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.devices).type_as(next(self.model.parameters())))  # run once

    def sigmoid(self, x):
        return 1.0 / (np.exp(-x) + 1)

    def make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def preprocess(self, image_ori, imgsz):
        # ---preprocess image for detection
        image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        image = letterbox(image, imgsz, stride=32)[0]
        image = image.astype(np.float32)
        image = image / 255.0  # 0 - 255 to 0.0 - 1.0
        image = np.transpose(image, [2, 0, 1])  # HWC to CHW, BGR to RGB
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image

    def poseprocess(self, outputs, imgsz):

        stride = [imgsz / output.shape[-2] for output in outputs]  # forward

        anchor_new = np.array(self.anchor).reshape(len(self.anchor), 1, -1, 1, 1, 2)

        z = []
        for i, output in enumerate(outputs):
            output = self.sigmoid(output)
            _, _, width, height, _ = output.shape
            grid = np.array(self.make_grid(width, height))
            output[..., 0:2] = (output[..., 0:2] * 2. - 0.5 + grid) * stride[i]  # x,y
            output[..., 2:4] = (output[..., 2:4] * 2) ** 2 * anchor_new[i]  # w, h
            z.append(output.reshape(1, -1, 7))
        pred = np.concatenate((z[0], z[1], z[2]), axis=1)
        # nms
        return pred

    def __call__(self, image_ori):
        # image_ori = data["data"]
        image = self.preprocess(image_ori, self.img_size)
        # print ("letterbox", image.shape)
        if self.inference == "onnx":
            outputs = []
            for i in range(len(self.anchor)):
                output = self.sess.run([self.sess.get_outputs()[i].name], input_feed={'images': image})[0]
                outputs.append(output)
            pred = self.poseprocess(outputs, self.img_size)
        else:
            pred = self.model(image)[0]


        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        pred_reformat = []
        instance_id = 0
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], image_ori.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    # Apply Classifier
                    xyxy = np.reshape(xyxy, (1, 4))
                    xyxy_ = np.copy(xyxy).tolist()[0]
                    xyxy_ = [int(i) for i in xyxy_]
                    pred_reformat.append(
                        {
                            "name": self.class_name[cls.item()],
                            "score": conf.item(),
                            "bbox":{"x_min": xyxy_[0],
                                    "y_min": xyxy_[1],
                                    "x_max": xyxy_[2],
                                    "y_max": xyxy_[3]}

                        })

        return pred_reformat