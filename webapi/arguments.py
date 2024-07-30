import base64
import json
from collections import OrderedDict
from typing import Union, Optional, List

import cv2
import dataclasses
import numpy as np
from dataclasses import dataclass
from pydantic import BaseModel

from .error_code import *

__all__ = [ 'RequestArgument', 'AlgorithmArgument',  'ResponseArgument']


def normalize_image_shape(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        num_channels = image.shape[-1]
        if num_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif num_channels == 3:
            pass
        elif num_channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError('Unsupported!')
    else:
        raise ValueError('Unsupported!')
    return image


class RequestArgument(BaseModel):
    """接口文档的直接翻译
    """
    request_id: Optional[str] = None  # request id
    data: Union[str, bytes]= None  # 图像文件数据, base64 编码
    data_path: Optional[str] = None  # 图片路径
    data_fmt: Optional[str] = None # 数据格式
    params: Optional[str] = None # 输入参数


    @staticmethod
    def from_post_data(post_data):
        try:
            request_id = post_data.get('request_id')
            data = post_data.get('data')
            data_path = post_data.get('data_path')
            data_fmt = post_data.get('data_fmt')
            params = post_data.get('params')

            # post_data ='imageBase64='+ image_data +"&" +"deviceId="+device_id+"&"+"taskId="+task_id # TODO
        except:
            raise AlgoError(AlgoErrorCodeEnum.ERROR_REQUEST_NOT_JSON)
        if post_data is None:
            raise AlgoError(AlgoErrorCodeEnum.ERROR_REQUEST_JSON_PARSE)

        if request_id is None:
            raise AlgoError(AlgoErrorCodeEnum.ERROR_MISSING_ARGS)

        if (data is None and data_path is None) or (data is not None and data_path is not None):
            raise AlgoError(AlgoErrorCodeEnum.ERROR_IVALID_ARG_VAL)

        try:
            # post_data = base64.b64decode(post_data.encode()).decode()
            post_data_dict = dict(
                request_id = request_id,
                data = data,
                data_path= data_path,
                data_fmt= data_fmt,
                params = params
            )
            return RequestArgument(**post_data_dict)

        except Exception as e:
            raise AlgoError(AlgoErrorCodeEnum.ERROR_REQUEST_JSON_PARSE)

@dataclass
class AlgorithmArgument:
    """将 RequestArgument 转换为算法的输入参数
    """

    request_id: Optional[str] = None  # request id
    data: Union[str, bytes] = None # 图像文件数据, base64 编码
    data_path: Optional[str] = None  # 图片路径
    data_fmt: Optional[str] = None # 数据格式
    params: Optional[str] = None # 输入参数



    # class attribute 需要确认
    IMAGE_MAX_WIDTH = 1920
    IMAGE_MAX_HEIGHT = 1920
    IMAGE_FORMAT=["JPG","jpg","png","jpeg"]

    @classmethod
    def convert_to_image(cls, data: Union[str, bytes, None],is_data=True) -> Optional[np.ndarray]:
        if is_data:
            try:
                if isinstance(data, str):
                    data = data.encode()
                filename = base64.b64decode(data)
            except Exception:
                raise AlgoError(AlgoErrorCodeEnum.ERROR_INPUT_IMAGE_BASE64)
            try:
                image = cv2.imdecode(np.frombuffer(filename, dtype=np.uint8), -1)
            except Exception as e:
                print(e)
                raise AlgoError(AlgoErrorCodeEnum.ERROR_INPUT_IMAGE_READ)
        else:
            try:
                image = cv2.imdecode(np.fromfile(data, dtype=np.uint8), -1) #bgr
            except Exception as e:
                print(e)
                raise AlgoError(AlgoErrorCodeEnum.ERROR_INPUT_IMAGE_READ)
        if image.shape[0] > cls.IMAGE_MAX_HEIGHT or image.shape[1] > cls.IMAGE_MAX_WIDTH:
            raise AlgoError(AlgoErrorCodeEnum.ERROR_INPUT_IMAGE_SIZE)
        try:
            image = normalize_image_shape(image)
        except:
            raise AlgoError(AlgoErrorCodeEnum.ERROR_INPUT_IMAGE_CN)
        return image

    @classmethod
    def from_request_arg(cls, request_arg: RequestArgument):
        algo_dict = {}
        #with mdsjg.utils.ContextTimer('convert_base64_to_image'):
        algo_dict['request_id'] = request_arg.request_id
        if request_arg.data is not None:
            algo_dict['data'] = AlgorithmArgument.convert_to_image(request_arg.data, is_data=True)
        if request_arg.data_path is not None:
            algo_dict['data'] = AlgorithmArgument.convert_to_image(request_arg.data_path, is_data=False)
        if request_arg.data_fmt is not None:
            if request_arg.data_fmt  not in cls.IMAGE_FORMAT:
                raise AlgoError(AlgoErrorCodeEnum.ERROR_INPUT_IMAGE_FORMAT)
            else:
                algo_dict['data_fmt'] = request_arg.data_fmt
        if request_arg.params is not None:
            algo_dict['params'] = request_arg.params

        return AlgorithmArgument(**algo_dict)


    def to_request_arg(self):
        raise NotImplementedError


@dataclass
class ResponseArgument:
    """服务的响应接口"""
    err_no: int  # 错误码
    err_msg: str  # 错误信息
    request_id: str # 请求id
    version: str # 接口版本
    result: Optional[str] = None  # 算法结