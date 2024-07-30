import time
import dataclasses
from fastapi import FastAPI
from fastapi import Request
from asgiref.sync import sync_to_async
from functions import FireDetector
from functions import cfg
from functions.utils import *

from werkzeug.exceptions import HTTPException

from webapi import *

app = FastAPI()

# 实例化模型对象
detector = FireDetector(**cfg.det_cfg)

## 配置日志记录器
logger_message(cfg.log_level[0], "services start")

@app.post('/fire_detector')
async def main(request: Request):
    # get data
    try:
        # http的解析
        if request.headers.get('Content-Type') is None:
            raise AlgoError(AlgoErrorCodeEnum.ERROR_REQUEST_NOT_JSON, 'content-type error')
        if request.headers.get('Content-Type').lower() != "application/json" and request.headers.get('Content-Type').lower() != "application/x-www-form-urlencoded":
            raise AlgoError(AlgoErrorCodeEnum.ERROR_REQUEST_NOT_JSON, 'content-type error')
            # 获取输入数据
        try:
            post_data = await request.json()
        except:
            raise AlgoError(AlgoErrorCodeEnum.ERROR_REQUEST_JSON_PARSE, 'request.json')
        # 解析输入数据
        request_arg = RequestArgument.from_post_data(post_data)
        algo_arg = AlgorithmArgument.from_request_arg(request_arg)
        logger_message(cfg.log_level[0], '{},{} start'.format(post_data.get('request_id'),time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        logger_message(cfg.log_level[0], '{},{} start'.format(post_data.get('require_id'),time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        logger_message(cfg.log_level[0], '{},{} end'.format(post_data.get('request_id'),time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        logger_message("WARNING", '{},{} end'.format(post_data.get('request_id'),time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        # logger_message("ERROR", '{},{} end'.format(post_data.get('request_id'),time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        # 前向推理，输出结果
        algo_result = await sync_to_async(detector)(algo_arg.data)

    # 返回结果
    except (ServiceError, AlgoError) as e:
        response_arg = ResponseArgument(
            req_id = "",
            err_no=e.code,
            err_msg=e.message,
            result=None)
        return dataclasses.asdict(response_arg)
    except HTTPException as e:
        response_arg = ResponseArgument(
            req_id = "",
            err_no=e.code,
            err_msg=f'{e.name}: {e.description}',
            result=None)
        return dataclasses.asdict(response_arg)
    except Exception as e:
        response_arg = ResponseArgument(
            req_id = "",
            err_no=AlgoErrorCodeEnum.ERROR_UNKNOWN.code,
            err_msg=AlgoErrorCodeEnum.ERROR_UNKNOWN.message,
            result=None)
        return dataclasses.asdict(response_arg)
    response_arg = ResponseArgument(
        err_no=ServiceErrorCodeEnum.SUCCESS.code,
        err_msg=ServiceErrorCodeEnum.SUCCESS.message,
        req_id = algo_arg.request_id,
        result=algo_result)
    response_arg = dataclasses.asdict(response_arg)
    return response_arg



if __name__ == '__main__':
    # app.run(host='0.0.0.0',
    #   port=8081,
    #   debug=True)
    app.run(host='0.0.0.0',
            port=2222,
            debug=True)