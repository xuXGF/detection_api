import dataclasses
from flask import Flask, request
from flask import jsonify
from functions import FireDetector
from functions import cfg

from werkzeug.exceptions import HTTPException

from webapi import *


app = Flask(__name__)

# 实例化模型对象
detector = FireDetector(**cfg.det_cfg)


@app.route('/fire_detector', methods=["POST"])
def fire_detector():
    # get data
    try:
        # http的解析 
        if request.headers.get('Content-Type') is None:
            raise AlgoError(AlgoErrorCodeEnum.ERROR_REQUEST_NOT_JSON, 'content-type error')
        if request.headers.get('Content-Type').lower() != "application/json" and request.headers.get('Content-Type').lower() != "application/x-www-form-urlencoded":
            raise AlgoError(AlgoErrorCodeEnum.ERROR_REQUEST_NOT_JSON, 'content-type error')
        # 获取输入数据 
        post_data = request.get_json(silent=True)
        # 解析输入数据
        request_arg = RequestArgument.from_post_data(post_data) # paraser input data to format 
        algo_arg = AlgorithmArgument.from_request_arg(request_arg)
        # 前向推理，输出结果
        algo_result = detector(algo_arg.data)

    # 返回结果 
    except (ServiceError, AlgoError) as e:
        response_arg = ResponseArgument(
            req_id = "",
            err_no=e.code,
            err_msg=e.message,
            result=None)
        return jsonify(dataclasses.asdict(response_arg))
    except HTTPException as e:
        response_arg = ResponseArgument(
            req_id = "",
            err_no=e.code,
            err_msg=f'{e.name}: {e.description}',
            result=None)
        return jsonify(dataclasses.asdict(response_arg))
    except Exception as e:
        response_arg = ResponseArgument(
            req_id = "",
            err_no=AlgoErrorCodeEnum.ERROR_UNKNOWN.code,
            err_msg=AlgoErrorCodeEnum.ERROR_UNKNOWN.message,            
            result=None)
        return jsonify(dataclasses.asdict(response_arg))
    response_arg = ResponseArgument(
        err_no=ServiceErrorCodeEnum.SUCCESS.code,
        err_msg=ServiceErrorCodeEnum.SUCCESS.message,
        req_id = algo_arg.request_id,
        result=algo_result)
    return jsonify(dataclasses.asdict(response_arg))

if __name__ == '__main__':
    app.run(host='0.0.0.0',
            port=2222,
            debug=True)