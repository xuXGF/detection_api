from enum import Enum

__all__ = ['ServiceErrorCodeEnum', 'AlgoErrorCodeEnum',
           'ServiceError', 'AlgoError']


class ServiceErrorCodeEnum(Enum):
    """服务错误码枚举类"""
    SUCCESS = (0, 'Service SUCCESS')
    ERROR = (-1, 'Service ERROR')

    @property
    def code(self):
        """获取错误码"""
        return self.value[0]

    @property
    def message(self):
        """获取错误码信息"""
        return self.value[1]


class AlgoErrorCodeEnum(Enum):
    """算法错误码枚举类"""
    SUCCESS = (0, 'SUCCESS')
    ERROR_SERVICE_AVAILABLE = (-1, 'service temporarily unavailable')
    ERROR_REQUEST_NOT_JSON = (-1000, 'request body should be json format')
    ERROR_REQUEST_JSON_PARSE = (-1001, 'request json parse error')
    ERROR_MISSING_ARGS = (-1002, 'missing required arguments')
    ERROR_IVALID_ARG_VAL = (-1003, 'invalid argument value')
    ERROR_ARGUMENT_FORMAT = (-1004, 'argument format error')
    ERROR_INPUT_IMAGE_EMPTY = (-1100, 'input image is empty')
    ERROR_INPUT_IMAGE_BASE64 = (-1101, 'input image base64 error')
    ERROR_INPUT_IMAGE_READ = (-1102, 'input image read error')
    ERROR_INPUT_IMAGE_CHECKSUM = (-1103, 'input image checksum error')
    ERROR_INPUT_IMAGE = (-1104, 'input image error')
    ERROR_INPUT_IMAGE_HEADER = (-1105, 'input image header error')
    ERROR_INPUT_IMAGE_SIZE = (-1106, 'input image size is too large')
    ERROR_INPUT_IMAGE_CN = (-1107, 'input image channel number error, only support 1,3,4')
    ERROR_INPUT_IMAGE_FORMAT =(-1108, 'input image format error, only support "jpg,jpeg,png" format')
    ERROR_PREDICT = (-1200, 'predict error')
    ERROR_BATCH_PREDICT = (-1201, 'batch predict error')
    ERROR_UNKNOWN = (9999, 'unknown error')

    @property
    def code(self):
        """获取错误码"""
        return self.value[0]

    @property
    def message(self):
        """获取错误码信息"""
        return self.value[1]


class ServiceError(Exception):
    """服务错误异常处理类"""
    def __init__(self, error_code: ServiceErrorCodeEnum, extra_str: str=None):
        self.name = error_code.name
        self.code = error_code.code
        if extra_str is None:
            self.message = error_code.message
        else:
            self.message = f'{error_code.message}: {extra_str}'
        Exception.__init__(self)

    def __repr__(self):
        return f'[{self.__class__.__name__} {self.code}] {self.message}'

    __str__ = __repr__


class AlgoError(Exception):
    """算法错误异常处理类"""
    def __init__(self, error_code: AlgoErrorCodeEnum, extra_str: str=None):
        self.name = error_code.name
        self.code = error_code.code
        if extra_str is None:
            self.message = error_code.message
        else:
            self.message = f'{error_code.message}: {extra_str}'
        Exception.__init__(self)

    def __repr__(self):
        return f'[{self.__class__.__name__} {self.code}] {self.message}'

    __str__ = __repr__
    