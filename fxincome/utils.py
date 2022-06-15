import json, joblib
import tensorflow.keras as keras
import numpy as np
from dataclasses import dataclass, field


def get_curve(points, type):
    """
    使用收益率曲线中已知的一组点，通过指定的拟合算法生成任意点的收益率。
    该函数返回一个收益率曲线函数func(x)，使用者利用func(x)即可计算期限对应的收益率。
    对于一个point(x, y)，x坐标是年（单位为年），y坐标是收益率（单位为%）。
        Args:
            points(ndarray): 收益率曲线中已知的点，形状是2D Array [[x1,y1], [x2,y2] ... [xn,yn]]，其中x不能重复
            type(str): [LINEAR, POLYNOMIAL, HERMIT, SPLINE]
        Returns:
            func(float): 输入任意年限，输出对应的拟合收益率。输入单位是年，输出单位是%
    """

    size = points.shape[0]
    if type == 'LINEAR':
        points = points[points[:, 0].argsort()]  # 以x坐标（年限）作升序排序

        def func(x):
            for i in range(1, size):
                if x <= points[i, 0]:
                    break
            return (points[i, 1] - points[i - 1, 1]) / (points[i, 0] - points[i - 1, 0]) * (x - points[i - 1, 0]) + \
                   points[
                       i - 1, 1]
    elif type == 'POLYNOMIAL':
        matrix_x = np.zeros([size, size])
        matrix_y = np.array(points[:, 1])
        for i in range(size):
            for j in range(size):
                matrix_x[i, j] = points[i, 0] ** j
        para = np.dot(np.linalg.inv(matrix_x), matrix_y)

        def func(x):
            xx = np.array([x ** i for i in range(size)])
            return np.dot(para, xx)
    elif type == 'HERMIT':
        matrix_x = np.zeros([(size - 1) * 4, (size - 1) * 4])
        matrix_y = np.zeros([(size - 1) * 4])
        y_1 = [(points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0])] + \
              [(points[i + 1, 1] - points[i - 1, 1]) / (points[i + 1, 0] - points[i - 1, 0]) for i in
               range(1, size - 1)] + \
              [(points[size - 1, 1] - points[size - 2, 1]) / (points[size - 1, 0] - points[size - 2, 0])]
        for i in range(size - 1):
            for j in range(2):
                matrix_x[2 * i + j, 4 * i] = points[i + j, 0] ** 3
                matrix_x[2 * i + j, 4 * i + 1] = points[i + j, 0] ** 2
                matrix_x[2 * i + j, 4 * i + 2] = points[i + j, 0]
                matrix_x[2 * i + j, 4 * i + 3] = 1
                matrix_y[2 * i + j] = points[i + j, 1]

                matrix_x[2 * (size - 1) + 2 * i + j, 4 * i] = 3 * points[i + j, 0] ** 2
                matrix_x[2 * (size - 1) + 2 * i + j, 4 * i + 1] = 2 * points[i + j, 0]
                matrix_x[2 * (size - 1) + 2 * i + j, 4 * i + 2] = 1
                matrix_y[2 * (size - 1) + 2 * i + j] = y_1[i + j]
        para = np.dot(np.linalg.inv(matrix_x), matrix_y)

        def func(x):
            xx = np.zeros((size - 1) * 4)
            for i in range(1, size):
                if x <= points[i, 0]:
                    break
            xx[4 * (i - 1)] = x ** 3
            xx[4 * (i - 1) + 1] = x ** 2
            xx[4 * (i - 1) + 2] = x
            xx[4 * (i - 1) + 3] = 1
            return np.dot(para, xx)
    elif type == 'SPLINE':
        matrix_x = np.zeros([(size - 1) * 4, (size - 1) * 4])
        matrix_y = np.zeros([(size - 1) * 4])
        for i in range(size - 1):
            for j in range(2):
                matrix_x[2 * i + j, 4 * i] = points[i + j, 0] ** 3
                matrix_x[2 * i + j, 4 * i + 1] = points[i + j, 0] ** 2
                matrix_x[2 * i + j, 4 * i + 2] = points[i + j, 0]
                matrix_x[2 * i + j, 4 * i + 3] = 1
                matrix_y[2 * i + j] = points[i + j, 1]
        for i in range(size - 2):
            matrix_x[(size - 1) * 2 + 2 * i, 4 * i] = 3 * points[i + 1, 0] ** 2
            matrix_x[(size - 1) * 2 + 2 * i, 4 * i + 1] = 2 * points[i + 1, 0]
            matrix_x[(size - 1) * 2 + 2 * i, 4 * i + 2] = 1
            matrix_x[(size - 1) * 2 + 2 * i, 4 * i + 4] = -3 * points[i + 1, 0] ** 2
            matrix_x[(size - 1) * 2 + 2 * i, 4 * i + 5] = -2 * points[i + 1, 0]
            matrix_x[(size - 1) * 2 + 2 * i, 4 * i + 6] = -1

            matrix_x[(size - 1) * 2 + 2 * i + 1, 4 * i] = 6 * points[i + 1, 0]
            matrix_x[(size - 1) * 2 + 2 * i + 1, 4 * i + 1] = 2
            matrix_x[(size - 1) * 2 + 2 * i + 1, 4 * i + 4] = -6 * points[i + 1, 0]
            matrix_x[(size - 1) * 2 + 2 * i + 1, 4 * i + 5] = -2
            matrix_x[(size - 1) * 2 + 2 * i + 1, 4 * i + 7] = -1
        matrix_x[-2, 0] = 6 * points[0, 0]
        matrix_x[-2, 1] = 2
        matrix_x[-1, -4] = 6 * points[-1, 0]
        matrix_x[-1, -3] = 2
        para = np.dot(np.linalg.inv(matrix_x), matrix_y)

        def func(x):
            xx = np.zeros((size - 1) * 4)
            for i in range(1, size):
                if x <= points[i, 0]:
                    break
            xx[4 * (i - 1)] = x ** 3
            xx[4 * (i - 1) + 1] = x ** 2
            xx[4 * (i - 1) + 2] = x
            xx[4 * (i - 1) + 3] = 1
            return np.dot(para, xx)
    else:
        raise NotImplementedError("Unknown fitting method")
    return func


@dataclass(frozen=True)
class ModelAttr:
    """
    A class that stores the attributes of a trained model. It can be serialized in a json format.
        Args:
            name(str): 模型的名字，唯一能hash的field，是ModelAttr的唯一标识
            features(list): 模型的特征，字符串list
            labels(list): 模型的目标，字符串list
            scaled_features(list): 需要做scaling的特征，可为空
            stats(Dict): 训练集做Scaling的统计特征，数据结构为Dict of Dict
                         对于zscore， {feature1: {mean: float, std: float}, feature2: ...}
                         对于minmax， {feature1: {min: float, max: float}, feature2: ...}
    """
    name: str
    features: list = field(compare=False)
    labels: list = field(compare=False)
    scaled_feats: list = field(default=None, compare=False)
    stats: dict = field(default=None, compare=False)


class JsonModel:
    """
    A class that serializes ModelAttr to json file.
    The json file is located at 'model_path/model_attrs.json'. It contains a list of ModelAttrs.

    """
    model_path = r"e:\MyWork\PycharmProjects\FXIncome\fxincome\ml\models\\"

    @staticmethod
    def load_attr(name: str):
        """
        从本地读取某个模型的要素
            Args:
                name(str): 模型名字
            Returns:
                model(ModelAttr): 以ModelAttr形式返回Model的要素，如没有这个名字的模型，则返回None
        """
        with open(JsonModel.model_path + 'model_attrs.json', 'r') as f:
            model_string = f.read()
        if not model_string:
            return None
        jdict = json.loads(model_string)
        try:
            model = ModelAttr(name, jdict[name]['features'], jdict[name]['labels'], jdict[name]['scaled_feats'],
                              jdict[name]['stats'])
        except:
            return None
        else:
            return model

    @staticmethod
    def save_attr(model: ModelAttr):
        """
        将某个模型对应的features更新至本地文件
            Args:
                model(ModelAttr): 将要存储的Model要素的对象
        """
        name = model.name
        with open(JsonModel.model_path + 'model_attrs.json', 'r') as f:
            model_string = f.read()
        if not model_string:
            models = {}
        else:
            models = json.loads(model_string)
        models[name] = model
        with open(JsonModel.model_path + 'model_attrs.json', 'w') as f:
            f.write(json.dumps(models, default=lambda x: x.__dict__))

    @staticmethod
    def load_plain_models(names: list):
        """
        从本地读取某个传统模型的要素和模型
            Args:
                names(list): a list of strs. 传统模型名字列表
            Returns:
                plain_dict(dict): key: ModelAttr, value: model
        """
        attrs = [JsonModel.load_attr(name) for name in names]
        plain_dict = {}
        for attr in attrs:
            plain_dict[attr] = joblib.load(JsonModel.model_path + attr.name)
        return plain_dict

    @staticmethod
    def load_nn_models(names: list):
        """
        从本地读取某个模型的要素
            Args:
                names(list): a list of strs. 神经网络模型名字列表
            Returns:
                nn_dict(dict): key: ModelAttr, value: model
        """
        attrs = [JsonModel.load_attr(name) for name in names]
        nn_dict = {}
        for attr in attrs:
            nn_dict[attr] = keras.models.load_model(JsonModel.model_path + attr.name)
        return nn_dict
