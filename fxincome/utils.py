import json, joblib
import xgboost as xgb
# import tensorflow.keras as keras
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelAttr:
    """
    A class that stores the attributes of a trained model. It can be serialized in a json format.
        Args:
            name(str): 模型的名字，唯一能hash的field，是ModelAttr的唯一标识
            features(list): 模型的特征，字符串list
            labels(dict): 模型的预测目标，数据结构为Dict of Dict
                          {label1: {value_scope: str, days_forward: int, threshold: float},
                          label2: ...}
            scaled_features(list): 需要做scaling的特征，可为空
            stats(dict): 训练集做Scaling的统计特征，数据结构为Dict of Dict
                         对于zscore， {feature1: {mean: float, std: float}, feature2: ...}
                         对于minmax， {feature1: {min: float, max: float}, feature2: ...}
    """
    name: str
    features: list = field(compare=False)
    labels: dict = field(compare=False)
    scaled_feats: list = field(default=None, compare=False)
    stats: dict = field(default=None, compare=False)


class JsonModel:
    """
    A database class that CRUD ModelAttrs in a single Json file. Users can query and save model attributes by this class.
    All the ModelAttrs are stored in a single Json file.
    The json file is located at json_path. The file name is model_attrs.json.
    """
    JSON_NAME = 'model_attrs.json'

    @staticmethod
    def load_attr(name: str, json_path: str):
        """
        从本地读取某个模型的要素
            Args:
                name(str): 模型名字
                json_path(str): Json文件的路径
            Returns:
                model(ModelAttr): 以ModelAttr形式返回Model的要素，如没有这个名字的模型，则返回None
        """
        with open(json_path + JsonModel.JSON_NAME, 'r') as f:
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
    def save_attr(model: ModelAttr, json_path: str):
        """
        将某个模型对应的features更新至本地文件
            Args:
                model(ModelAttr): 将要存储的Model要素的对象
                json_path(str): Json文件的路径
        """
        name = model.name
        with open(json_path + 'model_attrs.json', 'r') as f:
            model_string = f.read()
        if not model_string:
            models = {}
        else:
            models = json.loads(model_string)
        models[name] = model
        with open(json_path + 'model_attrs.json', 'w') as f:
            f.write(json.dumps(models, default=lambda x: x.__dict__))

    @staticmethod
    def delete_attr(name: str, json_path: str):
        """
        从本地文件中删除某个模型的要素
            Args:
                name(str): 模型名字
                json_path(str): Json文件的路径
        """
        with open(json_path + 'model_attrs.json', 'r') as f:
            model_string = f.read()
        if not model_string:
            return
        models = json.loads(model_string)
        models.pop(name)
        with open(json_path + 'model_attrs.json', 'w') as f:
            f.write(json.dumps(models, default=lambda x: x.__dict__))


    @staticmethod
    def load_plain_models(names: list, json_model_path: str, serialization_type: {'joblib', 'xgb'}):
        """
        从本地读取某个传统模型的要素和模型
            Args:
                names(list): a list of strs. 传统模型名字列表
                json_model_path(str): Json文件和模型文件的路径，Json和模型必须在同一个文件夹下。
                serialization_type(str): 模型序列化时使用的方法，可选'joblib', 'xgb'
            Returns:
                plain_dict(dict): key: ModelAttr, value: model
        """
        attrs = [JsonModel.load_attr(name, json_model_path) for name in names]
        plain_dict = {}
        for attr in attrs:
            if serialization_type == 'joblib':
                plain_dict[attr] = joblib.load(json_model_path + attr.name)
            elif serialization_type == 'xgb':
                xgb_model = xgb.Booster()
                xgb_model.load_model(json_model_path + attr.name)
                plain_dict[attr] = xgb_model
            else:
                raise ValueError('Invalid serialization type.')
        return plain_dict

    # @staticmethod
    # def load_nn_models(names: list, json_path: str):
    #     """
    #     从本地读取某个模型的要素
    #         Args:
    #             names(list): a list of strs. 神经网络模型名字列表
    #             json_path(str): Json文件的路径
    #         Returns:
    #             nn_dict(dict): key: ModelAttr, value: model
    #     """
    #     attrs = [JsonModel.load_attr(name) for name in names]
    #     nn_dict = {}
    #     for attr in attrs:
    #         nn_dict[attr] = keras.models.load_model(json_path + attr.name)
    #     return nn_dict
