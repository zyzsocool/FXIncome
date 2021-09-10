import json, joblib
import tensorflow.keras as keras
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelAttr:
    """
    A class that stores the attributes of a trained model. It can be serialized in a json format.
        Args:
            name(str): 模型的名字，唯一能hash的field，是ModelAttr的唯一标识
            features(list): 模型的特征，字符串list
            scaled_features(list): 需要做scaling的特征，可为空
            stats(Dict): 训练集做Scaling的统计特征，数据结构为Dict of Dict
                         对于zscore， {feature1: {mean: float, std: float}, feature2: ...}
                         对于minmax， {feature1: {min: float, max: float}, feature2: ...}
    """
    name: str
    features: list = field(compare=False)
    scaled_feats: list = field(default=None, compare=False)
    stats: dict = field(default=None, compare=False)


class JsonModel:
    """
    A class that serializes ModelAttr to json file.
    The json file is located at 'model_path/model_attrs.json'. It contains a list of ModelAttrs.

    """
    model_path = 'ml/models/'

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
            model = ModelAttr(name, jdict[name]['features'], jdict[name]['scaled_feats'], jdict[name]['stats'])
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
    def load_models(plain_names: list, nn_names: list):
        """
        从本地读取某个模型的要素
            Args:
                plain_names(list): a list of strs. 传统模型名字列表
                nn_names(list): a list of strs. 神经网络模型名字列表
            Returns:
                plain_dict(dict): key: ModelAttr, value: model
                nn_dict(dict): key: ModelAttr, value: model
        """
        plain_attrs = [JsonModel.load_attr(name) for name in plain_names]
        nn_attrs = [JsonModel.load_attr(name) for name in nn_names]
        plain_dict = {}
        nn_dict = {}
        for attr in plain_attrs:
            plain_dict[attr] = joblib.load(JsonModel.model_path + attr.name)
        for attr in nn_attrs:
            nn_dict[attr] = keras.models.load_model(JsonModel.model_path + attr.name)
        return plain_dict, nn_dict
