import json


class ModelAttr:
    """
    A class that stores the attributes of a trained model. It can be serialized in a json format.
    """
    def __init__(self, name: str, features: list, scaled_feats: list = None, stats: dict = None):
        self.name = name
        self.features = features
        self.scaled_feats = scaled_feats
        self.stats = stats


class JsonModel:

    @staticmethod
    def load_model(name: str):
        """
        从本地读取某个模型的要素
            Args:
                name(str): 模型名字
            Returns:
                model(ModelAttr): 以ModelAttr形式返回Model的要素，如没有这个名字的模型，则返回None
        """
        with open('ml/models/model_attrs.json', 'r') as f:
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
    def save_model(model: ModelAttr):
        """
        将某个模型对应的features更新至本地文件
            Args:
                model(ModelAttr): 将要存储的Model要素的对象
        """
        name = model.name
        with open('ml/models/model_attrs.json', 'r') as f:
            model_string = f.read()
        if not model_string:
            models = {}
        else:
            models = json.loads(model_string)
        models[name] = model
        with open('ml/models/model_attrs.json', 'w') as f:
            f.write(json.dumps(models, default=lambda x: x.__dict__))
