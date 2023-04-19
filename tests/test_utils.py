import pytest
import joblib
# import tensorflow.keras
import numpy as np
from fxincome.utils import ModelAttr, JsonModel
from fxincome.const import PATH


class TestModel:
    @pytest.fixture(scope='class')
    def global_data(self):
        lstm_name = 'Checkpoint-10-SEQ-1-PRED-20210903-1639.model'
        lstm_features = [
            'close',  # scale
            'amount',  # scale
            't10y',  # scale
            'fr007_5y',  # scale
            'pct_chg',
            'avg_chg_10',
            'spread_t10y',
            'spread_fr007',
            'spread_fr007_5y',
            'spread_usdcny'  # scale
        ]
        lstm_scaled_feats = ['close', 'amount', 't10y', 'fr007_5y', 'spread_usdcny']
        lstm_stats = joblib.load('ml/models/stats-10-SEQ-1-PRED-20210903-1639.pkl')
        lstm_labels = {'target': {'value_scope': '[0,1]'}}
        lstm_model = ModelAttr(
            name=lstm_name,
            features=lstm_features,
            labels=lstm_labels,
            scaled_feats=lstm_scaled_feats,
            stats=lstm_stats
        )
        xgb_name = '0.626-1d_fwd-XGB-20210618-1454-v2016.pkl'
        xgb_features = ['close', 'pct_chg', 'avg_chg_5', 'avg_chg_10', 'fr007_chg_5', 'spread_t1y',
                        'spread_fr007', 'spread_usdcny', 'usdcny_chg_5']
        xgb_labels = {'LABEL': {'value_scope': '[0,1]',
                                'days_forward': 10,
                                'spread_threshold': -0.01}}
        xgb_model = ModelAttr(
            name=xgb_name,
            features=xgb_features,
            labels=xgb_labels
        )

        lr_name = '0.605-1d_fwd-RFC-20210619-1346-v2018.pkl'
        lr_model = ModelAttr(
            name=lr_name,
            features=xgb_features,
            labels=xgb_labels
        )
        return {
            'lstm_name': lstm_name,
            'lstm_features': lstm_features,
            'lstm_labels': lstm_labels,
            'lstm_scaled_feats': lstm_scaled_feats,
            'lstm_stats': lstm_stats,
            'xgb_name': xgb_name,
            'xgb_features': xgb_features,
            'xgb_labels': xgb_labels,
            'lr_name': lr_name,
            'lr_features': xgb_features,
            'lr_labels': xgb_labels,
            'lstm_model': lstm_model,
            'xgb_model': xgb_model,
            'lr_model': lr_model
        }

    def test_save_attr(self, global_data):
        lstm_model = global_data['lstm_model']
        JsonModel.save_attr(lstm_model, PATH.YTM_MODEL)
        model = JsonModel.load_attr(lstm_model.name, PATH.YTM_MODEL)
        assert lstm_model.name == model.name
        assert lstm_model.features == model.features
        assert lstm_model.labels == model.labels
        assert lstm_model.scaled_feats == model.scaled_feats
        assert lstm_model.stats == model.stats
        JsonModel.save_attr(global_data['xgb_model'], PATH.SPREAD_MODEL)
        JsonModel.save_attr(global_data['lr_model'], PATH.SPREAD_MODEL)

    def test_load_attr(self, global_data):
        xgb_model = global_data['xgb_model']
        model = JsonModel.load_attr(xgb_model.name, PATH.SPREAD_MODEL)
        assert xgb_model.name == model.name
        assert xgb_model.features == model.features
        assert xgb_model.labels == model.labels
        assert JsonModel.load_attr('Non-Exists', PATH.SPREAD_MODEL) is None

    def test_delete_attr(self, global_data):
        xgb_model = global_data['xgb_model']
        JsonModel.delete_attr(xgb_model.name, PATH.SPREAD_MODEL)
        assert JsonModel.load_attr(xgb_model.name, PATH.SPREAD_MODEL) is None
        JsonModel.save_attr(global_data['xgb_model'], PATH.SPREAD_MODEL)
        model = JsonModel.load_attr(xgb_model.name, PATH.SPREAD_MODEL)
        assert model.labels == xgb_model.labels

    def test_load_plain_models(self, global_data):
        plain_names = [global_data['xgb_name']]
        plain_dict = JsonModel.load_plain_models(plain_names, PATH.SPREAD_MODEL, 'xgb')
        xgb_model = joblib.load(PATH.SPREAD_MODEL + global_data['xgb_name'])
        plain_model = plain_dict[global_data['xgb_model']]
        assert plain_model.get_params()['gamma'] == xgb_model.get_params()['gamma']
        plain_names = [global_data['lr_name']]
        plain_dict = JsonModel.load_plain_models(plain_names, PATH.SPREAD_MODEL, 'joblib')
        lr_model = joblib.load(PATH.SPREAD_MODEL + global_data['lr_name'])
        plain_model = plain_dict[global_data['lr_model']]
        assert plain_model.get_params()['logistic__C'] == lr_model.get_params()['logistic__C']

    # def test_load_nn_models(self, global_data):
    #     nn_names = [global_data['lstm_name']]
    #     nn_dict = JsonModel.load_nn_models(nn_names)
    #     lstm_model = tensorflow.keras.models.load_model(JsonModel.model_path + global_data['lstm_name'])
    #     nn_model = nn_dict[global_data['lstm_model']]
    #     assert nn_model.summary() == lstm_model.summary()


