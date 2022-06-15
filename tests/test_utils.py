import pytest
import joblib
import os
import tensorflow.keras
import numpy as np
from fxincome.utils import ModelAttr, JsonModel, get_curve


class TestModel:
    @pytest.fixture(scope='class')
    def global_data(self):
        os.chdir('../fxincome')
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
        xgb_name = '0.626-1d_fwd-XGB-20210618-1454-v2016.pkl'
        xgb_features = ['close', 'pct_chg', 'avg_chg_5', 'avg_chg_10', 'fr007_chg_5', 'spread_t1y',
                        'spread_fr007', 'spread_usdcny', 'usdcny_chg_5']
        rfc_name = '0.605-1d_fwd-RFC-20210619-1346-v2018.pkl'
        rfc_features = xgb_features
        labels = ['target']
        lstm_model = ModelAttr(lstm_name, lstm_features, labels, lstm_scaled_feats, lstm_stats)
        xgb_model = ModelAttr(xgb_name, xgb_features, labels)
        rfc_model = ModelAttr(rfc_name, rfc_features, labels)
        return {
            'lstm_name': lstm_name,
            'lstm_features': lstm_features,
            'lstm_scaled_feats': lstm_scaled_feats,
            'lstm_stats': lstm_stats,
            'xgb_name': xgb_name,
            'xgb_features': xgb_features,
            'rfc_name': rfc_name,
            'labels': labels,
            'lstm_model': lstm_model,
            'xgb_model': xgb_model,
            'rfc_model': rfc_model
        }

    def test_save_attr(self, global_data):
        lstm_model = global_data['lstm_model']
        JsonModel.save_attr(lstm_model)
        model = JsonModel.load_attr(lstm_model.name)
        assert lstm_model.name == model.name
        assert lstm_model.features == model.features
        assert lstm_model.scaled_feats == model.scaled_feats
        assert lstm_model.stats == model.stats
        JsonModel.save_attr(global_data['xgb_model'])
        JsonModel.save_attr(global_data['rfc_model'])

    def test_load_attr(self, global_data):
        xgb_model = global_data['xgb_model']
        model = JsonModel.load_attr(xgb_model.name)
        assert xgb_model.name == model.name
        assert xgb_model.features == model.features
        assert JsonModel.load_attr('Non-Exists') is None

    def test_load_plain_models(self, global_data):
        plain_names = [global_data['xgb_name'], global_data['rfc_name']]
        plain_dict = JsonModel.load_plain_models(plain_names)
        xgb_model = joblib.load(JsonModel.model_path + global_data['xgb_name'])
        plain_model = plain_dict[global_data['xgb_model']]
        assert plain_model.get_params()['gamma'] == xgb_model.get_params()['gamma']

    def test_load_nn_models(self, global_data):
        nn_names = [global_data['lstm_name']]
        nn_dict = JsonModel.load_nn_models(nn_names)
        lstm_model = tensorflow.keras.models.load_model(JsonModel.model_path + global_data['lstm_name'])
        nn_model = nn_dict[global_data['lstm_model']]
        assert nn_model.summary() == lstm_model.summary()


class TestCurve:

    @pytest.fixture(scope='class')
    def global_data(self):
        points = np.array([[0, 1.5855],
                           [1, 2.3438],
                           [2, 2.5848],
                           [3, 2.6617],
                           [4, 2.7545],
                           [5, 2.8526],
                           [6, 2.9594],
                           [7, 3.0125],
                           [8, 3.0022],
                           [9, 2.9863],
                           [10, 2.9879],
                           [15, 3.3447],
                           [20, 3.3764],
                           [30, 3.5329],
                           [40, 3.5785],
                           [50, 3.595]])

        return {'points': points,
                'linear_point_between': 2.7081,
                'hermit_point_between': 2.706775,
                'ytm_3y': 2.6617,
                'ytm_5y': 2.8526}

    def test_get_curve(self, global_data):
        linear_fitting = get_curve(global_data['points'], 'LINEAR')
        hermit_fitting = get_curve(global_data['points'], 'HERMIT')
        assert global_data['ytm_3y'] == pytest.approx(linear_fitting(3))
        assert global_data['ytm_3y'] == pytest.approx(hermit_fitting(3))
        assert global_data['ytm_5y'] == pytest.approx(linear_fitting(5))
        assert global_data['ytm_5y'] == pytest.approx(hermit_fitting(5))
        assert global_data['linear_point_between'] == pytest.approx(linear_fitting(3.5))
        assert global_data['hermit_point_between'] == pytest.approx(hermit_fitting(3.5))
