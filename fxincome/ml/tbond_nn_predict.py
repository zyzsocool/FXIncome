import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from fxincome.ml import tbond_nn_predata
from fxincome import logger
from fxincome.const import TBOND_PARAM

data_columns = TBOND_PARAM.TRAIN_FEATS + TBOND_PARAM.LABELS
x_days = 10
# Use unseen samples to test the best model
test_data = pd.read_csv(r'd:\ProjectRicequant\fxincome\test_samples.csv', parse_dates=['date'])
test_x, test_y = tbond_nn_predata.gen_trainset(test_data, data_columns, TBOND_PARAM.FEAT_OUTLINERS,
                                               seq_len=x_days)
best_model = keras.models.load_model('models/Checkpoint-10-SEQ-1-PRED-20210824-1144.model')
score = best_model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])