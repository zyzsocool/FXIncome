import pandas as pd
import os
import datetime
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from fxincome.ml import tbond_nn_predata
from fxincome import logger
from fxincome.const import TBOND_PARAM


def plot_graph(x_train, y_train, x_test, y_test, model):
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(y_train, label='real')
    plt.plot(model.predict(x_train), label='predict')
    plt.legend(loc='upper right')
    plt.title('train')

    plt.subplot(3, 1, 2)
    plt.plot(y_test, label='real')
    plt.plot(model.predict(x_test), label='predict')
    plt.legend(loc='upper right')
    plt.title('test')

    plt.show()

def train(train_x, train_y, val_x, val_y, test_x, test_y, model_name, batch_size=32, epochs=20):
    """
    train_x, train_y, test_x, test_y 都是numpy ndarray
        Args:
            train_x(ndarray): 3D array， [size, time steps, feature dims]
            test_x(ndarray): 3D array， [size, time steps, feature dims]
    """
    logger.info(f"shape of train_x is {train_x.shape} ")
    logger.info(f"shape of train_y is {train_y.shape} ")
    logger.info(f"shape of val_x is {val_x.shape} ")
    logger.info(f"shape of val_y is {val_y.shape} ")
    logger.info(f"shape of train_x.shape[1:] is {train_x.shape[1:]}")

    model = Sequential()
    model.add(LSTM(32, input_shape=(train_x.shape[1:]), return_sequences=True,
                   # kernel_regularizer=keras.regularizers.l2(0.01),
                   # kernel_constraint=keras.constraints.max_norm(1.)
                   ))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(LSTM(32,
                   # kernel_regularizer=keras.regularizers.l2(0.01),
                   # kernel_constraint=keras.constraints.max_norm(1.)
                   ))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-4)
    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['binary_accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    tensorboard = TensorBoard(log_dir=f"logs/{model_name}")
    checkpoint = ModelCheckpoint(filepath=f"models/Checkpoint-{model_name}.model",
                                 monitor='val_loss',
                                 verbose=1, save_best_only=True)  # saves only the best ones
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=False)
    # Train model
    history = model.fit(
        train_x, train_y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_x, val_y),
        callbacks=[tensorboard, checkpoint, early_stopping_cb],
    )
    # Score model
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def evaluate():
    """
    Use features_processed_latest.csv to evaluate the trained models
    """

    ROOT_PATH = 'd:/ProjectRicequant/fxincome/'
    src_file = 'fxincome_features.csv'
    model_name = 'models/Checkpoint-10-SEQ-1-PRED-20210903-1639.model'
    stats_name = model_name.replace('Checkpoint', 'stats')
    stats_name = 'pkl'.join(stats_name.rsplit('model', 1))  # 把右边第1个'model'换成'pkl'
    best_model = keras.models.load_model(model_name)
    stats = joblib.load(stats_name)
    df = pd.read_csv(os.path.join(ROOT_PATH, src_file), parse_dates=['date'])
    df = tbond_nn_predata.feature_engineering(df, TBOND_PARAM.ALL_FEATS, future_period=1, label_type='fwd')
    train_df, val_df, test_df, train_stats = tbond_nn_predata.pre_process(df, TBOND_PARAM.SCALED_FEATS, scale_type='zscore')
    logger.info(f"stats equal? {train_stats == stats}")

    src_file = 'fxincome_features_latest.csv'
    df = pd.read_csv(os.path.join(ROOT_PATH, src_file), parse_dates=['date'])
    df = tbond_nn_predata.feature_engineering(df, TBOND_PARAM.ALL_FEATS, future_period=1, label_type='fwd')

    latest_df, stats = tbond_nn_predata.scale(df, TBOND_PARAM.SCALED_FEATS, stats=stats, type='zscore')
    latest_df.to_csv(os.path.join(ROOT_PATH, 'latest_samples.csv'), index=False, encoding='utf-8')

    data_columns = TBOND_PARAM.NN_TRAIN_FEATS + TBOND_PARAM.LABELS
    x_days = 10
    train_x, train_y = tbond_nn_predata.gen_trainset(train_df, data_columns, TBOND_PARAM.FEAT_OUTLINERS,
                                                     seq_len=x_days, balance=False)
    val_x, val_y = tbond_nn_predata.gen_trainset(val_df, data_columns, TBOND_PARAM.FEAT_OUTLINERS,
                                                     seq_len=x_days, balance=False)
    test_x, test_y = tbond_nn_predata.gen_trainset(test_df, data_columns, TBOND_PARAM.FEAT_OUTLINERS,
                                                     seq_len=x_days, balance=False)
    latest_x, latest_y = tbond_nn_predata.gen_trainset(latest_df, data_columns, TBOND_PARAM.FEAT_OUTLINERS,
                                                   seq_len=x_days, balance=False)

    # plot_graph(train_x, train_y, test_x, test_y, model=best_model)

    best_model.summary()

    score = best_model.evaluate(train_x, train_y, verbose=0)
    logger.info(f'Best model for train samples accuracy:{score[1]:.4f}, loss:{score[0]:.4f}')
    score = best_model.evaluate(val_x, val_y, verbose=0)
    logger.info(f'Best model for val samples accuracy:{score[1]:.4f}, loss:{score[0]:.4f}')
    score = best_model.evaluate(test_x, test_y, verbose=0)
    logger.info(f'Best model for train samples accuracy:{score[1]:.4f}, loss:{score[0]:.4f}')
    score = best_model.evaluate(latest_x, latest_y, verbose=0)
    logger.info(f'Best model for latest samples accuracy:{score[1]:.4f}, loss:{score[0]:.4f}')

    # preds = best_model.predict(train_x[49:50,:,:])
    # logger.info(f"X Shape: {train_x.shape} X[:50] Shape: {train_x[49:50,:,:].shape} Prediction Shape: {preds.shape}")
    # for pred in preds:
    #     logger.info(f"pred[0]: {pred[0]}")

def main():
    x_days = 10  # 用过去10天的x数据
    y_days = 1  # 预测1天的的y数据
    predict_days = 1  # 预测1天后的

    EPOCHS = 50
    BATCH_SIZE = 32
    MODEL_NAME = f"{x_days}-SEQ-{predict_days}-PRED-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"

    ROOT_PATH = 'd:/ProjectRicequant/fxincome/'
    src_file = 'fxincome_features.csv'
    df = pd.read_csv(os.path.join(ROOT_PATH, src_file), parse_dates=['date'])
    df = tbond_nn_predata.feature_engineering(df, TBOND_PARAM.ALL_FEATS, future_period=1, label_type='fwd')
    train_df, val_df, test_df, stats = tbond_nn_predata.pre_process(df, TBOND_PARAM.SCALED_FEATS, scale_type='zscore')
    data_columns = TBOND_PARAM.NN_TRAIN_FEATS + TBOND_PARAM.LABELS
    train_x, train_y = tbond_nn_predata.gen_trainset(train_df, data_columns, TBOND_PARAM.FEAT_OUTLINERS,
                                                     seq_len=x_days)
    val_x, val_y = tbond_nn_predata.gen_trainset(val_df, data_columns, TBOND_PARAM.FEAT_OUTLINERS,
                                                 seq_len=x_days, balance=False)
    test_x, test_y = tbond_nn_predata.gen_trainset(test_df, data_columns, TBOND_PARAM.FEAT_OUTLINERS,
                                                   seq_len=x_days, balance=False)
    train(train_x, train_y, val_x, val_y, test_x, test_y, MODEL_NAME, BATCH_SIZE, EPOCHS)
    joblib.dump(stats, f"models/stats-{MODEL_NAME}.pkl")

if __name__ == '__main__':
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    # main()
    evaluate()
