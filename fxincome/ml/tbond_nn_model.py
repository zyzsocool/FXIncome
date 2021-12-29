import pandas as pd
import os
import datetime
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
    model.add(LSTM(64, input_shape=(train_x.shape[1:]), return_sequences=True,
                   # kernel_regularizer=keras.regularizers.l2(0.01),
                   # kernel_constraint=keras.constraints.max_norm(1.)
                   ))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(LSTM(64,
                   # kernel_regularizer=keras.regularizers.l2(0.01),
                   # kernel_constraint=keras.constraints.max_norm(1.)
                   ))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-4)
    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['binary_accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    tensorboard = TensorBoard(log_dir=f"logs/{model_name}")
    checkpoint = ModelCheckpoint(filepath=f"models/Checkpoint-{model_name}.model",
                                 monitor='val_binary_accuracy', mode='max',
                                 verbose=1, save_best_only=True)  # saves only the best ones
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=False)
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
    model.save(f"models/Final-{model_name}")

def evaluate():
    """
    Use features_processed_latest.csv to evaluate the trained models
    """

    ROOT_PATH = 'd:/ProjectRicequant/fxincome/'
    src_file = 'fxincome_features.csv'
    df = pd.read_csv(os.path.join(ROOT_PATH, src_file), parse_dates=['date'])
    df = tbond_nn_predata.feature_engineering(df, TBOND_PARAM.ALL_FEATS, future_period=1, label_type='fwd')
    train_df, val_df, test_df = tbond_nn_predata.pre_process(df, TBOND_PARAM.SCALED_FEATS, scale='zscore')

    src_file = 'fxincome_features_latest.csv'
    df = pd.read_csv(os.path.join(ROOT_PATH, src_file), parse_dates=['date'])
    df = tbond_nn_predata.feature_engineering(df, TBOND_PARAM.ALL_FEATS, future_period=1, label_type='fwd')

    latest_train, latest_val, latest_test = tbond_nn_predata.pre_process(df, TBOND_PARAM.SCALED_FEATS, scale='zscore')
    latest_train.to_csv(os.path.join(ROOT_PATH, 'train_samples.csv'), index=False, encoding='utf-8')
    latest_val.to_csv(os.path.join(ROOT_PATH, 'validation_samples.csv'), index=False, encoding='utf-8')
    latest_test.to_csv(os.path.join(ROOT_PATH, 'test_samples.csv'), index=False, encoding='utf-8')

    data_columns = TBOND_PARAM.NN_TRAIN_FEATS + TBOND_PARAM.LABELS
    x_days = 10
    train_x, train_y = tbond_nn_predata.generate_dataset(test_df, data_columns, TBOND_PARAM.FEAT_OUTLINERS,
                                                         seq_len=x_days, balance=False)
    test_x, test_y = tbond_nn_predata.generate_dataset(latest_train, data_columns, TBOND_PARAM.FEAT_OUTLINERS,
                                                       seq_len=x_days, balance=False)

    best_model = keras.models.load_model('models/Checkpoint-10-SEQ-1-PRED-20210827-1705.model')
    # final_model = keras.models.load_model('models/Final-10-SEQ-1-PRED-20210826-1709')

    # plot_graph(train_x, train_y, test_x, test_y, model=best_model)
    # plot_graph(train_x, train_y, test_x, test_y, model=final_model)

    best_model.summary()

    score = best_model.evaluate(train_x, train_y, verbose=0)
    logger.info(f'Best model for test samples test accuracy:{score[1]:.4f}, loss:{score[0]:.4f}')
    score = best_model.evaluate(test_x, test_y, verbose=0)
    logger.info(f'Best model for latest samples test accuracy:{score[1]:.4f}, loss:{score[0]:.4f}')
    # score = final_model.evaluate(train_x, train_y, verbose=0)
    # logger.info(f'Final model for test samples test accuracy:{score[1]:.4f}, loss:{score[0]:.4f}')
    # score = final_model.evaluate(test_x, test_y, verbose=0)
    # logger.info(f'Final model for latest samples test accuracy:{score[1]:.4f}, loss:{score[0]:.4f}')


def main():
    x_days = 10  # 用过去10天的x数据
    y_days = 1  # 预测1天的的y数据
    predict_days = 1  # 预测1天后的

    EPOCHS = 30
    BATCH_SIZE = 32
    MODEL_NAME = f"{x_days}-SEQ-{predict_days}-PRED-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"

    ROOT_PATH = 'd:/ProjectRicequant/fxincome/'
    src_file = 'fxincome_features.csv'
    df = pd.read_csv(os.path.join(ROOT_PATH, src_file), parse_dates=['date'])
    df = tbond_nn_predata.feature_engineering(df, TBOND_PARAM.ALL_FEATS, future_period=1, label_type='fwd')
    train_df, val_df, test_df = tbond_nn_predata.pre_process(df, TBOND_PARAM.SCALED_FEATS, scale='zscore')
    data_columns = TBOND_PARAM.NN_TRAIN_FEATS + TBOND_PARAM.LABELS
    train_x, train_y = tbond_nn_predata.generate_dataset(train_df, data_columns, TBOND_PARAM.FEAT_OUTLINERS,
                                                         seq_len=x_days)
    val_x, val_y = tbond_nn_predata.generate_dataset(val_df, data_columns, TBOND_PARAM.FEAT_OUTLINERS,
                                                         seq_len=x_days, balance=False)
    test_x, test_y = tbond_nn_predata.generate_dataset(test_df, data_columns, TBOND_PARAM.FEAT_OUTLINERS,
                                                       seq_len=x_days, balance=False)
    train(train_x, train_y, val_x, val_y, test_x, test_y, MODEL_NAME, BATCH_SIZE, EPOCHS)

if __name__ == '__main__':
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    # main()
    evaluate()
