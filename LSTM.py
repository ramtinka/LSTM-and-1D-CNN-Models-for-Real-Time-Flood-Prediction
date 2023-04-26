import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
#from google.colab import files


def load_data(file_name):
    #uploaded = files.upload()
    raw_data = pd.read_csv(file_name, encoding='unicode_escape', header=0, index_col=0, parse_dates=True, squeeze=True)
    return raw_data.values.reshape(-1, 1)


def scale_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data), scaler


def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x_i = data[i:i + seq_length, :]
        y_i = data[i + seq_length, :]
        x.append(x_i)
        y.append(y_i)
    return np.array(x), np.array(y)


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dropout(0.25))
    model.add(Dense(input_shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def plot_results(history, y_actual, y_pred, output_filename):
    plt.figure()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(y_actual, label='Actual')
    plt.plot(y_pred, label='Prediction')
    plt.legend()
    plt.show()
    plt.savefig(output_filename, dpi=300)


def main():
    seq_length = 24

    train_data = load_data('Grand Forks.csv')
    scaled_train_data, train_scaler = scale_data(train_data)

    test_data = load_data('Grand Forks-2022.csv')
    scaled_test_data, test_scaler = scale_data(test_data)

    x_train_full, y_train_full = create_sequences(scaled_train_data, seq_length)
    x_test, y_test = create_sequences(scaled_test_data, seq_length)

    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, shuffle=False)

    model = build_model((x_train.shape[1], x_train.shape[2]))
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=128, verbose=1)

    y_pred = model.predict(x_test)
    y_pred_actual = test_scaler.inverse_transform(y_pred)
    y_test_actual = test_scaler.inverse_transform(y_test)

    mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual)
    print(f'Test MAPE: {mape:.3f}')

    plot_results(history, y_test_actual, y_pred_actual, "test_set.jpg")


if __name__ == '__main__':
    main()