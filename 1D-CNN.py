import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
#from google.colab import files

# Load the data
#uploaded = files.upload()

def load_data(file_name):
    raw = pd.read_csv(file_name, encoding='unicode_escape', header=0, index_col=0, parse_dates=True, squeeze=True)
    data = raw.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

train_data, scaler_train = load_data('Grand Forks.csv')
test_data, scaler_test = load_data('Grand Forks-2022.csv')

# Create sequences of length 24 hours
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x_i = data[i:i + seq_length, :]
        y_i = data[i + seq_length, :]
        x.append(x_i)
        y.append(y_i)
    return np.array(x), np.array(y)

seq_length = 24
x, y = create_sequences(train_data, seq_length)
x_test, y_test = create_sequences(test_data, seq_length)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=False)

# Build the 1D-CNN model
def build_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    return model

# Build and compile the model
input_shape = (x_train.shape[1], x_train.shape[2])
model = build_model(input_shape)
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='mse', optimizer=optimizer)

# Define callbacks
filename = 'model_1D-CNN-training.csv'
checkpoint = [
    keras.callbacks.CSVLogger(filename, separator=",", append=True),
    keras.callbacks.ModelCheckpoint(filepath="Models/Segmentation_{epoch}.h5", save_best_only=True, save_weights_only=True)
]

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=128, verbose=1, callbacks=checkpoint)

# Plot the training loss
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

# Make predictions on the test set
prediction = model.predict(x_test)
prediction = scaler_test.inverse_transform(prediction)
y_test_actual = scaler_test.inverse_transform(y_test)

# Calculate the Mean Absolute Percentage Error (MAPE)
MAPE = mean_absolute_percentage_error(y_test_actual, prediction)
print('Test MAPE: %.3f' % MAPE)

# Plot the actual and predicted values
plt.plot(y_test_actual, label='Actual')
plt.plot(prediction, label='Prediction')
plt.legend()
plt.show()
plt.savefig("1D-CNN.jpg", dpi=300)
