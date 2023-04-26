# LSTM-and-1D-CNN-Models-for-Real-Time-Flood-Prediction
Comparative Study of Deep Learning LSTM and 1D-CNN Models for Real-time Flood Prediction in Red River of the North, USA

This repository contains a Python script that demonstrates the use of a deep learning LSTM and 1D-CNN models to predict real-time flood levels in the Red River of the North, USA. The script includes data preprocessing, model training, evaluation, and plotting results.

## Prerequisites

- Python 3.7 or later
- Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository to your local machine:

For LSTM model:

```bash
git clone https://github.com/ramtinka/LSTM-and-1D-CNN-Models-for-Real-Time-Flood-Prediction.git
cd flood_prediction_lstm
```

2. Ensure you have the data files in a folder named `data` within the repository folder. The folder should contain two files:

- `Grand Forks.csv`: Training data
- `Grand Forks-2022.csv`: Testing data

3. Update the file paths in the `main()` function in `lstm.py` to match your local file system:

```python
train_data = load_data('data/Grand Forks.csv')
test_data = load_data('data/Grand Forks-2022.csv')
```

4. Run the script:

```bash
python3 lstm.py
```

The script will train the LSTM model using the training data and evaluate it using the testing data. The output will display the Mean Absolute Percentage Error (MAPE) of the model predictions and save a plot of the test set results as `test_set.jpg`.


