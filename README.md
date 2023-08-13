# LSTM-and-1D-CNN-Models-for-Real-Time-Flood-Prediction
Comparative Study of Deep Learning LSTM and 1D-CNN Models for Real-time Flood Prediction in Red River of the North, USA

This repository contains a Python script demonstrating the use of deep learning LSTM and 1D-CNN models to predict real-time flood levels in the North, USA Red River. The script includes data preprocessing, model training, evaluation, and plotting results.

## Prerequisites

- Python 3.7 or later
- Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

# LSTM Model:

1. Clone the repository to your local machine:


```bash
git clone https://github.com/ramtinka/LSTM-and-1D-CNN-Models-for-Real-Time-Flood-Prediction.git

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

The script will train the LSTM and 1D-CNN models using the training data and evaluate them using the testing data. The output will display the model predictions' Mean Absolute Percentage Error (MAPE).


## Cite Us: 

If you find this code beneficial for your research, we kindly request that you cite our work:

[Comparative Study of Deep Learning LSTM and 1D-CNN Models for Real-time Flood Prediction in Red River of the North, USA](https://ieeexplore.ieee.org/abstract/document/10187358)
```
@InProceedings{Atashi2023,
  title =        {Comparative Study of Deep Learning LSTM and 1D-CNN Models for Real-time Flood Prediction in Red River of the North, USA},
  author =       {Atashi, V. and Kardan, R. and Gorji, H. T. and Lim, Y. H.},
  booktitle =    {2023 IEEE International Conference on Electro Information Technology (eIT)},
  pages =        {022-028},
  year =         {2023},
  month =        {May},
  publisher =    {IEEE},
  url =          {https://ieeexplore.ieee.org/abstract/document/10187358}
}
```
---

