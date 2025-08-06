
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.models import  clone_model as clone_DNN
from sklearn.base import clone as clone_ML

import os

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import joblib
from lightgbm import early_stopping
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#2021 data
csv_data_folder_2021 = ".../droplet_data_original_LLM_output/"
model_folder_DropletGenerationRate_2021 = '.../Forward/DropletGenerationRate/'
model_folder_DropletDiameter_2021 = '.../Forward/DropletDiameter/'

#2024 data
csv_data_folder_2024 = ".../Forward/droplet_data_original_LLM_output/"
model_folder_DropletGenerationRate_2024 = '.../Forward/DropletGenerationRate/'
model_folder_DropletDiameter_2024 = '..../Forward/DropletDiameter/'


#data
csv_data_folder = csv_data_folder_2021
model_folder_DropletGenerationRate = model_folder_DropletGenerationRate_2021
model_folder_DropletDiameter = model_folder_DropletDiameter_2021

path_X_Orginal = csv_data_folder+ "X_droplet_imbalance_Original_data_Forward.csv"
path_Y_Orginal = csv_data_folder+ "Y_droplet_imbalance_Original_data_Forward.csv"

path_X_LLAVA = csv_data_folder + "X_droplet_imbalance_data_llava:7b.csv"
path_Y_LLAVA = csv_data_folder + "Y_droplet_imbalance_data_llava:7b.csv"


path_X_DEEPSEEK = csv_data_folder + "X_droplet_imbalance_data_deepseek-r1:8b.csv"
path_Y_DEEPSEEK = csv_data_folder + "Y_droplet_imbalance_data_deepseek-r1:8b.csv"

path_X_GEMMA = csv_data_folder + "X_droplet_imbalance_data_gemma2.csv"
path_Y_GEMMA = csv_data_folder + "Y_droplet_imbalance_data_gemma2.csv"

path_X_LLAMA = csv_data_folder + "X_droplet_imbalance_data_llama3.1.csv"
path_Y_LLAMA = csv_data_folder + "Y_droplet_imbalance_data_llama3.1.csv"

path_X_MISTRAL = csv_data_folder + "X_droplet_imbalance_data_mistral.csv"
path_Y_MISTRAL = csv_data_folder + "Y_droplet_imbalance_data_mistral.csv"



#DropletDiameter

#LLAVA
fileName_DNN_DropletDiameter_LLAVA = model_folder_DropletDiameter + 'DNN_LLAVA_DropletDiameter_2021_Forward_Embedding.keras'
fileName_XGBOOST_DropletDiameter_LLAVA = model_folder_DropletDiameter + 'XGBOOST_LLAVA_DropletDiameter_2021_Forward_Embedding.pkl'
fileName_LightGBM_DropletDiameter_LLAVA = model_folder_DropletDiameter +'LightGBM_LLAVA_DropletDiameter_2021_Forward_Embedding.pkl'
fileName_SVM_DropletDiameter_LLAVA = model_folder_DropletDiameter + 'SVM_LLAVA_DropletDiameter_2021_Forward_Embedding.pkl'

model_DNN_LLAVA_Regression_DropletDiameter = load_model(fileName_DNN_DropletDiameter_LLAVA)
model_XGBOOST_LLAVA_Regression_DropletDiameter = joblib.load(fileName_XGBOOST_DropletDiameter_LLAVA)
model_LightGBM_LLAVA_Regression_DropletDiameter = joblib.load(fileName_LightGBM_DropletDiameter_LLAVA)
model_SVM_LLAVA_Regression_DropletDiameter = joblib.load(fileName_SVM_DropletDiameter_LLAVA)


#DEEPSEEK
fileName_DNN_DropletDiameter_DEEPSEEK = model_folder_DropletDiameter + 'DNN_DEEPSEEK_DropletDiameter_2021_Forward_Embedding.keras'
fileName_XGBOOST_DropletDiameter_DEEPSEEK = model_folder_DropletDiameter +'XGBOOST_DEEPSEEK_DropletDiameter_2021_Forward_Embedding.pkl'
fileName_LightGBM_DropletDiameter_DEEPSEEK = model_folder_DropletDiameter +'LightGBM_DEEPSEEK_DropletDiameter_2021_Forward_Embedding.pkl'
fileName_SVM_DropletDiameter_DEEPSEEK = model_folder_DropletDiameter +'SVM_DEEPSEEK_DropletDiameter_2021_Forward_Embedding.pkl'

model_DNN_DEEPSEEK_Regression_DropletDiameter = load_model(fileName_DNN_DropletDiameter_DEEPSEEK)
model_XGBOOST_DEEPSEEK_Regression_DropletDiameter = joblib.load(fileName_XGBOOST_DropletDiameter_DEEPSEEK)
model_LightGBM_DEEPSEEK_Regression_DropletDiameter = joblib.load(fileName_LightGBM_DropletDiameter_DEEPSEEK)
model_SVM_DEEPSEEK_Regression_DropletDiameter = joblib.load(fileName_SVM_DropletDiameter_DEEPSEEK)

#GEMMA
fileName_DNN_DropletDiameter_GEMMA = model_folder_DropletDiameter + 'DNN_GEMMA_DropletDiameter_2021_Forward_Embedding.keras'
fileName_XGBOOST_DropletDiameter_GEMMA = model_folder_DropletDiameter +'XGBOOST_GEMMA_DropletDiameter_2021_Forward_Embedding.pkl'
fileName_LightGBM_DropletDiameter_GEMMA = model_folder_DropletDiameter +'LightGBM_GEMMA_DropletDiameter_2021_Forward_Embedding.pkl'
fileName_SVM_DropletDiameter_GEMMA = model_folder_DropletDiameter +'SVM_GEMMA_DropletDiameter_2021_Forward_Embedding.pkl'

model_DNN_GEMMA_Regression_DropletDiameter = load_model(fileName_DNN_DropletDiameter_GEMMA)
model_XGBOOST_GEMMA_Regression_DropletDiameter = joblib.load(fileName_XGBOOST_DropletDiameter_GEMMA)
model_LightGBM_GEMMA_Regression_DropletDiameter = joblib.load(fileName_LightGBM_DropletDiameter_GEMMA)
model_SVM_GEMMA_Regression_DropletDiameter = joblib.load(fileName_SVM_DropletDiameter_GEMMA)

#LLAMA
fileName_DNN_DropletDiameter_LLAMA = model_folder_DropletDiameter + 'DNN_LLAMA_DropletDiameter_2021_Forward_Embedding.keras'
fileName_XGBOOST_DropletDiameter_LLAMA = model_folder_DropletDiameter +'XGBOOST_LLAMA_DropletDiameter_2021_Forward_Embedding.pkl'
fileName_LightGBM_DropletDiameter_LLAMA = model_folder_DropletDiameter +'LightGBM_LLAMA_DropletDiameter_2021_Forward_Embedding.pkl'
fileName_SVM_DropletDiameter_LLAMA = model_folder_DropletDiameter +'SVM_LLAMA_DropletDiameter_2021_Forward_Embedding.pkl'

model_DNN_LLAMA_Regression_DropletDiameter = load_model(fileName_DNN_DropletDiameter_LLAMA)
model_XGBOOST_LLAMA_Regression_DropletDiameter = joblib.load(fileName_XGBOOST_DropletDiameter_LLAMA)
model_LightGBM_LLAMA_Regression_DropletDiameter = joblib.load(fileName_LightGBM_DropletDiameter_LLAMA)
model_SVM_LLAMA_Regression_DropletDiameter = joblib.load(fileName_SVM_DropletDiameter_LLAMA)


#MISTRAL
fileName_DNN_DropletDiameter_MISTRAL = model_folder_DropletDiameter + 'DNN_MISTRAL_DropletDiameter_2021_Forward_Embedding.keras'
fileName_XGBOOST_DropletDiameter_MISTRAL = model_folder_DropletDiameter +'XGBOOST_MISTRAL_DropletDiameter_2021_Forward_Embedding.pkl'
fileName_LightGBM_DropletDiameter_MISTRAL = model_folder_DropletDiameter +'LightGBM_MISTRAL_DropletDiameter_2021_Forward_Embedding.pkl'
fileName_SVM_DropletDiameter_MISTRAL = model_folder_DropletDiameter +'SVM_MISTRAL_DropletDiameter_2021_Forward_Embedding.pkl'

model_DNN_MISTRAL_Regression_DropletDiameter = load_model(fileName_DNN_DropletDiameter_MISTRAL)
model_XGBOOST_MISTRAL_Regression_DropletDiameter = joblib.load(fileName_XGBOOST_DropletDiameter_MISTRAL)
model_LightGBM_MISTRAL_Regression_DropletDiameter = joblib.load(fileName_LightGBM_DropletDiameter_MISTRAL)
model_SVM_MISTRAL_Regression_DropletDiameter = joblib.load(fileName_SVM_DropletDiameter_MISTRAL)

#Original
fileName_DNN_DropletDiameter_Original = model_folder_DropletDiameter +'DNN_DropletDiameter_2021_Forward_RawFeature.keras'
fileName_XGBOOST_DropletDiameter_Original = model_folder_DropletDiameter + 'XGBOOST_DropletDiameter_2021_Forward_RawFeature.pkl'
fileName_LightGBM_DropletDiameter_Original = model_folder_DropletDiameter + 'LightGBM_DropletDiameter_2021_Forward_RawFeature.pkl'
fileName_SVM_DropletDiameter_Original = model_folder_DropletDiameter + 'SVM_DropletDiameter_2021_Forward_RawFeature.pkl'

model_DNN_Original_Regression_DropletDiameter = load_model(fileName_DNN_DropletDiameter_Original)
model_XGBOOST_Original_Regression_DropletDiameter = joblib.load(fileName_XGBOOST_DropletDiameter_Original)
model_LightGBM_Original_Regression_DropletDiameter = joblib.load(fileName_LightGBM_DropletDiameter_Original)
model_SVM_Original_Regression_DropletDiameter = joblib.load(fileName_SVM_DropletDiameter_Original)







#DropletGenerationRate

#LLAVA
fileName_DNN_DropletGenerationRate_LLAVA = model_folder_DropletGenerationRate + 'DNN_LLAVA_DropletGenerationRate_2021_Forward_Embedding.keras'
fileName_XGBOOST_DropletGenerationRate_LLAVA = model_folder_DropletGenerationRate + 'XGBOOST_LLAVA_DropletGenerationRate_2021_Forward_Embedding.pkl'
fileName_LightGBM_DropletGenerationRate_LLAVA = model_folder_DropletGenerationRate + 'LightGBM_LLAVA_DropletGenerationRate_2021_Forward_Embedding.pkl'
fileName_SVM_DropletGenerationRate_LLAVA = model_folder_DropletGenerationRate + 'SVM_LLAVA_DropletGenerationRate_2021_Forward_Embedding.pkl'

model_DNN_LLAVA_Regression_DropletGenerationRate = load_model(fileName_DNN_DropletGenerationRate_LLAVA)
model_XGBOOST_LLAVA_Regression_DropletGenerationRate = joblib.load(fileName_XGBOOST_DropletGenerationRate_LLAVA)
model_LightGBM_LLAVA_Regression_DropletGenerationRate = joblib.load(fileName_LightGBM_DropletGenerationRate_LLAVA)
model_SVM_LLAVA_Regression_DropletGenerationRate = joblib.load(fileName_SVM_DropletGenerationRate_LLAVA)


#DEEPSEEK
fileName_DNN_DropletGenerationRate_DEEPSEEK = model_folder_DropletGenerationRate + 'DNN_DEEPSEEK_DropletGenerationRate_2021_Forward_Embedding.keras'
fileName_XGBOOST_DropletGenerationRate_DEEPSEEK = model_folder_DropletGenerationRate + 'XGBOOST_DEEPSEEK_DropletGenerationRate_2021_Forward_Embedding.pkl'
fileName_LightGBM_DropletGenerationRate_DEEPSEEK = model_folder_DropletGenerationRate + 'LightGBM_DEEPSEEK_DropletGenerationRate_2021_Forward_Embedding.pkl'
fileName_SVM_DropletGenerationRate_DEEPSEEK = model_folder_DropletGenerationRate + 'SVM_DEEPSEEK_DropletGenerationRate_2021_Forward_Embedding.pkl'

model_DNN_DEEPSEEK_Regression_DropletGenerationRate = load_model(fileName_DNN_DropletGenerationRate_DEEPSEEK)
model_XGBOOST_DEEPSEEK_Regression_DropletGenerationRate = joblib.load(fileName_XGBOOST_DropletGenerationRate_DEEPSEEK)
model_LightGBM_DEEPSEEK_Regression_DropletGenerationRate = joblib.load(fileName_LightGBM_DropletGenerationRate_DEEPSEEK)
model_SVM_DEEPSEEK_Regression_DropletGenerationRate = joblib.load(fileName_SVM_DropletGenerationRate_DEEPSEEK)


#GEMMA
fileName_DNN_DropletGenerationRate_GEMMA = model_folder_DropletGenerationRate + 'DNN_GEMMA_DropletGenerationRate_2021_Forward_Embedding.keras'
fileName_XGBOOST_DropletGenerationRate_GEMMA =  model_folder_DropletGenerationRate + 'XGBOOST_GEMMA_DropletGenerationRate_2021_Forward_Embedding.pkl'
fileName_LightGBM_DropletGenerationRate_GEMMA = model_folder_DropletGenerationRate + 'LightGBM_GEMMA_DropletGenerationRate_2021_Forward_Embedding.pkl'
fileName_SVM_DropletGenerationRate_GEMMA = model_folder_DropletGenerationRate + 'SVM_GEMMA_DropletGenerationRate_2021_Forward_Embedding.pkl'

model_DNN_GEMMA_Regression_DropletGenerationRate = load_model(fileName_DNN_DropletGenerationRate_GEMMA)
model_XGBOOST_GEMMA_Regression_DropletGenerationRate = joblib.load(fileName_XGBOOST_DropletGenerationRate_GEMMA)
model_LightGBM_GEMMA_Regression_DropletGenerationRate = joblib.load(fileName_LightGBM_DropletGenerationRate_GEMMA)
model_SVM_GEMMA_Regression_DropletGenerationRate = joblib.load(fileName_SVM_DropletGenerationRate_GEMMA)

#LLAMA
fileName_DNN_DropletGenerationRate_LLAMA = model_folder_DropletGenerationRate + 'DNN_LLAMA_DropletGenerationRate_2021_Forward_Embedding.keras'
fileName_XGBOOST_DropletGenerationRate_LLAMA = model_folder_DropletGenerationRate + 'XGBOOST_LLAMA_DropletGenerationRate_2021_Forward_Embedding.pkl'
fileName_LightGBM_DropletGenerationRate_LLAMA = model_folder_DropletGenerationRate + 'LightGBM_LLAMA_DropletGenerationRate_2021_Forward_Embedding.pkl'
fileName_SVM_DropletGenerationRate_LLAMA = model_folder_DropletGenerationRate + 'SVM_LLAMA_DropletGenerationRate_2021_Forward_Embedding.pkl'

model_DNN_LLAMA_Regression_DropletGenerationRate = load_model(fileName_DNN_DropletGenerationRate_LLAMA)
model_XGBOOST_LLAMA_Regression_DropletGenerationRate = joblib.load(fileName_XGBOOST_DropletGenerationRate_LLAMA)
model_LightGBM_LLAMA_Regression_DropletGenerationRate = joblib.load(fileName_LightGBM_DropletGenerationRate_LLAMA)
model_SVM_LLAMA_Regression_DropletGenerationRate = joblib.load(fileName_SVM_DropletGenerationRate_LLAMA)

#MISTRAL
fileName_DNN_DropletGenerationRate_MISTRAL = model_folder_DropletGenerationRate +'DNN_MISTRAL_DropletGenerationRate_2021_Forward_Embedding.keras'
fileName_XGBOOST_DropletGenerationRate_MISTRAL = model_folder_DropletGenerationRate + 'XGBOOST_MISTRAL_DropletGenerationRate_2021_Forward_Embedding.pkl'
fileName_LightGBM_DropletGenerationRate_MISTRAL = model_folder_DropletGenerationRate + 'LightGBM_MISTRAL_DropletGenerationRate_2021_Forward_Embedding.pkl'
fileName_SVM_DropletGenerationRate_MISTRAL = model_folder_DropletGenerationRate + 'SVM_MISTRAL_DropletGenerationRate_2021_Forward_Embedding.pkl'

model_DNN_MISTRAL_Regression_DropletGenerationRate = load_model(fileName_DNN_DropletGenerationRate_MISTRAL)
model_XGBOOST_MISTRAL_Regression_DropletGenerationRate = joblib.load(fileName_XGBOOST_DropletGenerationRate_MISTRAL)
model_LightGBM_MISTRAL_Regression_DropletGenerationRate = joblib.load(fileName_LightGBM_DropletGenerationRate_MISTRAL)
model_SVM_MISTRAL_Regression_DropletGenerationRate = joblib.load(fileName_SVM_DropletGenerationRate_MISTRAL)

#Original
fileName_DNN_DropletGenerationRate_Original = model_folder_DropletGenerationRate +'DNN_DropletGenerationRate_2021_Forward_RawFeature.keras'
fileName_XGBOOST_DropletGenerationRate_Original = model_folder_DropletGenerationRate + 'XGBOOST_DropletGenerationRate_2021_Forward_RawFeature.pkl'
fileName_LightGBM_DropletGenerationRate_Original = model_folder_DropletGenerationRate + 'LightGBM_DropletGenerationRate_2021_Forward_RawFeature.pkl'
fileName_SVM_DropletGenerationRate_Original = model_folder_DropletGenerationRate + 'SVM_DropletGenerationRate_2021_Forward_RawFeature.pkl'

model_DNN_Original_Regression_DropletGenerationRate = load_model(fileName_DNN_DropletGenerationRate_Original)
model_XGBOOST_Original_Regression_DropletGenerationRate = joblib.load(fileName_XGBOOST_DropletGenerationRate_Original)
model_LightGBM_Original_Regression_DropletGenerationRate = joblib.load(fileName_LightGBM_DropletGenerationRate_Original)
model_SVM_Original_Regression_DropletGenerationRate = joblib.load(fileName_SVM_DropletGenerationRate_Original)




model_DropletGenerationRate={

    #LLAVA
    'DNN-LLAVA': model_DNN_LLAVA_Regression_DropletGenerationRate,
    'XGBoost-LLAVA': model_XGBOOST_LLAVA_Regression_DropletGenerationRate,
    'LightGBM-LLAVA': model_LightGBM_LLAVA_Regression_DropletGenerationRate,
    'SVM-LLAVA': model_SVM_LLAVA_Regression_DropletGenerationRate,


    #DEEPSEEK
    'DNN-DEEPSEEK-R1': model_DNN_DEEPSEEK_Regression_DropletGenerationRate,
    'XGBoost-DEEPSEEK-R1': model_XGBOOST_DEEPSEEK_Regression_DropletGenerationRate,
    'LightGBM-DEEPSEEK-R1': model_LightGBM_DEEPSEEK_Regression_DropletGenerationRate,
    'SVM-DEEPSEEK-R1':model_SVM_DEEPSEEK_Regression_DropletGenerationRate,

    # #GEMMA
    'DNN-GEMMA2': model_DNN_GEMMA_Regression_DropletGenerationRate,
    'XGBoost-GEMMA2': model_XGBOOST_GEMMA_Regression_DropletGenerationRate,
    'LightGBM-GEMMA2': model_LightGBM_GEMMA_Regression_DropletGenerationRate,
    'SVM-GEMMA2': model_SVM_GEMMA_Regression_DropletGenerationRate,

    # #LLAMA
    'DNN-LLAMA3.1': model_DNN_LLAMA_Regression_DropletGenerationRate,
    'XGBoost-LLAMA3.1': model_XGBOOST_LLAMA_Regression_DropletGenerationRate,
    'LightGBM-LLAMA3.1': model_LightGBM_LLAMA_Regression_DropletGenerationRate,
    'SVM-LLAMA3.1': model_SVM_LLAMA_Regression_DropletGenerationRate,

    # # MISTRAL
    'DNN-MISTRAL': model_DNN_MISTRAL_Regression_DropletGenerationRate,
    'XGBoost-MISTRAL': model_XGBOOST_MISTRAL_Regression_DropletGenerationRate,
    'LightGBM-MISTRAL': model_LightGBM_MISTRAL_Regression_DropletGenerationRate,
    'SVM-MISTRAL': model_SVM_MISTRAL_Regression_DropletGenerationRate,

    # #Original
    'DNN': model_DNN_Original_Regression_DropletGenerationRate,
    'XGBoost': model_XGBOOST_Original_Regression_DropletGenerationRate,
    'LightGBM': model_LightGBM_Original_Regression_DropletGenerationRate,
    'SVM': model_SVM_Original_Regression_DropletGenerationRate

}


model_DropletDiameter={

    # LLAVA
    'DNN-LLAVA': model_DNN_LLAVA_Regression_DropletDiameter,
    'XGBoost-LLAVA': model_XGBOOST_LLAVA_Regression_DropletDiameter,
    'LightGBM-LLAVA': model_LightGBM_LLAVA_Regression_DropletDiameter,
    'SVM-LLAVA': model_SVM_LLAVA_Regression_DropletDiameter,


    #DEEPSEEK
    'DNN-DEEPSEEK-R1': model_DNN_DEEPSEEK_Regression_DropletDiameter,
    'XGBoost-DEEPSEEK-R1': model_XGBOOST_DEEPSEEK_Regression_DropletDiameter,
    'LightGBM-DEEPSEEK-R1': model_LightGBM_DEEPSEEK_Regression_DropletDiameter,
    'SVM-DEEPSEEK-R1':model_SVM_DEEPSEEK_Regression_DropletDiameter,

    # #GEMMA2
    'DNN-GEMMA2': model_DNN_GEMMA_Regression_DropletDiameter,
    'XGBoost-GEMMA2': model_XGBOOST_GEMMA_Regression_DropletDiameter,
    'LightGBM-GEMMA2': model_LightGBM_GEMMA_Regression_DropletDiameter,
    'SVM-GEMMA2': model_SVM_GEMMA_Regression_DropletDiameter,

    #LLAMA
    'DNN-LLAMA3.1': model_DNN_LLAMA_Regression_DropletDiameter,
    'XGBoost-LLAMA3.1': model_XGBOOST_LLAMA_Regression_DropletDiameter,
    'LightGBM-LLAMA3.1': model_LightGBM_LLAMA_Regression_DropletDiameter,
    'SVM-LLAMA3.1': model_SVM_LLAMA_Regression_DropletDiameter,

    ## MISTRAL
    'DNN-MISTRAL': model_DNN_MISTRAL_Regression_DropletDiameter,
    'XGBoost-MISTRAL': model_XGBOOST_MISTRAL_Regression_DropletDiameter,
    'LightGBM-MISTRAL': model_LightGBM_MISTRAL_Regression_DropletDiameter,
    'SVM-MISTRAL': model_SVM_MISTRAL_Regression_DropletDiameter,

    # #Original
    'DNN': model_DNN_Original_Regression_DropletDiameter,
    'XGBoost': model_XGBOOST_Original_Regression_DropletDiameter,
    'LightGBM': model_LightGBM_Original_Regression_DropletDiameter,
    'SVM': model_SVM_Original_Regression_DropletDiameter

}

#Select model
models = model_DropletGenerationRate

# Define model data sources (based on index ranges)
data_paths = [
    (range(0, 4), path_X_LLAVA, path_Y_LLAVA),
    (range(4, 8), path_X_DEEPSEEK, path_Y_DEEPSEEK),
    (range(8, 12), path_X_GEMMA, path_Y_GEMMA),
    (range(12, 16), path_X_LLAMA, path_Y_LLAMA),
    (range(16, 20), path_X_MISTRAL, path_Y_MISTRAL),
    (range(20, 24), path_X_Orginal, path_Y_Orginal),
]

def load_dataset(index):
    """Returns the correct dataset based on model index."""
    for index_range, x_path, y_path in data_paths:
        if index in index_range:
            X = pd.read_csv(x_path, index_col=0).values
            y = pd.read_csv(y_path, index_col=0).values
            return X, y
    raise ValueError(f"No dataset mapping found for model index: {index}")



def preprocess_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled



# Define metrics
metrics = {
    'MAE': [] , 'MSE': [], 'R²': [], 'RMSE': []
}
stop_early = EarlyStopping(monitor='val_loss',mode='min', patience=15)
EPOCH=1000
BATCH_SIZE = 512
VERBOSE=0

def perform_repeated_k_fold(n_splits=5, n_repeats=5):
    X1 = None
    y = None
    all_metrics_test_set = {model_name: {metric: [] for metric in metrics} for model_name in models}
    all_metrics_valid_set = {model_name: {metric: [] for metric in metrics} for model_name in models}

    for index, (model_name, model_func) in enumerate(models.items()):
        print("current model: ", model_name)
        fold_metrics = {metric: [] for metric in metrics}
        fold_metrics_valid = {metric: [] for metric in metrics}

        # Load and preprocess data
        X_raw, y_raw = load_dataset(index)
        y = y_raw[:, 0]  # # Select target: 0: generation rate; 1: diameter
        X = preprocess_features(X_raw)
        for repeat in range(n_repeats):

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
            for train_index, test_index in kf.split(X):

                # 80% train, 20% temp
                X_train, X_temp = X[train_index], X[test_index]
                y_train, y_temp = y[train_index], y[test_index]

                # 10% val, 10% test
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=repeat
                )

                if model_name in ['DNN-LLAVA','DNN-DEEPSEEK-R1', 'DNN-GEMMA2', 'DNN-LLAMA3.1', 'DNN-MISTRAL', 'DNN']:

                    model = clone_DNN(model_func)
                    model.set_weights(model_func.get_weights())
                    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
                    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCH * 10,
                              callbacks=[stop_early], shuffle=True, verbose=VERBOSE, batch_size=BATCH_SIZE)


                elif model_name in ['XGBoost', 'XGBoost-LLAVA', 'XGBoost-DEEPSEEK-R1', 'XGBoost-GEMMA2', 'XGBoost-LLAMA3.1', 'XGBoost-MISTRAL']:

                    model = clone_ML(model_func)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                elif model_name in ['LightGBM', 'LightGBM-LLAVA', 'LightGBM-DEEPSEEK-R1', 'LightGBM-GEMMA2', 'LightGBM-LLAMA3.1', 'LightGBM-MISTRAL']:

                    model = clone_ML(model_func)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

                elif model_name in ['SVM', 'SVM-LLAVA', 'SVM-DEEPSEEK-R1', 'SVM-GEMMA2', 'SVM-LLAMA3.1', 'SVM-MISTRAL']:

                    model = clone_ML(model_func)
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                y_pred_valid = model.predict(X_val)

                fold_metrics['MAE'].append(mean_absolute_error(y_test, y_pred))
                fold_metrics['MSE'].append(mean_squared_error(y_test, y_pred))
                fold_metrics['R²'].append(r2_score(y_test, y_pred))
                fold_metrics['RMSE'].append(root_mean_squared_error(y_test, y_pred))

                fold_metrics_valid['MAE'].append(mean_absolute_error(y_val, y_pred_valid))
                fold_metrics_valid['MSE'].append(mean_squared_error(y_val, y_pred_valid))
                fold_metrics_valid['R²'].append(r2_score(y_val, y_pred_valid))
                fold_metrics_valid['RMSE'].append(root_mean_squared_error(y_val, y_pred_valid))

            for metric in metrics:
                all_metrics_test_set[model_name][metric].append(np.array(fold_metrics[metric]))
                all_metrics_valid_set[model_name][metric].append(np.array(fold_metrics_valid[metric]))

    return all_metrics_test_set, all_metrics_valid_set
# Calculate statistics
def calculate_statistics(metrics_array):

    means_list=[]
    medians_list = []
    std_devs_list = []
    std_errors_list = []
    for array in metrics_array:
        means = np.round(np.mean(array), 4)
        medians = np.round(np.median(array), 4)
        std_devs = np.round(np.std(array), 4)
        std_errors = np.round(std_devs / np.sqrt(len(array)), 4)
        means_list.append(means)
        medians_list.append(medians)
        std_devs_list.append(std_devs)
        std_errors_list.append(std_errors)

    return means_list, medians_list, std_devs_list, std_errors_list



#Save metrics values to a table and save
def save_summary_table(metrics, all_metrics, output_folder):
    summary_list = []
    for model_name, metrics_dict in all_metrics.items():
        for metric in metrics.keys():
            for repeat in range(n_repeats):
                arr = metrics_dict[metric][repeat]
                mean = np.round(np.mean(arr), 4)
                median = np.round(np.median(arr), 4)
                std_dev = np.round(np.std(arr), 4)
                std_error = np.round(std_dev / np.sqrt(len(arr)), 4)
                abs_diff = abs((mean - median))
                summary_list.append([repeat+1, model_name, metric, mean, median , std_dev, std_error, abs_diff])

    summary_df = pd.DataFrame(summary_list,
                              columns=['Repeat', 'Model', 'Metric', 'Mean', 'Median', 'Standard Deviation',
                                       'Standard Error', 'AbsDiff'])
    min_diff_summary_df = summary_df.loc[summary_df.groupby(['Model', 'Metric'])['AbsDiff'].idxmin()]
    min_diff_summary_df.to_csv(os.path.join(output_folder, 'min_diff_summary_table.csv'), index=False)
    summary_df.to_csv(os.path.join(output_folder, 'summary_table.csv'), index=False)


# Define repeated k-fold parameters
n_splits = 10
n_repeats = 15
all_metrics_test_set, all_metrics_valid_set = perform_repeated_k_fold(n_splits=n_splits, n_repeats=n_repeats)

output_folder_test_set = ".../Test_Set/DropletGenerationRate"
os.makedirs(output_folder_test_set, exist_ok=True)

output_folder_valid_set = ".../Valid_Set/DropletGenerationRate"
os.makedirs(output_folder_valid_set, exist_ok=True)

# Save all_metrics to a file in the predefined folder
import pickle
file_path = os.path.join(output_folder_test_set, 'all_metrics_test_set.pkl')
with open(file_path, 'wb') as file:
    pickle.dump(all_metrics_test_set, file)

file_path = os.path.join(output_folder_valid_set, 'all_metrics_valid_set.pkl')
with open(file_path, 'wb') as file:
    pickle.dump(output_folder_valid_set, file)

save_summary_table(metrics, all_metrics_test_set, output_folder_test_set)
save_summary_table(metrics, all_metrics_valid_set, output_folder_valid_set)
