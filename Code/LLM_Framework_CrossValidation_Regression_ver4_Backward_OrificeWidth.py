import tensorflow as tf
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import kstest
from matplotlib.pyplot import figure
from tensorflow import keras
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, SpatialDropout1D, Activation, Concatenate
from keras.layers import ReLU, PReLU, LeakyReLU, ELU
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, save_model, load_model
from tensorflow.keras.utils import plot_model
from keras import backend as K
import keras_nlp
import keras_tuner as kt
from keras_tuner import HyperModel
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import pandas as pd
import joblib
from lightgbm import early_stopping
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

path_X_DistilBERT = "X_droplet_imbalance_data_DistilBERT_Backward.csv"
path_Y_DistilBERT = "Y_droplet_imbalance_data_DistilBERT_Backward.csv"

path_X_SentenceTransformer = "X_droplet_imbalance_data_SentenceTransformer_Backward.csv"
path_Y_SentenceTransformer = "Y_droplet_imbalance_data_SentenceTransformer_Backward.csv"

path_X_OpenGPT = "X_droplet_imbalance_data_OpenGPT-2_Backward.csv"
path_Y_OpenGPT = "Y_droplet_imbalance_data_OpenGPT-2_Backward.csv"

path_X_Orginal = "X_droplet_imbalance_Original_data_Backward.csv"
path_Y_Orginal = "Y_droplet_imbalance_Original_data_Backward.csv"




#DropletOrificeWidth
#DistilBERT
fileName_DNN_DropletOrificeWidth_DistilBERT = 'LLM_BayesianOptimization_DistilBERT_Regression_DropletOrificeWidth_Backward.keras'
fileName_XGBOOST_DropletOrificeWidth_DistilBERT = 'LLM_Framework_XGBOOST_Embedding_DistilBERT_Regression_DropletOrificeWidth_Backward.pkl'
fileName_LightGBM_DropletOrificeWidth_DistilBERT = 'LLM_Framework_LightGBM_Embedding_DistilBERT_Regression_DropletOrificeWidth_Backward.pkl'
fileName_SVM_DropletOrificeWidth_DistilBERT = 'LLM_Framework_SVM_Embedding_DistilBERT_Regression_DropletOrificeWidth_Backward.pkl'

model_DNN_DistilBERT_Regression_DropletOrificeWidth = load_model(fileName_DNN_DropletOrificeWidth_DistilBERT)
model_XGBOOST_DistilBERT_Regression_DropletOrificeWidth = joblib.load(fileName_XGBOOST_DropletOrificeWidth_DistilBERT)
model_LightGBM_DistilBERT_Regression_DropletOrificeWidth = joblib.load(fileName_LightGBM_DropletOrificeWidth_DistilBERT)
model_SVM_DistilBERT_Regression_DropletOrificeWidth = joblib.load(fileName_SVM_DropletOrificeWidth_DistilBERT)
#SentenceTransformer
fileName_DNN_DropletOrificeWidth_SentenceTransformer = 'LLM_BayesianOptimization_SentenceTransformer_Regression_DropletOrificeWidth_Backward.keras'
fileName_XGBOOST_DropletOrificeWidth_SentenceTransformer = 'LLM_Framework_XGBOOST_Embedding_SentenceTransformer_Regression_DropletOrificeWidth_Backward.pkl'
fileName_LightGBM_DropletOrificeWidth_SentenceTransformer = 'LLM_Framework_LightGBM_Embedding_SentenceTransformer_Regression_DropletOrificeWidth_Backward.pkl'
fileName_SVM_DropletOrificeWidth_SentenceTransformer = 'LLM_Framework_SVM_Embedding_SentenceTransformer_Regression_DropletOrificeWidth_Backward.pkl'

model_DNN_SentenceTransformer_Regression_DropletOrificeWidth = load_model(fileName_DNN_DropletOrificeWidth_SentenceTransformer)
model_XGBOOST_SentenceTransformer_Regression_DropletOrificeWidth = joblib.load(fileName_XGBOOST_DropletOrificeWidth_SentenceTransformer)
model_LightGBM_SentenceTransformer_Regression_DropletOrificeWidth = joblib.load(fileName_LightGBM_DropletOrificeWidth_SentenceTransformer)
model_SVM_SentenceTransformer_Regression_DropletOrificeWidth = joblib.load(fileName_SVM_DropletOrificeWidth_SentenceTransformer)
#OpenGPT
fileName_DNN_DropletOrificeWidth_OpenGPT = 'LLM_BayesianOptimization_OpenGPT-2_Regression_DropletOrificeWidth_Backward.keras'
fileName_XGBOOST_DropletOrificeWidth_OpenGPT = 'LLM_Framework_XGBOOST_Embedding_OpenGPT-2_Regression_DropletOrificeWidth_Backward.pkl'
fileName_LightGBM_DropletOrificeWidth_OpenGPT = 'LLM_Framework_LightGBM_Embedding_OpenGPT-2_Regression_DropletOrificeWidth_Backward.pkl'
fileName_SVM_DropletOrificeWidth_OpenGPT = 'LLM_Framework_SVM_Embedding_OpenGPT-2_Regression_DropletOrificeWidth_Backward.pkl'

model_DNN_OpenGPT_Regression_DropletOrificeWidth = load_model(fileName_DNN_DropletOrificeWidth_OpenGPT)
model_XGBOOST_OpenGPT_Regression_DropletOrificeWidth = joblib.load(fileName_XGBOOST_DropletOrificeWidth_OpenGPT)
model_LightGBM_OpenGPT_Regression_DropletOrificeWidth = joblib.load(fileName_LightGBM_DropletOrificeWidth_OpenGPT)
model_SVM_OpenGPT_Regression_DropletOrificeWidth = joblib.load(fileName_SVM_DropletOrificeWidth_OpenGPT)
#Original
fileName_DNN_DropletOrificeWidth_Original = 'LLM_BayesianOptimization_Original_Data_NO_Embedding_Regression_DropletOrificeWidth_Backward.keras'
fileName_XGBOOST_DropletOrificeWidth_Original = 'LLM_Framework_XGBOOST_Original_Data_NO_Embedding_Regression_DropletOrificeWidth_Backward.pkl'
fileName_LightGBM_DropletOrificeWidth_Original = 'LLM_Framework_LightGBM_Original_Data_NO_Embedding_Regression_DropletOrificeWidth_Backward.pkl'
fileName_SVM_DropletOrificeWidth_Original = 'LLM_Framework_SVM_Original_Data_NO_Embedding_Regression_DropletOrificeWidth_Backward.pkl'

model_DNN_Original_Regression_DropletOrificeWidth = load_model(fileName_DNN_DropletOrificeWidth_Original)
model_XGBOOST_Original_Regression_DropletOrificeWidth = joblib.load(fileName_XGBOOST_DropletOrificeWidth_Original)
model_LightGBM_Original_Regression_DropletOrificeWidth = joblib.load(fileName_LightGBM_DropletOrificeWidth_Original)
model_SVM_Original_Regression_DropletOrificeWidth = joblib.load(fileName_SVM_DropletOrificeWidth_Original)



model_DropletOrificeWidth={
    #DistilBERT
    'DNN-DistilBERT': model_DNN_DistilBERT_Regression_DropletOrificeWidth,
    'XGBoost-DistilBERT': model_XGBOOST_DistilBERT_Regression_DropletOrificeWidth,
    'LightGBM-DistilBERT': model_LightGBM_DistilBERT_Regression_DropletOrificeWidth,
    'SVM-DistilBERT':model_SVM_DistilBERT_Regression_DropletOrificeWidth,
    #SentenceTransformer
    'DNN-SentenceTransformer': model_DNN_SentenceTransformer_Regression_DropletOrificeWidth,
    'XGBoost-SentenceTransformer': model_XGBOOST_SentenceTransformer_Regression_DropletOrificeWidth,
    'LightGBM-SentenceTransformer': model_LightGBM_SentenceTransformer_Regression_DropletOrificeWidth,
    'SVM-SentenceTransformer': model_SVM_SentenceTransformer_Regression_DropletOrificeWidth,
    #OpenGPT
    'DNN-OpenGPT-2': model_DNN_OpenGPT_Regression_DropletOrificeWidth,
    'XGBoost-OpenGPT-2': model_XGBOOST_OpenGPT_Regression_DropletOrificeWidth,
    'LightGBM-OpenGPT-2': model_LightGBM_OpenGPT_Regression_DropletOrificeWidth,
    'SVM-OpenGPT-2': model_SVM_OpenGPT_Regression_DropletOrificeWidth,
    #Original
    'DNN': model_DNN_Original_Regression_DropletOrificeWidth,
    'XGBoost': model_XGBOOST_Original_Regression_DropletOrificeWidth,
    'LightGBM': model_LightGBM_Original_Regression_DropletOrificeWidth,
    'SVM': model_SVM_Original_Regression_DropletOrificeWidth

}


models = model_DropletOrificeWidth

# Define metrics
metrics = {
    'MAE': [] , 'MSE': [], 'R²': [], 'RMSE': []
}
stop_early = EarlyStopping(monitor='val_loss',mode='min', patience=80)
EPOCH=1000
BATCH_SIZE = 512
VERBOSE=0

def perform_repeated_k_fold(n_splits=5, n_repeats=5):
    X = None
    y = None
    all_metrics = {model_name: {metric: [] for metric in metrics} for model_name in models}

    for index, (model_name, model_func) in enumerate(models.items()):
        print("current model: ", model_name)
        fold_metrics = {metric: [] for metric in metrics}

        if index < 4:
            X = (pd.read_csv(path_X_DistilBERT, index_col=0)).values
            y = (pd.read_csv(path_Y_DistilBERT, index_col=0)).values
        elif 4 <= index < 8:
            X = (pd.read_csv(path_X_SentenceTransformer, index_col=0)).values
            y = (pd.read_csv(path_Y_SentenceTransformer, index_col=0)).values
        elif 8 <= index < 12:
            X = (pd.read_csv(path_X_OpenGPT, index_col=0)).values
            y = (pd.read_csv(path_Y_OpenGPT, index_col=0)).values
        elif 12 <= index < 16:
            X = (pd.read_csv(path_X_Orginal, index_col=0)).values
            y = (pd.read_csv(path_Y_Orginal, index_col=0)).values

        from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
        preprocessor_x = StandardScaler()  # n_quantiles=100, output_distribution='normal')
        X = preprocessor_x.fit_transform(X)
        y = y[:,0]  # Backward: Select target: 0:Orifice width, 1:Aspect ratio, 2: Expansion ratio; 3:Normalized orifice length;
        # 4:Normalized water inlet; 5:Normalized oil inlet; 6:Flow rate ratio; 7:Capillary number

        for repeat in range(n_repeats):

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=repeat)
            for train_index, test_index in kf.split(X):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = model_func
                if model_name in ['DNN', 'DNN-DistilBERT', 'DNN-SentenceTransformer', 'DNN-OpenGPT-2']:

                    model = model_func
                    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCH * 10,
                              callbacks=[stop_early], shuffle=True, verbose=VERBOSE, batch_size=BATCH_SIZE)

                elif model_name in ['SVM', 'SVM-DistilBERT', 'SVM-SentenceTransformer', 'SVM-OpenGPT-2']:

                    model = model_func
                    model.fit(X_train, y_train)

                elif model_name in ['XGBoost', 'XGBoost-DistilBERT', 'XGBoost-SentenceTransformer', 'XGBoost-OpenGPT-2']:

                    model = model_func
                    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

                elif model_name in ['LightGBM', 'LightGBM-DistilBERT', 'LightGBM-SentenceTransformer', 'LightGBM-OpenGPT-2']:

                    model = model_func
                    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

                y_pred = model.predict(X_test)
                fold_metrics['MAE'].append(mean_absolute_error(y_test, y_pred))
                fold_metrics['MSE'].append(mean_squared_error(y_test, y_pred))
                fold_metrics['R²'].append(r2_score(y_test, y_pred))
                fold_metrics['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred)))

            for metric in metrics:
                all_metrics[model_name][metric].append(np.array(fold_metrics[metric]))

    return all_metrics
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
all_metrics = perform_repeated_k_fold(n_splits=n_splits, n_repeats=n_repeats)
print(all_metrics)
output_folder = "LLM_Framework_Data/DistilBERT_SentenceTransformer_OpenGPT_Original_Backward/Regression_20_08/DropletOrificeWidth"
os.makedirs(output_folder, exist_ok=True)

# Save all_metrics to a file in the predefined folder
import pickle
file_path = os.path.join(output_folder, 'all_metrics.pkl')
with open(file_path, 'wb') as file:
    pickle.dump(all_metrics, file)

save_summary_table(metrics, all_metrics, output_folder)
