from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, f1_score, recall_score, roc_auc_score, accuracy_score
from keras.models import Model, save_model, load_model
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import os
import joblib

from tensorflow.keras.models import  clone_model as clone_DNN
from sklearn.base import clone as clone_ML
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


csv_data_folder_2021 = ".../Forward/droplet_data_original_LLM_output/"
model_folder_DropletRegime_2021 = '.../Forward/DropletRegime/'
#data
csv_data_folder = csv_data_folder_2021
model_folder_DropletRegime = model_folder_DropletRegime_2021

csv_data_folder = csv_data_folder_2021


path_X_Orginal = csv_data_folder+"X_droplet_imbalance_Original_data_Forward.csv"
path_Y_Orginal = csv_data_folder+"Y_droplet_imbalance_Original_data_Forward.csv"

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



#LLAVA
fileName_DNN_DropletRegime_LLAVA = model_folder_DropletRegime_2021 + 'DNN_LLAVA_DropletRegime_2021_Forward_Embedding.keras'
fileName_XGBOOST_DropletRegime_LLAVA = model_folder_DropletRegime_2021 + 'XGBOOST_LLAVA_DropletRegime_2021_Forward_Embedding.pkl'
fileName_SVM_DropletRegime_LLAVA = model_folder_DropletRegime_2021 + 'SVM_LLAVA_DropletRegime_2021_Forward_Embedding.pkl'
fileName_LightGBM_DropletRegime_LLAVA = model_folder_DropletRegime_2021 + 'LightGBM_LLAVA_DropletRegime_2021_Forward_Embedding.pkl'

model_DNN_LLAVA_Classification_DropletRegime = load_model(fileName_DNN_DropletRegime_LLAVA)
model_XGBOOST_LLAVA_Classification_DropletRegime = joblib.load(fileName_XGBOOST_DropletRegime_LLAVA)
model_SVM_LLAVA_Classification_DropletRegime = joblib.load(fileName_SVM_DropletRegime_LLAVA)
model_LightGBM_LLAVA_Classification_DropletRegime = joblib.load(fileName_LightGBM_DropletRegime_LLAVA)

#DEEPSEEK
fileName_DNN_DropletRegime_DEEPSEEK = model_folder_DropletRegime_2021 + 'DNN_DEEPSEEK_DropletRegime_2021_Forward_Embedding.keras'
fileName_XGBOOST_DropletRegime_DEEPSEEK = model_folder_DropletRegime_2021 + 'XGBOOST_DEEPSEEK_DropletRegime_2021_Forward_Embedding.pkl'
fileName_SVM_DropletRegime_DEEPSEEK = model_folder_DropletRegime_2021 + 'SVM_DEEPSEEK_DropletRegime_2021_Forward_Embedding.pkl'
fileName_LightGBM_DropletRegime_DEEPSEEK = model_folder_DropletRegime_2021 + 'LightGBM_DEEPSEEK_DropletRegime_2021_Forward_Embedding.pkl'

model_DNN_DEEPSEEK_Classification_DropletRegime = load_model(fileName_DNN_DropletRegime_DEEPSEEK)
model_XGBOOST_DEEPSEEK_Classification_DropletRegime = joblib.load(fileName_XGBOOST_DropletRegime_DEEPSEEK)
model_SVM_DEEPSEEK_Classification_DropletRegime = joblib.load(fileName_SVM_DropletRegime_DEEPSEEK)
model_LightGBM_DEEPSEEK_Classification_DropletRegime = joblib.load(fileName_LightGBM_DropletRegime_DEEPSEEK)

#GEMMA
fileName_DNN_DropletRegime_GEMMA = model_folder_DropletRegime_2021 + 'DNN_GEMMA_DropletRegime_2021_Forward_Embedding.keras'
fileName_XGBOOST_DropletRegime_GEMMA = model_folder_DropletRegime_2021 + 'XGBOOST_GEMMA_DropletRegime_2021_Forward_Embedding.pkl'
fileName_SVM_DropletRegime_GEMMA = model_folder_DropletRegime_2021 + 'SVM_GEMMA_DropletRegime_2021_Forward_Embedding.pkl'
fileName_LightGBM_DropletRegime_GEMMA = model_folder_DropletRegime_2021 + 'LightGBM_GEMMA_DropletRegime_2021_Forward_Embedding.pkl'

model_DNN_GEMMA_Classification_DropletRegime = load_model(fileName_DNN_DropletRegime_GEMMA)
model_XGBOOST_GEMMA_Classification_DropletRegime = joblib.load(fileName_XGBOOST_DropletRegime_GEMMA)
model_SVM_GEMMA_Classification_DropletRegime = joblib.load(fileName_SVM_DropletRegime_GEMMA)
model_LightGBM_GEMMA_Classification_DropletRegime = joblib.load(fileName_LightGBM_DropletRegime_GEMMA)

#LLAMA
fileName_DNN_DropletRegime_LLAMA = model_folder_DropletRegime_2021 + 'DNN_LLAMA_DropletRegime_2021_Forward_Embedding.keras'
fileName_XGBOOST_DropletRegime_LLAMA = model_folder_DropletRegime_2021 + 'XGBOOST_LLAMA_DropletRegime_2021_Forward_Embedding.pkl'
fileName_SVM_DropletRegime_LLAMA = model_folder_DropletRegime_2021 + 'SVM_LLAMA_DropletRegime_2021_Forward_Embedding.pkl'
fileName_LightGBM_DropletRegime_LLAMA = model_folder_DropletRegime_2021 + 'LightGBM_LLAMA_DropletRegime_2021_Forward_Embedding.pkl'

model_DNN_LLAMA_Classification_DropletRegime = load_model(fileName_DNN_DropletRegime_LLAMA)
model_XGBOOST_LLAMA_Classification_DropletRegime = joblib.load(fileName_XGBOOST_DropletRegime_LLAMA)
model_SVM_LLAMA_Classification_DropletRegime = joblib.load(fileName_SVM_DropletRegime_LLAMA)
model_LightGBM_LLAMA_Classification_DropletRegime = joblib.load(fileName_LightGBM_DropletRegime_LLAMA)

#MISTRAL
fileName_DNN_DropletRegime_MISTRAL = model_folder_DropletRegime_2021 + 'DNN_MISTRAL_DropletRegime_2021_Forward_Embedding.keras'
fileName_XGBOOST_DropletRegime_MISTRAL = model_folder_DropletRegime_2021 + 'XGBOOST_MISTRAL_DropletRegime_2021_Forward_Embedding.pkl'
fileName_SVM_DropletRegime_MISTRAL = model_folder_DropletRegime_2021 + 'SVM_MISTRAL_DropletRegime_2021_Forward_Embedding.pkl'
fileName_LightGBM_DropletRegime_MISTRAL = model_folder_DropletRegime_2021 + 'LightGBM_MISTRAL_DropletRegime_2021_Forward_Embedding.pkl'

model_DNN_MISTRAL_Classification_DropletRegime = load_model(fileName_DNN_DropletRegime_MISTRAL)
model_XGBOOST_MISTRAL_Classification_DropletRegime = joblib.load(fileName_XGBOOST_DropletRegime_MISTRAL)
model_SVM_MISTRAL_Classification_DropletRegime = joblib.load(fileName_SVM_DropletRegime_MISTRAL)
model_LightGBM_MISTRAL_Classification_DropletRegime = joblib.load(fileName_LightGBM_DropletRegime_MISTRAL)

#Original
fileName_DNN_DropletRegime_Original = model_folder_DropletRegime_2021 +'DNN_DropletRegime_2021_Forward_RawFeature.keras'
fileName_XGBOOST_DropletRegime_Original = model_folder_DropletRegime_2021 +'XGBOOST_DropletRegime_2021_Forward_RawFeature.pkl'
fileName_SVM_DropletRegime_Original = model_folder_DropletRegime_2021 +'SVM_DropletRegime_2021_Forward_RawFeature.pkl'
fileName_LightGBM_DropletRegime_Original = model_folder_DropletRegime_2021 +'LightGBM_DropletRegime_2021_Forward_RawFeature.pkl'

model_DNN_Original_Classification_DropletRegime = load_model(fileName_DNN_DropletRegime_Original)
model_XGBOOST_Original_Classification_DropletRegime = joblib.load(fileName_XGBOOST_DropletRegime_Original)
model_SVM_Original_Classification_DropletRegime = joblib.load(fileName_SVM_DropletRegime_Original)
model_LightGBM_Original_Classification_DropletRegime = joblib.load(fileName_LightGBM_DropletRegime_Original)



models = {

    #LLAVA
    'DNN-LLAVA': model_DNN_LLAVA_Classification_DropletRegime,
    'XGBoost-LLAVA': model_XGBOOST_LLAVA_Classification_DropletRegime,
    'SVM-LLAVA': model_SVM_LLAVA_Classification_DropletRegime,
    'LightGBM-LLAVA': model_LightGBM_LLAVA_Classification_DropletRegime,

    #DEEPSEEK
    'DNN-DEEPSEEK-R1': model_DNN_DEEPSEEK_Classification_DropletRegime,
    'XGBoost-DEEPSEEK-R1': model_XGBOOST_DEEPSEEK_Classification_DropletRegime,
    'SVM-DEEPSEEK-R1': model_SVM_DEEPSEEK_Classification_DropletRegime,
    'LightGBM-DEEPSEEK-R1': model_LightGBM_DEEPSEEK_Classification_DropletRegime,

    #GEMMA2
    'DNN-GEMMA2': model_DNN_GEMMA_Classification_DropletRegime,
    'XGBoost-GEMMA2': model_XGBOOST_GEMMA_Classification_DropletRegime,
    'SVM-GEMMA2': model_SVM_GEMMA_Classification_DropletRegime,
    'LightGBM-GEMMA2': model_LightGBM_GEMMA_Classification_DropletRegime,

    # LLAMA
    'DNN-LLAMA3.1': model_DNN_LLAMA_Classification_DropletRegime,
    'XGBoost-LLAMA3.1': model_XGBOOST_LLAMA_Classification_DropletRegime,
    'SVM-LLAMA3.1': model_SVM_LLAMA_Classification_DropletRegime,
    'LightGBM-LLAMA3.1': model_LightGBM_LLAMA_Classification_DropletRegime,

    # MISTRAL
    'DNN-MISTRAL': model_DNN_MISTRAL_Classification_DropletRegime,
    'XGBoost-MISTRAL': model_XGBOOST_MISTRAL_Classification_DropletRegime,
    'SVM-MISTRAL': model_SVM_MISTRAL_Classification_DropletRegime,
    'LightGBM-MISTRAL': model_LightGBM_MISTRAL_Classification_DropletRegime,

    #Original
    'DNN': model_DNN_Original_Classification_DropletRegime,
    'XGBoost': model_XGBOOST_Original_Classification_DropletRegime,
    'SVM': model_SVM_Original_Classification_DropletRegime,
    'LightGBM': model_LightGBM_Original_Classification_DropletRegime

}


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





metrics = {
    'accuracy': [] , 'precision': [], 'f1_score': [], 'recall': [], 'roc_auc': []
}

stop_early = EarlyStopping(monitor='val_loss',mode='min', patience=15)
EPOCH=1000
BATCH_SIZE = 512
VERBOSE=0

def perform_repeated_k_fold(n_splits=5, n_repeats=5):

    # skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = None
    y = None
    all_metrics_test_set = {model_name: {metric: [] for metric in metrics} for model_name in models}
    all_metrics_valid_set = {model_name: {metric: [] for metric in metrics} for model_name in models}



    for index, (model_name, model_func) in enumerate(models.items()):
        print("current model: ", model_name)
        fold_metrics = {metric: [] for metric in metrics}
        fold_metrics_valid = {metric: [] for metric in metrics}
        # Load and preprocess data
        X_raw, y_raw = load_dataset(index)
        y = y_raw[:, 2]  #Select target: 0: generation rate; 1: diameter; 2: regime
        y = np.add(y, -1)
        X = preprocess_features(X_raw)

        for repeat in range(n_repeats):

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
            for train_index, test_index in skf.split(X, y):

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
                    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCH * 10,
                              callbacks=[stop_early], shuffle=True, verbose=VERBOSE, batch_size=BATCH_SIZE)
                    y_pred = np.argmax(model.predict(X_test), axis=1)
                    y_pred_valid = np.argmax(model.predict(X_val), axis=1)
                elif model_name in ['SVM', 'SVM-LLAVA', 'SVM-DEEPSEEK-R1', 'SVM-GEMMA2', 'SVM-LLAMA3.1', 'SVM-MISTRAL']:
                    model = clone_ML(model_func)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_valid = model.predict(X_val)
                elif model_name in ['XGBoost', 'XGBoost-LLAVA', 'XGBoost-DEEPSEEK-R1', 'XGBoost-GEMMA2', 'XGBoost-LLAMA3.1', 'XGBoost-MISTRAL']:
                    model = clone_ML(model_func)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                    y_pred = model.predict(X_test)
                    y_pred_valid = model.predict(X_val)
                elif model_name in ['LightGBM', 'LightGBM-LLAVA', 'LightGBM-DEEPSEEK-R1', 'LightGBM-GEMMA2', 'LightGBM-LLAMA3.1', 'LightGBM-MISTRAL']:
                    model = clone_ML(model_func)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                    y_pred = model.predict(X_test)
                    y_pred_valid = model.predict(X_val)

                fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                fold_metrics['precision'].append(precision_score(y_test, y_pred))
                fold_metrics['f1_score'].append(f1_score(y_test, y_pred))
                fold_metrics['recall'].append(recall_score(y_test, y_pred))
                fold_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))

                fold_metrics_valid['accuracy'].append(accuracy_score(y_val, y_pred_valid))
                fold_metrics_valid['precision'].append(precision_score(y_val, y_pred_valid))
                fold_metrics_valid['f1_score'].append(f1_score(y_val, y_pred_valid))
                fold_metrics_valid['recall'].append(recall_score(y_val, y_pred_valid))
                fold_metrics_valid['roc_auc'].append(roc_auc_score(y_val, y_pred_valid))





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
output_folder_test_set = ".../Test_Set/DropletRegime"
os.makedirs(output_folder_test_set, exist_ok=True)

output_folder_valid_set = ".../Valid_Set/DropletRegime"
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





