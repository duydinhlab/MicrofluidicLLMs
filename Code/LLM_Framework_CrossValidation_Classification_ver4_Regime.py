from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, f1_score, recall_score, roc_auc_score, accuracy_score
from keras.models import Model, save_model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

path_X_DistilBERT = "X_droplet_imbalance_data_DistilBERT.csv"
path_Y_DistilBERT = "Y_droplet_imbalance_data_DistilBERT.csv"

path_X_SentenceTransformer = "X_droplet_imbalance_data_SentenceTransformer.csv"
path_Y_SentenceTransformer = "Y_droplet_imbalance_data_SentenceTransformer.csv"

path_X_OpenGPT = "X_droplet_imbalance_data_OpenGPT-2.csv"
path_Y_OpenGPT = "Y_droplet_imbalance_data_OpenGPT-2.csv"

path_X_Orginal = "X_droplet_imbalance_Original_data_Forward.csv"
path_Y_Orginal = "Y_droplet_imbalance_Original_data_Forward.csv"




#DistilBERT
fileName_DNN_DropletRegime_DistilBERT = 'LLM_BayesianOptimization_DistilBERT_Classification_DropletObserveRegime_16_07_2024.keras'
fileName_XGBOOST_DropletRegime_DistilBERT = 'LLM_Framework_XGBOOST_Embedding_DistilBERT_Classification_DropletRegime.pkl'
#fileName_RandomForest_DropletRegime_DistilBERT = 'LLM_Framework_RandomForest_Embedding_DistilBERT_Classification_DropletRegime_14_07_2024.pkl'
fileName_SVM_DropletRegime_DistilBERT = 'LLM_Framework_SVM_Embedding_DistilBERT_Classification_DropletRegime_14_07_2024.pkl'
fileName_LightGBM_DropletRegime_DistilBERT = 'LLM_Framework_LightGBM_Embedding_DistilBERT_Classification_DropletRegime_14_07_2024.pkl'

model_DNN_DistilBERT_Classification_DropletRegime = load_model(fileName_DNN_DropletRegime_DistilBERT)
model_XGBOOST_DistilBERT_Classification_DropletRegime = joblib.load(fileName_XGBOOST_DropletRegime_DistilBERT)
# model_RandomForest_Classification_DropletRegime = joblib.load(fileName_RandomForest_DropletRegime_DistilBERT)
model_SVM_DistilBERT_Classification_DropletRegime = joblib.load(fileName_SVM_DropletRegime_DistilBERT)
model_LightGBM_DistilBERT_Classification_DropletRegime = joblib.load(fileName_LightGBM_DropletRegime_DistilBERT)

#SentenceTransformer
fileName_DNN_DropletRegime_SentenceTransformer = 'LLM_BayesianOptimization_SentenceTransformer_Classification_DropletObserveRegime_22_07_2024.keras'
fileName_XGBOOST_DropletRegime_SentenceTransformer = 'LLM_Framework_XGBOOST_Embedding_SentenceTransformer_Classification_DropletRegime.pkl'
fileName_SVM_DropletRegime_SentenceTransformer = 'LLM_Framework_SVM_Embedding_SentenceTransformer_Classification_DropletRegime_22_07_2024.pkl'
fileName_LightGBM_DropletRegime_SentenceTransformer = 'LLM_Framework_LightGBM_Embedding_SentenceTransformer_Classification_DropletRegime_22_07_2024.pkl'

model_DNN_SentenceTransformer_Classification_DropletRegime = load_model(fileName_DNN_DropletRegime_SentenceTransformer)
model_XGBOOST_SentenceTransformer_Classification_DropletRegime = joblib.load(fileName_XGBOOST_DropletRegime_SentenceTransformer)
model_SVM_SentenceTransformer_Classification_DropletRegime = joblib.load(fileName_SVM_DropletRegime_SentenceTransformer)
model_LightGBM_SentenceTransformer_Classification_DropletRegime = joblib.load(fileName_LightGBM_DropletRegime_SentenceTransformer)

#OpenGPT
fileName_DNN_DropletRegime_OpenGPT = 'LLM_BayesianOptimization_OpenGPT-2_Classification_DropletObserveRegime_22_07_2024.keras'
fileName_XGBOOST_DropletRegime_OpenGPT = 'LLM_Framework_XGBOOST_Embedding_OpenGPT-2_Classification_DropletRegime.pkl'
fileName_SVM_DropletRegime_OpenGPT = 'LLM_Framework_SVM_Embedding_OpenGPT-2_Classification_DropletRegime.pkl'
fileName_LightGBM_DropletRegime_OpenGPT = 'LLM_Framework_LightGBM_Embedding_OpenGPT-2_Classification_DropletRegime.pkl'

model_DNN_OpenGPT_Classification_DropletRegime = load_model(fileName_DNN_DropletRegime_OpenGPT)
model_XGBOOST_OpenGPT_Classification_DropletRegime = joblib.load(fileName_XGBOOST_DropletRegime_OpenGPT)
model_SVM_OpenGPT_Classification_DropletRegime = joblib.load(fileName_SVM_DropletRegime_OpenGPT)
model_LightGBM_OpenGPT_Classification_DropletRegime = joblib.load(fileName_LightGBM_DropletRegime_OpenGPT)

#Original
fileName_DNN_DropletRegime_Original = 'LLM_BayesianOptimization_Original_Data_NO_Embedding_Classification_DropletObserveRegime_Forward.keras'
fileName_XGBOOST_DropletRegime_Original = 'LLM_Framework_XGBOOST_Original_Data_NO_Embedding_Classification_DropletRegime_Forward.pkl'
fileName_SVM_DropletRegime_Original = 'LLM_Framework_SVM_Original_Data_NO_Embedding_Classification_DropletRegime_Forward.pkl'
fileName_LightGBM_DropletRegime_Original = 'LLM_Framework_LightGBM_Original_Data_NO_Embedding_Classification_DropletRegime_Forward.pkl'

model_DNN_Original_Classification_DropletRegime = load_model(fileName_DNN_DropletRegime_Original)
model_XGBOOST_Original_Classification_DropletRegime = joblib.load(fileName_XGBOOST_DropletRegime_Original)
model_SVM_Original_Classification_DropletRegime = joblib.load(fileName_SVM_DropletRegime_Original)
model_LightGBM_Original_Classification_DropletRegime = joblib.load(fileName_LightGBM_DropletRegime_Original)



models = {

    #DistilBERT
    'DNN-DistilBERT': model_DNN_DistilBERT_Classification_DropletRegime,
    'XGBoost-DistilBERT': model_XGBOOST_DistilBERT_Classification_DropletRegime,
    'SVM-DistilBERT': model_SVM_DistilBERT_Classification_DropletRegime,
    'LightGBM-DistilBERT': model_LightGBM_DistilBERT_Classification_DropletRegime,

    #SentenceTransformer
    'DNN-SentenceTransformer': model_DNN_SentenceTransformer_Classification_DropletRegime,
    'XGBoost-SentenceTransformer': model_XGBOOST_SentenceTransformer_Classification_DropletRegime,
    'SVM-SentenceTransformer': model_SVM_SentenceTransformer_Classification_DropletRegime,
    'LightGBM-SentenceTransformer': model_LightGBM_SentenceTransformer_Classification_DropletRegime,

    #OpenGPT
    'DNN-OpenGPT-2': model_DNN_OpenGPT_Classification_DropletRegime,
    'XGBoost-OpenGPT-2': model_XGBOOST_OpenGPT_Classification_DropletRegime,
    'SVM-OpenGPT-2': model_SVM_OpenGPT_Classification_DropletRegime,
    'LightGBM-OpenGPT-2': model_LightGBM_OpenGPT_Classification_DropletRegime,

    #Original
    'DNN': model_DNN_Original_Classification_DropletRegime,
    'XGBoost': model_XGBOOST_Original_Classification_DropletRegime,
    'SVM': model_SVM_Original_Classification_DropletRegime,
    'LightGBM': model_LightGBM_Original_Classification_DropletRegime

}




metrics = {
    'accuracy': [] , 'precision': [], 'f1_score': [], 'recall': [], 'roc_auc': []
}

stop_early = EarlyStopping(monitor='val_loss',mode='min', patience=10)
EPOCH=1000
BATCH_SIZE = 512
VERBOSE=0

def perform_repeated_stratified_k_fold(n_splits=5, n_repeats=5):

    # skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
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

        from sklearn.preprocessing import StandardScaler
        preprocessor_x = StandardScaler()
        X = preprocessor_x.fit_transform(X)
        y = y[:,2] #Select target: 0: generation rate; 1: diameter; 2: regime
        y= np.add(y, -1)
        for repeat in range(n_repeats):

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
            for train_index, test_index in skf.split(X, y):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                if model_name in ['DNN', 'DNN-DistilBERT', 'DNN-SentenceTransformer', 'DNN-OpenGPT-2']:
                    model = model_func
                    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCH * 10,
                              callbacks=[stop_early], shuffle=True, verbose=VERBOSE, batch_size=BATCH_SIZE)
                    y_pred = np.argmax(model.predict(X_test), axis=1)
                elif model_name in ['SVM', 'SVM-DistilBERT', 'SVM-SentenceTransformer', 'SVM-OpenGPT-2']:
                    model = model_func
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                elif model_name in ['XGBoost', 'XGBoost-DistilBERT', 'XGBoost-SentenceTransformer', 'XGBoost-OpenGPT-2']:
                    model = model_func
                    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                    y_pred = model.predict(X_test)
                elif model_name in ['LightGBM', 'LightGBM-DistilBERT', 'LightGBM-SentenceTransformer', 'LightGBM-OpenGPT-2']:
                    model = model_func
                    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
                    y_pred = model.predict(X_test)

                fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                fold_metrics['precision'].append(precision_score(y_test, y_pred))
                fold_metrics['f1_score'].append(f1_score(y_test, y_pred))
                fold_metrics['recall'].append(recall_score(y_test, y_pred))
                fold_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))

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
all_metrics = perform_repeated_stratified_k_fold(n_splits=n_splits, n_repeats=n_repeats)
print(all_metrics)
output_folder = "LLM_Framework_Data/DistilBERT_SentenceTransformer_OpenGPT_Original/Classification_20_08_2024/DropletRegime"
os.makedirs(output_folder, exist_ok=True)

# Save all_metrics to a file in the predefined folder
import pickle
file_path = os.path.join(output_folder, 'all_metrics.pkl')
with open(file_path, 'wb') as file:
    pickle.dump(all_metrics, file)

save_summary_table(metrics, all_metrics, output_folder)




