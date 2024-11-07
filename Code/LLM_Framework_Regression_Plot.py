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



# Define metrics
metrics = {
    'MAE': [] , 'MSE': [], 'R²': [], 'RMSE': []
}





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

# Plot and save the metric comparison
FONTSIZE=32
from matplotlib.lines import Line2D
def plot_metric_comparison(models_metrics, metric_name, output_folder):
    repetitions = np.arange(1, n_repeats+1)
    # Generate distinct colors for each model
    colors = plt.cm.get_cmap('tab20', len(models_metrics))
    # Define different markers
    # markers = list(Line2D.markers.keys()) #
    markers = ['o', 's', 'D', "^", "v", '<', '>', 'p', '*', 'h', 'H', 'x', 'd', 'P', 'X']

    for metric_name in metrics.keys():
        plt.figure(figsize=(28, 16))
        buf_mean = []
        for (model_index ,(model_name, metrics_dict)) in enumerate(models_metrics.items()):
            means, medians, _, std_errors = calculate_statistics(metrics_dict[metric_name])
            buf_mean.append(means)
            color = colors(model_index)
            marker = markers[model_index % len(markers)]
            if "OpenGPT-2" in model_name:
                model_name = model_name.replace("OpenGPT-2", "GPT-2")

            plt.plot(repetitions, means, marker=marker, markersize=32, color=color,linestyle='-', label=f'{model_name}') 
            plt.fill_between(repetitions, np.array(means) - np.array(std_errors), np.array(means) + np.array(std_errors),
                             color=color , alpha=0.2)#label=f'{model_name} std_error {metric_name}'#
            # plt.plot(repetitions, medians, 's-', label=f'{model_name} median {metric_name}')

        plt.xlabel('Repetition', fontsize = FONTSIZE)
        plt.xticks(range(1, n_repeats + 1), fontsize = FONTSIZE)

        # Automatically determine number of ticks
        min_ticks = 5
        max_ticks = 10
        data_range = np.max(buf_mean) - np.min(buf_mean)
        # Determine number of ticks
        num_ticks = min(max_ticks, max(min_ticks, round(data_range / (data_range / (max_ticks - min_ticks)))))
        # Calculate interval based on determined number of ticks
        interval = data_range / num_ticks
        # Set y-ticks with the calculated interval
        plt.yticks(np.arange(np.min(buf_mean), np.max(buf_mean) + interval, interval), fontsize=FONTSIZE)


        #plt.yticks(np.arange(0, np.max(buf_mean), 50), fontsize=FONTSIZE)
        #plt.yticks( fontsize=FONTSIZE)

        plt.ylabel(metric_name, fontsize = FONTSIZE)
        plt.title(f'Comparison of {metric_name} across models', fontsize = FONTSIZE)
        plt.grid(True)

        # Position the legend outside the plot
        legend = plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4, fontsize = FONTSIZE) #ncol=len(models)
        # Adjust marker size in the legend
        # Adjust marker size in the legend
        for handle in legend.legend_handles:
            handle.set_markersize(32)
        plt.tight_layout()
        plot_path = os.path.join(output_folder, f'model_{metric_name}_comparison.pdf')
        plt.savefig(plot_path, bbox_inches='tight')
    plt.show()


# Run the evaluation, save the summaries, and plot the comparisons



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


import pickle
# Define repeated k-fold parameters
n_splits = 10
n_repeats = 15
# output_folder = "LLM_Framework_Data/DistilBERT_SentenceTransformer_OpenGPT_Original/Regression_20_08_2024/DropletGenerationRate"
output_folder = "LLM_Framework_Data/DistilBERT_SentenceTransformer_OpenGPT_Original/Regression_20_08_2024/DropletDiameter"
#output_folder = "LLM_Framework_Data/DistilBERT_SentenceTransformer_OpenGPT_Original_Backward/Regression_20_08/DropletCapillaryNumber"
# output_folder = "LLM_Framework_Data/DistilBERT_SentenceTransformer_OpenGPT_Original_Backward/Regression_20_08/DropletAspectRatio"
# output_folder = "LLM_Framework_Data/DistilBERT_SentenceTransformer_OpenGPT_Original_Backward/Regression_20_08/DropletExpansionRatio"
# output_folder = "LLM_Framework_Data/DistilBERT_SentenceTransformer_OpenGPT_Original_Backward/Regression_20_08/DropletFlowRateRatio"
# output_folder = "LLM_Framework_Data/DistilBERT_SentenceTransformer_OpenGPT_Original_Backward/Regression_20_08/DropletNormalizedOilInlet"
# output_folder = "LLM_Framework_Data/DistilBERT_SentenceTransformer_OpenGPT_Original_Backward/Regression_20_08/DropletNormalizedOrificeLength"
# output_folder = "LLM_Framework_Data/DistilBERT_SentenceTransformer_OpenGPT_Original_Backward/Regression_20_08/DropletNormalizedWaterInlet"
#output_folder = "LLM_Framework_Data/DistilBERT_SentenceTransformer_OpenGPT_Original_Backward/Regression_20_08/DropletOrificeWidth"
# Load all_metrics from a file in the predefined folder
file_path = os.path.join(output_folder, 'all_metrics.pkl')
with open(file_path, 'rb') as file:
    all_metrics = pickle.load(file)

plot_metric_comparison(all_metrics, metrics, output_folder)