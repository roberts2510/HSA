import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sn
import sklearn,plotly,scipy
import warnings
from scipy.signal import savgol_filter
from plotly.subplots import make_subplots
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV,cross_val_score , cross_val_predict,train_test_split
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,roc_curve, auc,confusion_matrix,classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, Ridge, LogisticRegression, LinearRegression, RidgeCV
from tqdm import tqdm


colorss = ['blue', 'magenta', 'orange']
target_names = [ '0%GHSA', '3%GHSA','5%GHSA','7%GHSA','10%GHSA','13%GHSA','15%GHSA','18%GHSA','20%GHSA','23%GHSA','25%GHSA']

def plot_confusion_matrix(y_test, y_pred):
    """
    This function plots the confusion matrix based on inputs
    """
    z = confusion_matrix(list(y_test), list(y_pred))
    x = target_names
    array = z
    df_cm = pd.DataFrame(array, x, x)
    plt.figure(figsize=(20, 15))
    sn.set(font_scale=2.5)
    ax = sn.heatmap(df_cm, annot=True, cmap='viridis', annot_kws={"size": 25})
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.title('Confusion matrix for results')
    plt.show()


def snv(input_data):
    """
    Perform standart normal variate transformation by rows
    """
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        output_data[i, :] = (
            input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])
    return output_data


def rubberband_corr(z):
    """
    Perform baseline correction on matrix row (single spectra)
    """
    x = np.arange(0, z.shape[1], 1)
    n = len(x)
    Y = z
    [m, n] = np.shape(Y)
    for i in range(int(m)):
        y = Y[i, :]
        if sum(np.isnan(y)) > 0:
            continue
        o = []
        period = 20
        for i in range(int(n / period)):
            time = y[period * i:period * (i + 1)]
            a = (np.where(time == time.min())[0])
            o.append(((a) + period * i)[0])
        Y = np.vstack((Y, y - np.interp(x, x[o], y[o])))

    return Y[m:, :] * (Y[m:, :] > 0)


def preprocess_data(
        data,
        Smooth,
        Standartization,
        Baselin_corr,
        median_filter_parameter,
        first_diff,
        cut_range_from):
    """
    Data preprocessing
    """
    all_data = []
    for k in [data]:
        if cut_range_from:
            df = (np.array(k[k.columns[:-3]])[:, cut_range_from:])
        else:
            df = np.array(k[k.columns[:-3]])
        if Smooth:
            df = savgol_filter(df, 15, 3)
        if Standartization:
            df = snv(df)
        if Baselin_corr:
            df = rubberband_corr(df)
        if median_filter_parameter:

            new_X_train = np.zeros(
                (df.shape[0], df.shape[1] // median_filter_parameter))
            for j in range((df.shape[0])):
                for i in range((df.shape[1] // med)):
                    new_X_train[j, i] = (
                        np.median(df[j, med * i:med * i + med]))
            df = (new_X_train)
        if first_diff:
            df = (np.diff(df))
        df = pd.DataFrame(df)
        df['label'] = list(k.label)
        df['concentr'] = list(k.concentr)
        df['conc'] = list(k.conc)

        all_data.append(df)

    return all_data


def add_noise(X, y):
    """
    Augments data twice in size with a noise samples

    """

    noise = np.random.uniform(low=0.1,
                              high=0.15,
                              size=X.shape) * np.random.normal(0,
                                                               np.mean(np.std(X)),
                                                               X.shape)
    new_signal = X + noise
    X = pd.DataFrame(X)
    new_signal = pd.DataFrame(new_signal)
    X = pd.concat((X, new_signal))
    y = np.concatenate((y, y))
    return X, y


def classification_rep(y_test, y_pred):
    """
    Returns classification report
    """
    temp = pd.DataFrame()
    temp['label'] = (y_test)
    temp['lde'] = y_pred
    print(
        classification_report(
            temp.label,
            temp.lde,
            target_names=target_names))


