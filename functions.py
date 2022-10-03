import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sn
import sklearn,plotly,scipy
from scipy.signal import savgol_filter
from plotly.subplots import make_subplots
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score as cv_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
target_names = ['ACE lung', 'ACE seminal', 'ACE heart']
colorss = ['blue', 'magenta', 'orange']


def plot_confusion_matrix(y_test, y_pred):
    """
    This function plots the confusion matrix based on inputs
    """
    z = confusion_matrix(list(y_test[:, 0]), list(y_pred))
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
            df = (np.array(k[k.columns[:-2]])[:, cut_range_from:])
        else:
            df = np.array(k[k.columns[:-2]])
        if Smooth:
            df = savgol_filter(df, 11, 2)
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
        df['classes'] = list(k.classes)
        df['conc'] = list(k.conc)
        df.conc = [i.replace('apf_sem', 'Seminal fluid ACE') for i in df.conc]
        df.conc = [i.replace('apf_serd', 'Heart ACE') for i in df.conc]
        df.conc = [i.replace('apf_legoch', 'Lung ACE') for i in df.conc]
        all_data.append(df)

    return all_data


def add_noise(X, y):
    """
    Augments data twice in size with a noise samples

    """

    noise = np.random.uniform(low=0.1,
                              high=0.5,
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
    temp['classes'] = list(y_test[:, 0])
    temp['lde'] = y_pred
    print(
        classification_report(
            temp.classes,
            temp.lde,
            target_names=target_names))


def plot_2d_lda_space(model, X, y):
    """
    Plot LDA 2D subspace
    """
    y_pred = model.predict(X)
    Xlda = model.transform((X))
    dataaa = pd.DataFrame(Xlda, columns=['LD1', 'LD2'])
    dataaa['classes'] = y_pred
    dataaa['col'] = list(y)

    fig = px.scatter(dataaa, x='LD1', y='LD2', color=[x[:-16] for x in y[:, 1]],
                     labels={
                         "LD1": "LD1",
                         "LD2": "LD2",
                         "color": ""
    })

    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,255)',
        width=1200,
        height=900,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01),
        font=dict(
            size=25))
    fig.update_traces(marker=dict(size=25))
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True)
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True)

    fig.update_layout()
#     fig.show()
    return fig

def train_model(X_train, y_train, model, show_cv_score=True):
    """
    This function trains model and returns cross validation score
    """
    scores = cv_score(model, (X_train), list(y_train[:, 0]), cv=3)
    if show_cv_score:
        print(
            "Model trained with accuracy on CV: %0.4f (+/- %0.4f)" %
            (scores.mean(), scores.std() * 2))
    model.fit((X_train), list(y_train[:, 0]))


def get_mean_weights(model, X, y, iterations):
    """
    This function returns classification weights averaged over dataset splits
    """
    weis = np.zeros((3, X.shape[1]))
    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4)
        X_train, y_train = add_noise(X_train, y_train)
        train_model(X_train, y_train, model, False)
        weis[0, :] += model.coef_[0]
        weis[1, :] += model.coef_[1]
        weis[2, :] += model.coef_[2]

    return weis / iterations


def select_weights(weis):
    """
    Selects weights based on following conditions
    """
    target_weights = [
        'apf_legoch_weights',
        'apf_sem_weights',
        'apf_serd_weights']
    low = 50
    high = 750
    y2 = savgol_filter([(feat_value / np.max(weis)) if ((feat_value > 0) & (feat_value > np.quantile(np.sort(weis[1, :]), 0.95))
                       & (feat_num > low) & (feat_num < high)) else 0 for feat_num, feat_value in enumerate(weis[1, :])], 11, 3)

    y1 = savgol_filter([(feat_value / np.max(weis)) if ((feat_value > 0) & (feat_value > np.quantile(np.sort(weis[0, :]), 0.95))
                                                        & (feat_num > low) & (feat_num < high)) else 0
                        for feat_num, feat_value in enumerate(weis[0, :])], 11, 3)

    y3 = savgol_filter([(feat_value / np.max(weis)) if ((feat_value > 0) & (feat_value > np.quantile(np.sort(weis[2, :]), 0.95))
                                                        & (feat_num > low) & (feat_num < high)) else 0
                        for feat_num, feat_value in enumerate(weis[2, :])], 11, 3)

    y1 = [i if i > 0 else 0 for i in y1]
    y2 = [i if i > 0 else 0 for i in y2]
    y3 = [i if i > 0 else 0 for i in y3]
    return y1, y2, y3



def select_features_for_model(y1, y2, y3, x_axis):
    """
    Selecting weights from our defined intervals
    """
    cond2 = (
        (x_axis > 433) & (
            x_axis < 460)) | (
        (x_axis > 868) & (
            x_axis < 883))
    cond3 = (
        (x_axis > 944) & (
            x_axis < 958)) | (
        (x_axis > 1235) & (
            x_axis < 1252))
    cond1 = (
        (x_axis > 532) & (
            x_axis < 545)) | (
        (x_axis > 1077) & (
            x_axis < 1092))

    imps = ((np.array(y2)[((np.where(cond2))[0])[np.argsort(np.array(y2)[((np.where(cond2))[0])])][::-1]],
             np.array(y3)[((np.where(cond3))[0])[np.argsort(np.array(y3)[((np.where(cond3))[0])])][::-1]],
             np.array(y1)[((np.where(cond1))[0])[np.argsort(np.array(y1)[((np.where(cond1))[0])])][::-1]][:5]))

    arrr1 = np.array(y2)[((np.where(cond2))[0])[np.argsort(
        np.array(y2)[((np.where(cond2))[0])])][::-1]]
    arrr2 = np.concatenate((np.array(y2)[((np.where(cond2))[0])[np.argsort(np.array(y2)[((np.where(cond2))[0])])][::-1]],
                            np.array(y3)[((np.where(cond3))[0])[np.argsort(np.array(y3)[((np.where(cond3))[0])])][::-1]]))

    features = np.concatenate((
        ((np.where(cond2))[0])[np.argsort(np.array(y2)[((np.where(cond2))[0])])][::-1],
        ((np.where(cond3))[0])[np.argsort(np.array(y3)[((np.where(cond3))[0])])][::-1],
        ((np.where(cond1))[0])[np.argsort(np.array(y1)[((np.where(cond1))[0])])][::-1]))
    return features, arrr1, arrr2


def get_accuracy_vs_num_features(data, weights, x_axis):
    """
    Obtaining accuracy for plotting
    """
    y1, y2, y3 = weights
    features, plot_helper1, plot_helper2 = select_features_for_model(
        y1, y2, y3, x_axis)
    accs = []
    sub_plots_imp = []
    sub_plots = []
    
    for feature in range(3, len(features)):
        
        if feature % 25 == 0:
                sub_plots.append(np.array(x_axis[features[:feature]]))

        if feature < len(plot_helper1) & feature % 25 == 0:
                sub_plots_imp.append(np.array(y2)[features[:feature]])
        if feature < len(plot_helper2) and feature > len(
                    plot_helper1) & feature % 25 == 0:
                sub_plots_imp.append(np.array(y3)[features[:feature]])
        if feature > len(plot_helper2) & feature % 25 == 0:
                sub_plots_imp.append(np.array(y1)[features[:feature]])
        cv_accs = []
        for iter in range(100):
            X_train1, X_test1, y_train1, y_test = train_test_split(
                    np.array(data)[:, :-2], np.array(data)[:, -2:], test_size=0.35)
            X_train = X_train1[:, features[:feature]]
            X_test = X_test1[:, features[:feature]]

            
            X_train, y_train = add_noise(X_train, y_train1)
            lda = LDA(n_components=2)
            lda.fit((X_train), list(y_train[:, 0]))
            y_pred = lda.predict(X_test)
            temp = pd.DataFrame()
            temp['classes'] = list(y_test[:, 0])
            temp['lde'] = y_pred
            cv_accs.append(accuracy_score(temp.classes, temp.lde))       
        accs.append([np.mean(cv_accs), np.std(cv_accs)])
#         accs.append()
    return accs, sub_plots, features


def plot_accs(x_axis, data, weights):
    """
    Plot accuracy vs number of features used
    """
    accs, sub_plots, features = get_accuracy_vs_num_features(
        data, weights, x_axis)
    accs = np.array(accs)
    levels = [0, 0.7, 1.4]
    target_names = ['Lung ACE', 'Seminal fluid ACE', 'Heart ACE']
    fig = make_subplots(rows=3, cols=1, specs=[[{}], [{}], [{}]], subplot_titles=(
        "(a) First 25 selected features", "(b) First 50 selected features", "(c) Result"))

    [fig.add_trace(go.Scatter(x=np.array(x_axis),
                              y=levels[i] + np.mean(np.array(data[data.classes == i])[:, :-2],
                                                    axis=0),
                              marker=dict(color=colorss[i]),
                              name=target_names[i]),
                   row=1,
                   col=1) for i in range(3)]

    for j in sub_plots[0]:
        fig.add_vline(
            x=j,
            line_dash="solid",
            row=1,
            col=1,
            line_color='green',
            opacity=0.15,
        )
    [fig.add_trace(go.Scatter(x=np.array(x_axis),
                              y=levels[i] + np.mean(np.array(data[data.classes == i])[:, :-2],
                                                    axis=0),
                              marker=dict(color=colorss[i]),
                              showlegend=False),
                   row=2,
                   col=1) for i in range(3)]

    for j in sub_plots[1]:
        fig.add_vline(
            x=j,
            line_dash="solid",
            row=2,
            col=1,
            line_color='green',
            opacity=0.15,
        )
    fig.add_trace(
        go.Scatter(
            x=np.arange(
                3,
                len(features)),
            y=accs[:,0],
            marker=dict(
                color='purple'),
            showlegend=False),
        row=3,
        col=1)
    
    fig.add_trace(
      go.Scatter(x = np.arange(
                3,
                len(features)),
        
          y=accs[:,0]+2*(accs[:,1]),
          mode='lines',
          line=dict(width=0),line_color='darkturquoise', showlegend=False
      ),
        row=3,
        col=1)
      
    fig.add_trace(go.Scatter(
          name='2σ',x =np.arange(
                3,
                len(features)),
      
          y=accs[:,0]-2*(accs[:,1]),
          line=dict(width=0),
          mode='lines',
          fill='tonexty',line_color='darkturquoise',
#           showlegend=False 
        ),
        row=3,
        col=1
      )

    
    ylevl = 2
    fig.update_layout(showlegend=True, legend=dict(
        orientation="h",
        yanchor="top", xanchor='center',
        y=1.2,
        x=0.5
    ), font=dict(
        size=20
    ), plot_bgcolor='rgba(255,255,255,255)')
    fig.add_annotation(x=444, y=ylevl,
                       text="<b>I<b>",
                       showarrow=False, row=1, col=1)
    fig.add_annotation(x=872, y=ylevl,
                       text="<b>II<b>",
                       showarrow=False, row=1, col=1)
    fig.add_annotation(x=1240, y=ylevl,
                       text="<b>III<b>",
                       showarrow=False, row=1, col=1)
    fig.add_annotation(x=444, y=ylevl,
                       text="<b>I<b>",
                       showarrow=False, row=2, col=1)
    fig.add_annotation(x=872, y=ylevl,
                       text="<b>II<b>",
                       showarrow=False, row=2, col=1)
    fig.add_annotation(x=1240, y=ylevl,
                       text="<b>III<b>",
                       showarrow=False, row=2, col=1)
    fig.add_annotation(x=537, y=ylevl,
                       text="<b>IV<b>",
                       showarrow=False, row=2, col=1)
    fig.add_annotation(x=950, y=ylevl,
                       text="<b>V<b>",
                       showarrow=False, row=2, col=1)
    fig.add_annotation(x=1084, y=ylevl,
                       text="<b>VI<b>",
                       showarrow=False, row=2, col=1)
    fig.update_xaxes(
        row=1,
        col=1,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True)
    fig.update_xaxes(
        row=2,
        col=1,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True)

    fig.update_yaxes(
        row=1,
        col=1,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        showticklabels=False)
    fig.update_yaxes(
        row=2,
        col=1,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        showticklabels=False)
    fig.update_yaxes(
        title_text="Accuracy",
        row=3,
        col=1,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True)
    fig.update_xaxes(
        title_text="Number of features used (bands)",
        row=3,
        col=1,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True)
#     fig.show()
    return fig


def save_weights(y1, y2, y3):
    """
    Saving weights for future uses
    """
    np.save('y1', y1)
    np.save('y2', y2)
    np.save('y3', y3)

import plotly.io as pio
#save a figure of 300dpi, with 1.5 inches, and  height 0.75inches

def plot_weights(y1, y2, y3, x_axis, data):
    """
    Plot weights with spectra averages
    """
    fig = go.Figure()
    barncolorss = ['coral', 'cyan', 'cornflowerblue']
    all_levels = [0, 0.5, 1]
    barnames = [
        'Вклады группы легочного АПФ',
        'Вклады группы семенного АПФ',
        'Вклады группы сердечного АПФ']
    target_names = ['Легочный', 'Семенной', 'Сердечный']

    for i, y in enumerate([y1, y2, y3]):

        fig.add_trace(go.Bar(x=x_axis, y=y, marker=dict(color=barncolorss[i]),
                             width=2.5, name=barnames[i], base=all_levels[i]))
        fig.add_trace(go.Scatter(x=x_axis,
                                 y=all_levels[i] + np.mean(np.array(data[data.classes == i])[:,:-2],
                                                           axis=0),
                                 marker=dict(color=colorss[i]),
                                 mode='lines',
                                 name=target_names[i]))
    fig.update_layout(
        width=2000,
        height=1500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01),
        font=dict(
            size=40),
        plot_bgcolor='rgba(255,255,255,255)')
    fig.update_yaxes(
        title='<b>I<b>',
        title_font=dict(
            size=40),
        tick0=0,
        dtick=1,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True)
    fig.update_xaxes(
        title='<b> Raman shift cm-1 <b>',
        title_font=dict(
            size=40),
        dtick=100,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True)
#     fig.show()
    return fig
#     pio.write_image(fig, "weights.JPEG", width=1.5*300, height=0.75*300 ,scale=1)

def plot_intervals(x_axis, weights, data):
    """
    Plot feature importance ranges
    """
    y1, y2, y3 = weights
    classes = [1, 0, 2]
    ranges = [[430, 463], [865, 885], [1075, 1094],
              [530, 547], [941, 965], [1231, 1258]]
    yranges = [[-0.06, 0.6], [-0.06, 0.6], [-0.06, 0.6],
               [-0.06, 0.6], [-0.06, 0.8], [-0.06, 1]]
    barnames = [
        '<b>Seminal fluid ACE            <b>',
        '<b>Lung ACE              <b>',
        '<b>Heart ACE         <b>']
    link_names = ['Sceletal def.', '', '', 'COO‾ def', 'C-C str.', 'CH₂ wag.']
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "<b>Interval I<b>",
            "<b>Interval III<b>",
            "<b>Interval V<b>",
            '<b>Interval II<b>',
            '<b>Interval IV<b>',
            '<b>Interval VI<b>'))
    for i, y in enumerate([y2, y1, y3]):
        for j in range(2):
            fig.add_trace(go.Bar(x=x_axis,
                                 y=y,
                                 width=2.5,
                                 legendgroup=str(i),
                                 legendgrouptitle_text=barnames[i],
                                 name=link_names[2 * i + j]),
                          row=j + 1,
                          col=i + 1)

            klss = np.array(data[data.classes == classes[i]])[:, :-2]
            fig.add_trace(
                go.Scatter(
                    x=np.array(x_axis),
                    y=np.mean(
                        klss,
                        axis=0),
                    marker=dict(
                        color=colorss[i]),
                    showlegend=False),
                row=j + 1,
                col=i + 1)
            fig.add_trace(
                go.Scatter(
                    x=np.array(x_axis),
                    y=np.mean(
                        klss,
                        axis=0) +
                    2 *
                    np.std(
                        klss.astype(float),
                        axis=0),
                    mode='lines',
                    line=dict(
                        width=0),
                    showlegend=False),
                row=j +
                1,
                col=i +
                1)

            fig.add_trace(
                go.Scatter(
                    x=np.array(x_axis),
                    y=np.mean(
                        klss,
                        axis=0) -
                    2 *
                    np.std(
                        klss.astype(float),
                        axis=0),
                    line=dict(
                        width=0),
                    mode='lines',
                    fill='tonexty',
                    showlegend=False),
                row=j +
                1,
                col=i +
                1)

            fig.update_xaxes(tickfont=dict(size=14),
                             tickangle=45,
                             range=ranges[2 * i + j],
                             dtick=5,
                             row=j + 1,
                             col=i + 1,
                             showline=True,
                             linewidth=1,
                             linecolor='black',
                             mirror=True)
            fig.update_yaxes(tickfont=dict(size=14),
                             range=yranges[2 * i + j],
                             row=j + 1,
                             col=i + 1,
                             showline=True,
                             linewidth=1,
                             linecolor='black',
                             mirror=True)

    fig.update_layout(
        width=1000,
        height=750,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            xanchor='center',
            y=-0.4,
            x=0.5),
        font=dict(
            size=20),
        plot_bgcolor='rgba(255,255,255,255)')
#     fig.show()
    return fig