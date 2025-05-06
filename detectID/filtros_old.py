import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
#import plotly.express as px
from scipy.signal import butter, lfilter, stft
from hampel import hampel
from sklearn.decomposition import PCA
from scipy.fftpack import fft, fftfreq
#from ssqueezepy import ssq_cwt, ssq_stft, extract_ridges
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from sklearn.cluster import KMeans
from prettytable import PrettyTable
'''

from decoders import interleaved as decoder

'''
from keras.layers import Flatten
from keras.layers import Dropout

from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
'''
#from fastdtw import fastdtw
#from scipy.spatial.distance import euclidean

#from dtaidistance import dtw
#from dtaidistance import dtw_visualisation as dtwvis

def iq_samples_abs(series):
    abs_series = {}
    for key in series.keys():
        for i in range(len(series[key])):
            valor = np.abs(series[key][i])
            if valor == 0:
                valor = -1 * 100
            else:
                valor = 20 * np.log10(valor)
            if key in abs_series:
                abs_series[key] = np.append(abs_series[key], valor)
            else:
                abs_series[key] = np.array(valor)

    abs_series = pd.DataFrame(abs_series)

    return abs_series


def remove_avg(series):
    removed_avg = {}

    for key in series.keys():
        removed_avg[key] = series[key] - series[key].mean()

    removed_avg = pd.DataFrame(removed_avg)

    return removed_avg


def hampel_filter(series):
    filtered = {}
    for key in series.keys():
        filtered[key] = hampel(series[key], window_size=31, n=3, imputation=True)
    filtered = pd.DataFrame(filtered)

    return filtered


def moving_avg_filter(series):
    moving_avg = {}

    for key in series.keys():
        moving_avg[key] = series[key].rolling(window=10, min_periods=1, center=True).mean()

    moving_avg = pd.DataFrame(moving_avg)
    return moving_avg


def band_pass_filter(series):
    fs = 7.64
    t = 1.0 / fs
    lowcut = 0.2
    highcut = 0.4
    n = len(series)
    xf = np.linspace(0.0, 1.0 / (2.0 * t), n // 2)
    b, a = butter(5, [lowcut / (fs / 2), highcut / (fs / 2)], 'band', analog=False, output='ba')

    bandpass_samples_filter = {}
    for key in series.keys():
        bandpass_samples_filter[key] = lfilter(b, a, series[key])

    bandpass_samples_filter = pd.DataFrame(bandpass_samples_filter)

    return bandpass_samples_filter


def csi_pca(series):
    series = series.reset_index()
    for subcarrier in series.keys():
        for sample in range(len(series[subcarrier])):
            if np.isnan(series[subcarrier][sample]) or np.isinf(series[subcarrier][sample]):
                series[subcarrier][sample] = 0

    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(series)

    principal_components = pd.DataFrame(data=principal_components, columns=['PCA'])

    return principal_components

def pca():
    #df = px.data.iris()
    #features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]

    
    path = "/home/oem/Documentos/exe_remoto_presencia/scans_lleno_csv/"
    participantes = ['001','100']#['036','112']
    posiciones = [1,16]
    matriz_global = {}
    matriz_global = pd.DataFrame(matriz_global)
    clases = {}
    clases = pd.DataFrame(clases)

    for s in participantes: 
        for p in posiciones:
            
            df = pd.read_csv(path + s + "/" + str(p) + ".csv")
            df = df.drop(['Unnamed: 0'], axis=1)
            df = df.apply(lambda col: col.apply(lambda val: complex(val.strip('()'))))
            df = hampel_filter(df)
            series = moving_avg_filter(df)
            series = iq_samples_abs(series)

            series = pd.DataFrame(series)
            matriz_global = pd.concat([matriz_global, series], ignore_index = True)

            #print('\n participante: ', s, 'posicion: ', p, 'forma: ', matriz_global.shape, '\n')
            
            target = np.repeat("participante_N°: "+ s + " posição: " + str(p), 500)
            clases = pd.concat([clases, pd.DataFrame(target)], ignore_index = True)

            #print(clases)

    #print('\n', matriz_global)
    #print('\n', matriz_global.shape)
    #print('\n', clases)
    #print('\n', clases.shape)
    #df_merged = matriz_global.merge(clases)
    df_join = matriz_global.join(clases, how = "inner")
    #table_g= pd.concat([matriz_global], [clases],axis=1)
    print('\n', df_join)

    
    pca = PCA()
    components = pca.fit_transform(df_join.iloc[:, 0:234])
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(2),
        color=df_join.iloc[:,-1]
    )

    '''
    n_components = 2

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df_join.iloc[:, 0:234])

    total_var = pca.explained_variance_ratio_.sum() * 100

    labels = {str(i): f"PC {i+1}" for i in range(n_components)}
    labels['color'] = 'Median Price'

    fig = px.scatter_matrix(
        components,
        color=df_join.iloc[:,-1],
        dimensions=range(n_components),
        labels=labels,
        title=f'Total Explained Variance: {total_var:.2f}%',
    )
    '''
    fig.update_traces(diagonal_visible=False)
    fig.show()
    


def n_max(n, series):
    maximos = {}
    for i in range(n):
        maior = -1
        f = -1
        for i in range(len(series)//2):
            if not i in maximos and series[i] >= maior:
                maior = series[i]
                f = i
        maximos[f] = maior

    return maximos


def csi_fft(series):
    # FFT
    yf = fft(series)
    xf = fftfreq(yf.size, 0.13)

    n_maximos = n_max(4, np.abs(yf))

    print("dicionario de maximos:")
    print(n_maximos)

    freqs = []
    for i in list(n_maximos.keys()):
        freqs.append(xf[i])
    print(freqs)
    freq = np.mean(freqs)

    print("\n\n\n\n")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Frequencia: ", freq, " taxa de respiracao: ", 60 * freq)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("\n\n\n\n")

    # rr = np.zeros(len(xf))
    # rr[maximo] = np.max(np.abs(yf))
    fig, ax = plt.subplots()
    plt.plot(xf, np.abs(yf))
    # plt.stem(xf, rr, linefmt='red')
    ax.set(xlabel='Frequencies (Hz)', ylabel='dB', title='FFT')
    plt.xlim(0, 1)
    plt.show()


def csi_sftf(series):
    f, t, zxx = stft(series, 1 / 0.13)
    amp = np.max(np.abs(zxx))
    plt.pcolormesh(t, f, np.abs(zxx), vmin=0, vmax=amp, shading='gouraud')
    plt.title('STFT')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(0, 0.7)
    plt.xlabel('Time [sec]')
    plt.show()

def plot(series, title):
    f, ax = plt.subplots()
    plt.plot(series)
    ax.set(xlabel='Samples', ylabel='dB', title=title)
    plt.show()


# def viz(x, tx, wx):
#     plt.imshow(np.abs(wx), aspect='auto')
#     plt.show()
#     plt.imshow(np.abs(tx), aspect='auto')
#     plt.show()


def viz(x, tf, ridge_idxs, yticks=None, ssq=False, transform='cwt', show_x=False, scale=1):
    if show_x:
        f, ax = plt.subplots()
        plt.plot(x)
        ax.set(xlabel="Time [samples]", ylabel="Signal Amplitude [A.U.]", title="x(t)")
        plt.show()

    f, ax = plt.subplots()

    ax.set(ylabel="Frequency scales [1/Hz]", title="abs({}{}) w/ ridge_idxs")
    if scale == 1:
        plt.ylim(50, 150)
    else:
        plt.ylim(0, 100)
    plt.imshow(np.abs(tf))
    plt.plot(ridge_idxs, 'r')
    plt.show()

#def analyze(csi):
if __name__ == "__main__":

    #csv_to_matrizdtw_vazio(0, 1)
    #csv_to_matrizdtw_lleno(1, 1)

    #for ref_num in range(1, 6, 1):
        #csv_to_matrizdtw_vazio(0, ref_num)
    
    #for ref_num in range(4, 6, 1):
        #csv_to_matrizdtw_lleno(1, ref_num)

    #pca()
    
    ######################################################

    path = '/home/oem/Documentos/csi/1.pcap'
    
    '''
    csi = pd.read_csv(path)
    csi = csi.drop(['Unnamed: 0'], axis=1)
    csi = csi.apply(lambda col: col.apply(lambda val: complex(val.strip('()'))))
    '''
    samples = decoder.read_pcap(path)
    csi = samples.get_pd_csi()
    
    plot(csi, "CSI data")
    
    #series_abs = iq_samples_abs(csi) #show all subcarrier
    #plot(series_abs, "All subcarrier's magnitude")
    
    series = hampel_filter(csi)
    plot(series, "After Hampel Filter")

    series = moving_avg_filter(series)
    plot(series, "After Moving Average Filter")

    series = band_pass_filter(series)
    plot(series, "After Band Pass Filter")

    #ml_analisis(series) #ML aplication

    series = csi_pca(series) #PCA
    plot(series, "Principal Component")

    x = series['PCA'].to_numpy() #convert panda to numpy

    #csi_sftf(series['PCA']) # sftf implement

    csi_fft(x) #fft implement

    print("\nfinalizado con exito")
