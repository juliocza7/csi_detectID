import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from scipy.signal import butter, lfilter, stft
from hampel import hampel
from sklearn.decomposition import PCA
from scipy.fftpack import fft, fftfreq
from ssqueezepy import ssq_cwt, ssq_stft, extract_ridges
from sklearn.cluster import KMeans

from decoders import interleaved as decoder
from pandas import concat

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

#from dtaidistance import dtw
#from dtaidistance import dtw_visualisation as dtwvis

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 

import scipy.stats as st
import math

import random

from time import time

import itertools

import pickle
import os

import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

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

def iq_samples_angle(series):
    angle_series = {}
    for key in series.keys():
        for i in range(len(series[key])):
            valor = np.angle(series[key][i])

            if key in angle_series:
                angle_series[key] = np.append(angle_series[key], valor)
            else:
                angle_series[key] = np.array(valor)

    angle_series = pd.DataFrame(angle_series)

    return angle_series

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
    fs = 30.0 #7.64
    t = 1.0 / fs
    lowcut = 0.2
    highcut = 10.0 #0.4
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

##########################################################################################################

def random_forest_ml(train_norm, test_norm, train_labels, test_labels):
    print("\n--Random Forest--")
    #Create a Gaussian Classifier
    randomfc=RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    randomfc.fit(train_norm, train_labels)

    rf_pred=randomfc.predict(test_norm)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", accuracy_score(test_labels, rf_pred))

    #matriz de confusão
    matconf = confusion_matrix(test_labels, rf_pred)
    print("\nmatriz confusão: \n",matconf)
    tn, fp, fn, tp = matconf.ravel()
    
    print("\nTrue Positives: ",tp)
    print("False Positives: ",fp)
    print("False Negatives: ",fn)
    print("True Negatives: ",tn)

    temp_rf = []

    Accuracy = (tn+tp)/(tp+tn+fp+fn)
    print("\nAccuracy: {:.4f}".format(Accuracy))
    temp_rf.append(Accuracy)

    Precision = tp/(tp+fp)
    if math.isnan(Precision) == True: Precision = 0.0000
    print("Precision: {:.4f}".format(Precision))
    temp_rf.append(Precision)

    Recall = tp/(tp+fn)
    if math.isnan(Recall) == True: Recall = 0.0000
    print("Recall: {:.4f}".format(Recall))
    temp_rf.append(Recall)

    f1score = ((2*(Precision * Recall))/(Precision + Recall))
    if math.isnan(f1score) == True: f1score = 0.0000
    print("f1-score: {:.4f}".format(f1score))
    temp_rf.append(f1score)

    #Sensibilidade
    Sensitivity = tp/(tp+fn)
    if math.isnan(Sensitivity) == True: Sensitivity = 0.0000
    print("\nSensitivity {:0.4f}".format(Sensitivity))
    temp_rf.append(Sensitivity)

    #Especificidade
    Specificity = tn/(tn+fp)
    if math.isnan(Specificity) == True: Specificity = 0.0000
    print("Specificity {:0.4f}".format(Specificity))
    temp_rf.append(Specificity)

    return temp_rf

def data_model(np_matriz0, targets0, np_matriz1, targets1): 
    #labels 1
    #targets1 = np.repeat(1, np_matriz1.shape[0]) #minority class
    train1, test1, train_labels1, test_labels1 = train_test_split(np_matriz1, targets1, test_size=0.3, shuffle=True, random_state=42)

    #labels 0
    #targets0 = np.repeat(0, np_matriz0.shape[0]) #majority class
    train0, test0, train_labels0, test_labels0 = train_test_split(np_matriz0, targets0, test_size=0.3, shuffle=True, random_state=42)

    #preparando datos para el modelo con dataset deslanaciado
    train = np.concatenate((train1, train0))
    test = np.concatenate((test1, test0))
    train_labels = np.concatenate((train_labels1, train_labels0))
    test_labels = np.concatenate((test_labels1, test_labels0))

    #standardize
    scaler = StandardScaler()
    # transform data
    train_norm = scaler.fit_transform(train)
    test_norm = scaler.fit_transform(test)

    print("Dados de treinamento 70% --- dados de teste 30%")
    print(train_norm.shape, ' ', train_labels.shape)
    print(test_norm.shape, ' ', test_labels.shape)
    #print("\n")

    return train_norm, test_norm, train_labels, test_labels


def select_subcarries(X, y):
    from sklearn.datasets import make_regression
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    # generate dataset
    #X, y = make_regression(n_samples=100, n_features=100, n_informative=10)
    # define feature selection
    fs = SelectKBest(score_func=f_regression, k=234)
    # apply feature selection
    #new_dataset = fs.fit_transform(dataset_features, dataset_labels)
    X = fs.fit_transform(X, y)
    #print('\nMatriz Bool: ',fs.get_support())
    new_features = np.where(fs.get_support())
    print('New features selections: ', new_features[0])#, ' -- ', X.shape)
    
    return X, y, new_features[0]








#################################### Funções para manipular dados CSI ###########################

def generateIntervals(ini_, end_, flag):

    intervals = []

    if (flag): intervals = [str(i).zfill(3) for i in range(ini_, end_ + 1)]
    else: intervals = [str(numero) for numero in range(ini_, end_ + 1)]
    
    return intervals

def loadDataPath():
    
    path0 = "/home/jsoto/exe_remoto_presencia/scans_vacio_csv/"
    path1 = "/home/jsoto/exe_remoto_presencia/scans_csv/"
    
    emptyRoomsTrain = generateIntervals(1, 1360, False) # 850 es para 50 participantes
    emptyRoomsTest = generateIntervals(1361, 1700, False) # dataset antiguo
    #emptyRoomsTest = generateIntervals(1361, 1530, False) # nuevo dataset

    participantesTreino = generateIntervals(1, 80, True) # 80 participantes
    partipantesTeste = generateIntervals(81, 100, True) # dataset antiguo # 20 participantes
    #partipantesTeste = generateIntervals(20, 29, True) # nuevo dataset

    posiciones = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

    return path0, path1, participantesTreino, partipantesTeste, posiciones, emptyRoomsTrain, emptyRoomsTest

def loadReferenceData(windowsSize_):

    rutavacia = '/home/jsoto/exe_remoto_presencia/csi_scans_csv/000/0-1.csv'

    #cargando sala vacia referencial
    df0 = pd.read_csv(rutavacia)
    df0 = df0.drop(['Unnamed: 0'], axis=1)
    df0 = df0.apply(lambda col: col.apply(lambda val: complex(val.strip('()'))))
    df0 = df0.iloc[0:windowsSize_] # utilizando la VENTANA DESLIZANTE en la sala vacia para la sala de referencia
    df0 = hampel_filter(df0)
    series0 = moving_avg_filter(df0)
    series0 = iq_samples_abs(series0)

    return series0

def presenceDetectionProcess0(path0Train_, emptyRooms_, series0, windowsSize_):

    matriz_np0 = np.empty((0, 234))

    for p in emptyRooms_:
        #print('\n--------> ', p, type(p))
        df = pd.read_csv(path0Train_ + p + ".csv")
        df = df.drop(['Unnamed: 0'], axis=1)
        df = df.apply(lambda col: col.apply(lambda val: complex(val.strip('()'))))

        if df.shape[0] > 500: 
            df = df.iloc[windowsSize_] #[::4]
            df = df.reset_index(drop=True)
        else: df = df.iloc[0:windowsSize_] # utilizando la VENTANA DESLIZANTE para las salas vacias

        df = hampel_filter(df)
        series = moving_avg_filter(df)
        series = iq_samples_abs(series)
        
        distancia1 = []      
        for key in series.keys():
            #distance, path = fastdtw(series[key], series0[key], dist=euclidean) #calculo de distancia euclediana con dtw
            #distancia1.append(distance)
            max_amplitudes = max(series[key])
            distancia1.append(max_amplitudes)
        matriz_np0 = np.concatenate((matriz_np0,np.array(distancia1).reshape(1,234)), axis=0)
        
        #matriz_np0 = np.concatenate((matriz_np0, series), axis=0)

    print("--> shape matriz 0: ", matriz_np0.shape)
    return matriz_np0
    

def presenceDetectionProcess1(path1_, participants_, positions_, series0, windowsSize_):

    matriz_np1 = np.empty((0, 234))

    for part in participants_:
        #print('|participante: ',part, type(part))

        for p in positions_:
            #print('                  \'---> ', p, type(p))
            df = pd.read_csv(path1_ + part + "/" + str(p) + ".csv")
            df = df.drop(['Unnamed: 0'], axis=1)
            df = df.apply(lambda col: col.apply(lambda val: complex(val.strip('()'))))
            
            if df.shape[0] > 500: 
                df = df.iloc[0:windowsSize_] #[::4]
                df = df.reset_index(drop=True)
            else: df = df.iloc[0:windowsSize_] # utilizando la VENTANA DESLIZANTE para las salas llenas

            df = hampel_filter(df)
            series = moving_avg_filter(df)
            series = iq_samples_abs(series)
            
            distancia = []
            for key in series.keys():
                #distance_, path_ = fastdtw(series[key], series0[key], dist=euclidean) #calculo de distancia euclediana con dtw
                #distancia.append(distance_)
                max_amplitudes = max(series[key])
                distancia.append(max_amplitudes)

            matriz_np1 = np.concatenate((matriz_np1,np.array(distancia).reshape(1,234)), axis=0)
            
            #matriz_np1 = np.concatenate((matriz_np1, series), axis=0)

    print("--> shape matriz 1: ", matriz_np1.shape)
    return matriz_np1

def dataModelStandardize(np_matriz0, targets0, np_matriz1, targets1): 

    #preparando datos para el modelo con dataset deslanaciado
    data = np.concatenate((np_matriz0, np_matriz1))
    data_labels = np.concatenate((targets0, targets1))

    #standardize
    scaler = StandardScaler()
    # transform data
    data_norm = scaler.fit_transform(data)

    return data_norm, data_labels



#################################### DETEÇÂO DE PRESENÇA EM TEMPO REAL ###############################################

def presence_real_time(sec):
    print('\n#############################################################################################')
    windowsSize_ = sec * 33 ##### <tiempo segundos * número de samples> tamanho da janela para formar a serie temporal de amplitude, tamaños: 1, 2, 3 ,5, 9, 11, 21
    print('### Comienzo con tiempo: ', sec, ' windows size de: ', windowsSize_)

    # process data
    path0_, path1_, partTrain_, partTest_, positions_, emptyRoomsTrain_, emptyRoomsTest_ = loadDataPath()
    series0_ = loadReferenceData(windowsSize_)

    
    # training data
    seriesEmptyTrain_ = presenceDetectionProcess0(path0_, emptyRoomsTrain_, series0_, windowsSize_)
    seriesPresenceTrain_ = presenceDetectionProcess1(path1_, partTrain_, positions_, series0_, windowsSize_)

    targets0 = np.repeat(0, seriesEmptyTrain_.shape[0])
    targets1 = np.repeat(1, seriesPresenceTrain_.shape[0])

    dataTrain_, dataTrainLabels_ = dataModelStandardize(seriesEmptyTrain_, targets0, seriesPresenceTrain_, targets1)
    
    #Classifiers
    randomfc=RandomForestClassifier(n_estimators=100) # RF
    randomfc.fit(dataTrain_, dataTrainLabels_) # RF

    
    # Guarda el modelo
    print('guardando modelo ...')
    with open("/home/jsoto/exe_remoto_presencia/csi/models/modelo-RF_MaxAmplitude_50p_"+ str(windowsSize_) +".pkl", "wb") as f:
        pickle.dump(randomfc, f)
    print('modelo guardado!!!')
    
    
    # Carga el modelo
    print('cargando modelo ...')
    with open(os.path.join("models", "../models/modelo-RF_MaxAmplitude_50p_480.pkl"), "rb") as f:    # modelo-svm_50p.pkl      modelo-J48_50p.pkl       modelo-NB_50p.pkl
        randomfc = pickle.load(f)
    print('ok...\n')
    

     # testing data...

    #emptyRoomsTest_17_ = random.sample(emptyRoomsTest_,17)
    #seriesEmptyTest_ = presenceDetectionProcess0(path0_, emptyRoomsTest_17_, series0_, windowsSize_)
    
    acuracia_ = []
    precision_ = []
    recall_ = []
    f1score_ = []
    tiempo_ = []

    for r in range(0,1,1):

        print('run test...')

        emptyRoomsTest_170 = random.sample(emptyRoomsTest_,170) # escoge 170 muestras que equivale a 10 participantes
        seriesEmptyTest_ = presenceDetectionProcess0(path0_, emptyRoomsTest_170, series0_, windowsSize_) # calcula con todos las muestras al mimso tiempo

        
        
        t1 = time()
        
        #seriesPresenceTest_ = presenceDetectionProcess1(path1_, [part_], positions_, series0_, windowsSize_)
        
        fullroomTest_10 = random.sample(partTest_,10)
        seriesPresenceTest_ = presenceDetectionProcess1(path1_, fullroomTest_10, positions_, series0_, windowsSize_) # calcula con todos las muestras al mimso tiempo

        print('\nshape test-empty   : ', seriesEmptyTest_.shape)
        print('shape test-presence: ', seriesPresenceTest_.shape, '\n')

        targets0 = np.repeat(0, seriesEmptyTest_.shape[0])
        targets1 = np.repeat(1, seriesPresenceTest_.shape[0])

        dataTest_, dataTestLabels_ = dataModelStandardize(seriesEmptyTest_, targets0, seriesPresenceTest_, targets1)

        pred = randomfc.predict(dataTest_) # random forest
            
        acc_ = accuracy_score(dataTestLabels_, pred)
        prec_ = precision_score(dataTestLabels_, pred)
        rec_ = recall_score(dataTestLabels_, pred)
        f1_ = f1_score(dataTestLabels_, pred)

        t2 = time()
        processTime_ = t2 - t1

        acuracia_.append(acc_)
        precision_.append(prec_)
        recall_.append(rec_)
        f1score_.append(f1_)
        tiempo_.append(processTime_)

        print('##### ---> round: ', r + 1, ' --- Acuracia: ', acc_, ' --- Precision: ', prec_, ' --- Recall: ', rec_, ' --- F1_score: ', f1_, ' --- tiempo: ', processTime_, '\n') # calcula con todos las muestras al mimso tiempo

    return acuracia_, tiempo_, precision_, recall_, f1score_



def presence_results():

    seconds = [1,2,3,5,9,11,21,30,40,50,60]
    for sec in seconds: 
        accTotal_, timeTotal_, precTotal_, recTotal_, f1Total_ = presence_real_time(sec) 
    
        # resultados
        confidence = 0.95 # intervalo de confianza

        accTotal_ = np.array(accTotal_).T
        print('\n#############################################################################################')
        print('lista de acuracias: ',accTotal_)
        print('lista de precisions: ',precTotal_)
        print('lista de recalls: ',recTotal_)
        print('lista de f1scores: ',f1Total_)
        print('lista de tiempos: ',timeTotal_)

        print('\n\n#############################################################################################')
        print('#############################################################################################')
        print('Window size (<seconds> | <packets>): ', sec, sec * 8)          
        #----- accuracia ------
        val = [np.random.choice(accTotal_,size=len(accTotal_),replace=True).mean() for i in range(1000)] 
        print('accuracia mean: ', np.mean(accTotal_),
                '  -----  interval: ', np.percentile(val,[100*(1-confidence)/2,100*(1-(1-confidence)/2)]))
         #----- precision ------
        val = [np.random.choice(precTotal_,size=len(precTotal_),replace=True).mean() for i in range(1000)] 
        print('precision mean: ', np.mean(precTotal_),
                '  -----  interval: ', np.percentile(val,[100*(1-confidence)/2,100*(1-(1-confidence)/2)]))
         #----- recall ------
        val = [np.random.choice(recTotal_,size=len(recTotal_),replace=True).mean() for i in range(1000)] 
        print('recall mean: ', np.mean(recTotal_),
                '  -----  interval: ', np.percentile(val,[100*(1-confidence)/2,100*(1-(1-confidence)/2)]))
         #----- f1_score ------
        val = [np.random.choice(f1Total_,size=len(f1Total_),replace=True).mean() for i in range(1000)] 
        print('f1-score mean: ', np.mean(f1Total_),
                '  -----  interval: ', np.percentile(val,[100*(1-confidence)/2,100*(1-(1-confidence)/2)]))
        #----- tiempo ------
        val = [np.random.choice(timeTotal_,size=len(timeTotal_),replace=True).mean() for i in range(1000)] 
        print('tiempo mean: ', np.mean(timeTotal_), #*1000 para pasar a milisegundos
                '  -----  interval: ', np.percentile(val,[100*(1-confidence)/2,100*(1-(1-confidence)/2)]))
        print('#############################################################################################')
        print('#############################################################################################')








############################################ IDENTIFICAÇÃO EM TEMPO REAL ################################################


def identification_real_time(grupo, window):
    import collections

    path = "/home/jsoto/exe_remoto_presencia/amplitudes_csi_csv/matriz_"


    principal = grupo
    participantes = grupo

    windows_size = window #tamanho da janela só para teste

    rf_metrics = []
    svm_metrics = []
    bayes_metrics = []
    j48_metrics = []

    accTo = []
    timTo = []
    
    for prin in principal:
        print("--- ", prin, " ---")
        np_matriz1 = pd.read_csv(path + prin +".csv", header=None).to_numpy() # minority class 1
        
        np_matriz0 = np.empty((0, 234)) # majority class 0
        print("--------- vs --: ")
        for part in participantes:
            if prin != part:
                print("             ---> ", part)
                np_matriz_temp = pd.read_csv(path + part +".csv", header=None).to_numpy() # majority class 0
                np.random.shuffle(np_matriz_temp) # suffle para balancear 
                np_matriz_temp = np_matriz_temp[0:1700,:] # escoger una porcion para balancear
                np_matriz0 = np.concatenate((np_matriz0, np_matriz_temp), axis=0)

        print("minority class _ 1: ", np_matriz1.shape)
        print("majority class _ 0: ", np_matriz0.shape)
        
        targets1 = np.repeat(1, np_matriz1.shape[0]) #minority class
        targets0 = np.repeat(0, np_matriz0.shape[0]) #majority class

        
           
        
        train_norm, test_norm, train_labels, test_labels = data_model(np_matriz0, targets0, np_matriz1, targets1) # split data

        ##########train_norm, test_norm, train_labels, test_labels = data_model(np_matriz0, np_matriz1) # split data (antiguo)

        #train_norm, train_labels, n_features = select_subcarries(train_norm, train_labels)
        ##########test_norm, test_labels = select_subcarries(test_norm, test_labels) #(antiguo)
        #test_norm = test_norm[:, n_features]

        print("\n### window size: ", windows_size)
        #print("\ntraining...")
        
        #Classifiers
        randomfc=RandomForestClassifier(n_estimators=100) # RF
        randomfc.fit(train_norm, train_labels) # RF



        # Guarda el modelo
        with open("/home/jsoto/exe_remoto_presencia/csi/models/ide_models/modelo-RF_500a.pkl", "wb") as f:
            pickle.dump(randomfc, f)

        rows, columns = test_norm.shape
        print("filas: ", rows, ' columnas: ', columns)
        

        total = 0
        cont = 0
        totaltime = 0

        for i in range(0,rows-windows_size+1,windows_size):
            print("\n###############################\ninicio: ", i," fin: ", i + windows_size) 
            print("\ntesting...")

            temp_part = test_norm[i:i + windows_size]
            temp_part_label = test_labels[i:i + windows_size]

            print(temp_part.shape, ' ', temp_part_label.shape)

            rf_pred=randomfc.predict(temp_part) # random forest
      
            print('real: ', temp_part_label, '--- prediction: ', rf_pred)

            ini = collections.Counter(temp_part_label)
            print("contadores de lista:", ini, ini.most_common()[0][0], ini.most_common()[0][1])

            total += 1

            if ini.most_common()[0][0] == 1:
                t1 = time()
                c = collections.Counter(rf_pred)
                pre = c.most_common()
                if pre[0][0] == 1: cont += 1
                t2 = time()
                elapsed = t2 - t1
                totaltime += elapsed
                print('Processing time %f seconds.' % elapsed)

            elif ini.most_common()[0][0] == 0:
                t1 = time()
                c = collections.Counter(rf_pred)
                pre = c.most_common()
                if pre[0][0] == 0: cont += 1
                t2 = time()
                elapsed = t2 - t1
                totaltime += elapsed
                print('Processing time %f seconds.' % elapsed)


            # Model Accuracy, how often is the classifier correct
            print("Accuracy:", accuracy_score(temp_part_label, rf_pred))
            

        print('\n\n################################################')  
        print('\n### Window size: ', windows_size)      
        print('Acuracia final: ', cont/total)
        print('tiempo promedio utilizado: ', (totaltime/total)*1000)
        accTo.append(cont/total)
        timTo.append((totaltime/total)*1000)

        #list_rf = random_forest_ml(train_norm, test_norm, train_labels, test_labels) # call random forest
        #rf_metrics.append(list_rf)

        print("\n-------------------------------------------------------------------------------------------------------")
    
    return np.mean(accTo), np.mean(timTo)



def identification_results():
    
    list_pre_select = ['006','007','010','017','036','112']
    participantes = ['001','002','003','004','005','008','009',
                     '011','012','013','014','015','016','018','019','020',
                     '021','022','023','024','025','026','027','028','029','030',
                     '031','032','033','034','035','037','038','039','040',
                     '041','042','043','044','045','046','047','048','049','050',
                     '051','052','053','054','055','056','057','058','059','060',
                     '061','062','063','064','065','066','067','068','069','070',
                     '071','072','073','074','075','076','077','078','079','080',
                     '081','082','083','084','085','086','087','088','089','090',
                     '091','092','093','094','095','096','097','098','099','100',
                     '101','102','103','104','105','106','107','108','109','110',
                     '111','113','114','115','116']

    
    #seleção de 10 grupos de 6 pessoas
    random.shuffle(participantes)
    groups = participantes[:54]
    groups = np.array(groups).reshape(9,6)
    groups = groups.tolist()
    groups.append(list_pre_select)

    result_sentado = []
    result_pe = []
    result_deitado = []
    result_movimento = []
    result_todo = []

    print(groups)

    accT = []
    timeT = []

    window = 500 ####tamanho da janela

    for g in groups:
        acc, tim = identification_real_time(g, window)
        accT.append(acc)
        timeT.append(tim)

    confidence = 0.95

    accT = np.array(accT).T
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    print('lista de acuracias: ',accT)
    print('lista de tiempos: ',timeT)

    print('#############################################################################################')
    print('Windows size: ', window)
    print('----- accuracia ------')
    val = [np.random.choice(accT,size=len(accT),replace=True).mean() for i in range(1000)] 
    print(' mean: ', np.mean(accT),
            '  -----  interval: ', np.percentile(val,[100*(1-confidence)/2,100*(1-(1-confidence)/2)]))

    print('----- tiempo ------')
    val = [np.random.choice(timeT,size=len(timeT),replace=True).mean() for i in range(1000)] 
    print(' mean: ', np.mean(timeT),
            '  -----  interval: ', np.percentile(val,[100*(1-confidence)/2,100*(1-(1-confidence)/2)]))
    



#####################################################################################################################################

if __name__ == "__main__":

    presence_results()
    identification_results()

    print("\nfinalizado con exito")
