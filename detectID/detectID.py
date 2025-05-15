import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle
import os
from time import time
#from decoders import interleaved as decoder #descomentar si procesa archivos pcap
from filters import iq_samples_abs, hampel_filter, moving_avg_filter
from model_lstm import get_results_identification_lstm
import json

import warnings
warnings.simplefilter("ignore", np.ComplexWarning)


###################################################################################################################
#################################### FUNCIONES GENERALES PARA MANIPULACION DE DATOS CSI ###########################
def get_amplitudes(df_csi_data):
    df_csi_data = iq_samples_abs(df_csi_data)

    return df_csi_data

def get_data_from_filters(df_csi_data):
    df_csi_data = hampel_filter(df_csi_data)
    df_csi_data = moving_avg_filter(df_csi_data)

    return df_csi_data

def get_selected_subcarries(df_csi_data):
    index_drop = [26] + list(range(54, 234))
    df_csi_data = df_csi_data.drop(df_csi_data.columns[index_drop], axis=1)

    return df_csi_data

def get_processing_csi_data_x_participant_from_csv_pcap(participant, position):

    # samples = decoder.read_pcap(path) # descomentar si se usa datos csi del pcap
    # df_csi_data = samples.get_pd_csi() # descomentar si se usa datos csi del pcap

    df_csi_data = pd.read_csv(PATH_PARTICIPANT_COMPLEX_CSV + participant + '/' + position + '.csv') # comentar si se usa datos csi del pcap
    
    df_csi_data = get_selected_subcarries(df_csi_data)

    df_csi_data = df_csi_data.drop(['Unnamed: 0'], axis=1)
    df_csi_data = df_csi_data.apply(lambda col: col.apply(lambda val: complex(val.strip('()'))))

    if df_csi_data.shape[0] > 500:
        df_csi_data = df_csi_data.iloc[0:SAMPLES] #[::4]
        df_csi_data = df_csi_data.reset_index(drop=True)
    else:
        df_csi_data = df_csi_data.iloc[0:SAMPLES]

    df_csi_data = get_data_from_filters(df_csi_data)
    df_csi_data = get_amplitudes(df_csi_data)

    return df_csi_data.to_numpy()


def labels_generator(label, shapedim): # genera los labels binarios 0 o 1
    return np.repeat(label, shapedim)


##########################################################################################################################
#################################### DETECCIÓN DE PRESENCIA EN TIEMPO REAL ###############################################
def get_processing_csi_data_x_room_from_csv_pcap(room):
    df_csi_data = pd.read_csv(PATH_EMPTYROOM_COMPLEX_CSV + room + '.csv') # comentar si se usa datos csi del pcap

    df_csi_data = get_selected_subcarries(df_csi_data)

    df_csi_data = df_csi_data.drop(['Unnamed: 0'], axis=1)
    df_csi_data = df_csi_data.apply(lambda col: col.apply(lambda val: complex(val.strip('()'))))

    if df_csi_data.shape[0] > 500:
        df_csi_data = df_csi_data.iloc[0:SAMPLES] #[::4]
        df_csi_data = df_csi_data.reset_index(drop=True)
    else:
        df_csi_data = df_csi_data.iloc[0:SAMPLES] # utilizando la VENTANA DESLIZANTE para las salas llenas

    df_csi_data = get_data_from_filters(df_csi_data)
    df_csi_data = get_amplitudes(df_csi_data)

    return df_csi_data.to_numpy()

def get_max_amplitude_per_position(np_matrix):
    array_maxamplitudes = []
    num_columns = np_matrix.shape[1]  # Obtiene el número de columnas del array

    for i in range(num_columns):
        column = np_matrix[:, i]  # Selecciona la i-ésima columna (todas las filas)
        max_amplitud = np.max(column)  # Calcula el valor máximo en esa columna
        array_maxamplitudes.append(max_amplitud)

    return array_maxamplitudes

def get_csi_data_from_emptyrooms_to_presencedetection():
    # obtiene informacion y amplitud maxima de salas vacias
    np_emptyrooms = np.empty((0, 52))
    emptyrooms_number = (len(PARTICIPANTS) * len(POSITIONS_PARTICIPANT))
    for room in emptyrooms_number:
        arrayemptyrooms_maxamplitudes = get_max_amplitude_per_position(
                                        get_processing_csi_data_x_room_from_csv_pcap(room))
        np_emptyrooms = np.concatenate((np_emptyrooms, np.array(arrayemptyrooms_maxamplitudes).reshape(1,52)), axis=0)

    return np_emptyrooms

def get_csi_data_to_presencedetection(participant):
    # obtiene informacion y amplitud maxima de participantes
    np_fullrooms = np.empty((0, 52))
    #for participant in PARTICIPANTS:
    for position in POSITIONS_PARTICIPANT:
        arrayparticipants_maxamplitudes = get_max_amplitude_per_position(
                                    get_processing_csi_data_x_participant_from_csv_pcap(participant, position))
        np_fullrooms = np.concatenate((np_fullrooms, np.array(arrayparticipants_maxamplitudes).reshape(1,52)), axis=0)

    return np_fullrooms

def split_data_to_presencedetection():
    participants_training = PARTICIPANTS[:RATIO[0]]
    participants_validate = PARTICIPANTS[RATIO[0]:RATIO[0] + RATIO[1]]
    participants_test = PARTICIPANTS[RATIO[0] + RATIO[1]:]

    return participants_training, participants_validate, participants_test

def real_time_presencedetection(window):
    dict_training_results = {}  # Diccionario para almacenar los resultados
    participants_training, participants_validate, participants_test = split_data_to_presencedetection() # division de cantidad de participantes 
    
    for participant in participants_training:
        np_fullrooms, np_emptyrooms = get_csi_data_to_presencedetection(participant)
        targets_fullrooms = labels_generator(1,np_fullrooms.shape[0])
        targets_emptyrooms = labels_generator(0,np_emptyrooms.shape[0])





        # TO DO
        # aqui llama modelo para entrenar
        # Simulación del entrenamiento del modelo y obtención de métricas
        # (Reemplaza esto con tu código real de entrenamiento y evaluación)
        accuracy = 0.85 + random.uniform(-0.05, 0.05)
        precision = 0.78 + random.uniform(-0.05, 0.05)
        recall = 0.92 + random.uniform(-0.05, 0.05)
        f1 = 0.84 + random.uniform(-0.05, 0.05)
        time_taken = 1.2 + random.uniform(-0.2, 0.2)
        ##### borrar hasta TO DO cuando se implemente modelo

        dict_training_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'time': time_taken
        }

    return dict_training_results

###
def presence_results():
    general_results = {}
    for window in WINDOWS_SIZES_TO_ROUNDS:
        general_results[window] = {}  # Inicializar un diccionario para los resultados de esta ventana
        dict_model_results = real_time_presencedetection(window)
        general_results[window].update(dict_model_results)

    file_name = 'indetification_results_' + model_name + '.json'
    try:
        with open(file_name, 'w') as archivo_json:
            json.dump(general_results, archivo_json, indent=4)
        print(f"Los resultados se han guardado exitosamente en '{file_name}'")
    except Exception as e:
        print(f"Ocurrió un error al guardar los resultados: {e}")





###########################################################################################################################
############################################ IDENTIFICACION EN TIEMPO REAL ################################################  
def get_csi_data_group_identification(participant): #retorna las amplitudes de todas las posiciones de un participante

    position_matrix = np.empty((0, 52))
    for position in POSITIONS_PARTICIPANT:
        tmp_matrix = get_processing_csi_data_x_participant_from_csv_pcap(participant, position)
        position_matrix = np.concatenate((position_matrix, tmp_matrix), axis=0)

    return position_matrix


def get_csi_data_to_indentification(group): # retorna las dos matrices, una con los datos del princiapl y otra con datos de los restantes o secundarios

    np_principal = get_csi_data_group_identification(group[0])

    np_secondaries = np.empty((0, 52))
    for participant in group[1:]:
        tmp_matrix_participant = get_csi_data_group_identification(participant)
        random_number = random.randint(0, 13)
        tmp_matrix_participant = tmp_matrix_participant[random_number * SAMPLES : (random_number * SAMPLES) + 1700, :] # 1700 es el numero de samples calculado para 5 personas y quede balanceado los dos grupos con 8500 samples cada uno
        np_secondaries = np.concatenate((np_secondaries, tmp_matrix_participant), axis=0)

    return np_principal, np_secondaries
    

def split_data_to_identification(np_matrix):
    total_ratio = sum(RATIO)

    train_percentage = RATIO[0] / total_ratio
    val_percentage = RATIO[1] / total_ratio
    test_percentage = RATIO[2] / total_ratio

    total_samples = np_matrix.shape[0]

    split_train = int(train_percentage * total_samples)
    split_val = int((train_percentage + val_percentage) * total_samples)

    train_data = np_matrix[:split_train, :]
    val_data = np_matrix[split_train:split_val, :]
    test_data = np_matrix[split_val:, :]

    return train_data, val_data, test_data



def real_time_identification(group, window):
    dict_identification_results = {} # diccionario para almacenar resultados para todos los modelos del grupo actual
    for principal in group:
        list_group = []
        list_group.append(principal)
        for secondaries in group:
            if principal != secondaries:
                list_group.append(secondaries)

        np_principal, np_secondaries = get_csi_data_to_indentification(list_group) # recibe todos los datos procesados

        train_principal_data, val_principal_data, test_principal_data = split_data_to_identification(np_principal)
        train_secondaries_data, val_secondaries_data, test_secondaries_data = split_data_to_identification(np_secondaries)

        train_principal_labels = labels_generator(1, train_principal_data.shape[0])
        val_principal_labels = labels_generator(1, val_principal_data.shape[0])
        test_principal_labels = labels_generator(1, test_principal_data.shape[0])

        train_secondaries_labels = labels_generator(0, train_secondaries_data.shape[0])
        val_secondaries_labels = labels_generator(0, val_secondaries_data.shape[0])
        test_secondaries_labels = labels_generator(0, test_secondaries_data.shape[0])

        dict_model_results = get_results_identification_lstm(train_principal_data, val_principal_data, test_principal_data,
                                        train_secondaries_data, val_secondaries_data, test_secondaries_data,
                                        train_principal_labels, val_principal_labels, test_principal_labels,
                                        train_secondaries_labels, val_secondaries_labels, test_secondaries_labels)

        identifier = ':'.join(list_group)
        dict_identification_results[identifier] = dict_model_results

    return dict_identification_results

###
def identification_results(model_name):
    general_results = {}  # Diccionario general para almacenar los resultados
    participantes = PARTICIPANTS.copy()
    random.shuffle(participantes)

    if len(participantes) >= 60:
        groups = participantes[:60]
        groups = np.array(groups).reshape(10, 6)
        groups = groups.tolist()
        for indice, grupo in enumerate(groups):
            print('grupo ', indice + 1, ': ', grupo)
    else:
        print(f"No hay suficientes participantes (solo {len(participantes)}) para formar 10 grupos de 6.")

    for window in WINDOWS_SIZES_TO_ROUNDS:
        for group in groups:
            general_results[window] = {}  # Inicializar un diccionario para los resultados de esta ventana
            dict_model_results = real_time_identification(group, window)
            general_results[window].update(dict_model_results)

    file_name = 'indetification_results_' + model_name + '.json'
    try:
        with open(file_name, 'w') as archivo_json:
            json.dump(general_results, archivo_json, indent=4)
        print(f"Los resultados se han guardado exitosamente en '{file_name}'")
    except Exception as e:
        print(f"Ocurrió un error al guardar los resultados: {e}")



#################################################################################################
############################ MAIN ###############################################################
def config_scheme():

    participantes = ['001','002','003','004','005','006','007','008','009','010',
                     '011','012','013','014','015','016','017','018','019','020',
                     '021','022','023','024','025','026','027','028','029','030',
                     '031','032','033','034','035','036','037','038','039','040',
                     '041','042','043','044','045','046','047','048','049','050',
                     '051','052','053','054','055','056','057','058','059','060',
                     '061','062','063','064','065','066','067','068','069','070',
                     '071','072','073','074','075','076','077','078','079','080',
                     '081','082','083','084','085','086','087','088','089','090',
                     '091','092','093','094','095','096','097','098','099','100',
                     '101','102','103','104','105','106','107','108','109','110',
                     '111','112','113','114','115','116','117','118','119','120',
                     '121','122','123','124','125']
    
    global MODEL_NAMES, \
        SAMPLES, \
        PARTICIPANTS, \
        PATH_PARTICIPANT_COMPLEX_CSV, \
        PATH_EMPTYROOM_COMPLEX_CSV, \
        PARTICIPANTS_NUMBER, \
        POSITIONS_PARTICIPANT, \
        RATIO, \
        WINDOWS_SIZES_TO_ROUNDS #, \
        #WINDOWS_IDENTIFICATON

    global WINDOWS_SIZE #temporal solo testeo  --- borrar
    WINDOWS_SIZE = 500 #temporal solo testeo  --- borrar

    MODEL_NAMES = ['RF', 'LSTM', 'AE']
    SAMPLES = 500
    PARTICIPANTS = participantes
    PATH_PARTICIPANT_COMPLEX_CSV = '/home/jsoto/exe_remoto_presencia/scans_csv/'
    PATH_EMPTYROOM_COMPLEX_CSV = '/home/jsoto/exe_remoto_presencia/scans_vacio_csv/'
    PARTICIPANTS_NUMBER = len(participantes)
    POSITIONS_PARTICIPANT = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
    RATIO = [80,20,25]

    WINDOWS_SIZES_TO_ROUNDS = ['1','3','5','11','21','30','40','50','60']
    #WINDOWS_IDENTIFICATON = ['1','3','5','9','11','21']

if __name__ == "__main__":

    ###### configuration scheme
    config_scheme()

    model_name = 1 # 0 : random forest, 1 : lstm, 2 : AutoEncoder

    presence_results(model_name)
    identification_results(model_name)

    print("\nfinalizado con exito")
