import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import argparse
#from decoders import interleaved as decoder #descomentar si procesa archivos pcap
from filters import iq_samples_abs, hampel_filter_complex, moving_avg_filter_complex
from model_lstm import get_results_presence_lstm, get_results_identification_lstm
from model_mlp import get_results_presence_mlp
import json
import os

import warnings
warnings.simplefilter("ignore", np.ComplexWarning)


###################################################################################################################
#################################### FUNCIONES GENERALES PARA MANIPULACION DE DATOS CSI ###########################
def get_amplitudes(df_csi_data):
    df_csi_data = iq_samples_abs(df_csi_data)

    return df_csi_data

def get_data_from_filters(df_csi_data):
    #df_csi_data = hampel_filter_complex(df_csi_data)
    #df_csi_data = moving_avg_filter_complex(df_csi_data)
    df_csi_data = df_csi_data.apply(hampel_filter_complex)
    df_csi_data = df_csi_data.apply(moving_avg_filter_complex)

    return df_csi_data

def get_selected_subcarries(df_csi_data):
    index_drop = [26] + list(range(54, 235))
    df_csi_data = df_csi_data.drop(df_csi_data.columns[index_drop], axis=1)

    return df_csi_data

def get_processing_csi_data_x_participant_from_csv_pcap(participant, position, flag_window = False, window = None):

    # samples = decoder.read_pcap(path) # descomentar si se usa datos csi del pcap
    # df_csi_data = samples.get_pd_csi() # descomentar si se usa datos csi del pcap

    df_csi_data = pd.read_csv(PATH_PARTICIPANT_COMPLEX_CSV + participant + '/' + position + '.csv') # comentar si se usa datos csi del pcap
    
    df_csi_data = get_selected_subcarries(df_csi_data)

    df_csi_data = df_csi_data.drop(['Unnamed: 0'], axis=1)
    df_csi_data = df_csi_data.apply(lambda col: col.apply(lambda val: complex(val.strip('()'))))

    if df_csi_data.shape[0] > 500:
        if flag_window:
            df_csi_data = df_csi_data.iloc[0:window]
            df_csi_data = df_csi_data.reset_index(drop=True)
        else:
            df_csi_data = df_csi_data.iloc[0:SAMPLES]
            df_csi_data = df_csi_data.reset_index(drop=True)
    else:
        if flag_window:
            df_csi_data = df_csi_data.iloc[0:window]
        else:
            df_csi_data = df_csi_data.iloc[0:SAMPLES]

    df_csi_data = get_data_from_filters(df_csi_data)
    df_csi_data = get_amplitudes(df_csi_data)

    return df_csi_data.to_numpy()


def labels_generator(label, shapedim): # genera los labels binarios 0 o 1
    return np.repeat(label, shapedim)


##########################################################################################################################
#################################### DETECCIÓN DE PRESENCIA EN TIEMPO REAL ###############################################
def get_processing_csi_data_x_room_from_csv_pcap(room, flag_window, window):
    df_csi_data = pd.read_csv(PATH_EMPTYROOM_COMPLEX_CSV + room + '.csv') # comentar si se usa datos csi del pcap

    df_csi_data = get_selected_subcarries(df_csi_data)

    df_csi_data = df_csi_data.drop(['Unnamed: 0'], axis=1)
    df_csi_data = df_csi_data.apply(lambda col: col.apply(lambda val: complex(val.strip('()'))))

    if df_csi_data.shape[0] > 500:
        if flag_window:
            df_csi_data = df_csi_data.iloc[0:window]
            df_csi_data = df_csi_data.reset_index(drop=True)
        else:
            df_csi_data = df_csi_data.iloc[0:SAMPLES]
            df_csi_data = df_csi_data.reset_index(drop=True)
    else:
        if flag_window:
            df_csi_data = df_csi_data.iloc[0:window]
        else:
            df_csi_data = df_csi_data.iloc[0:SAMPLES]

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

def get_csi_data_from_emptyrooms_to_presencedetection(room, flag_window=False, window=None):
    # obtiene informacion y amplitud maxima de la sala vacia seleccionada
    np_emptyrooms = np.empty((0, 52))
    arrayemptyrooms_maxamplitudes = get_max_amplitude_per_position(
                                    get_processing_csi_data_x_room_from_csv_pcap(room, flag_window, window))
    np_emptyrooms = np.array(arrayemptyrooms_maxamplitudes).reshape(1,52)

    return np_emptyrooms

def get_csi_data_from_participant_to_presencedetection(participant, flag_window=False, window=None):
    # obtiene informacion y amplitud maxima del participante seleccionado
    np_fullroom_participant = np.empty((0, 52))
    for position in POSITIONS_PARTICIPANT:
        print('participant: ', participant, ' position: ', position)
        arrayparticipants_maxamplitudes = get_max_amplitude_per_position(
                                    get_processing_csi_data_x_participant_from_csv_pcap(participant, position, flag_window, window))
        np_fullroom_participant = np.concatenate((np_fullroom_participant, np.array(arrayparticipants_maxamplitudes).reshape(1,52)), axis=0)

    return np_fullroom_participant

def get_procesing_emptyrooms_data_per_set(range, flag_window = False, window = None):
    np_emptyrooms = np.empty((0, 52))
    start_num, end_num = range
    for room in range(start_num + 1, end_num + 1):
        np_emptyrooms_tmp = get_csi_data_from_emptyrooms_to_presencedetection(room, flag_window, window)
        np_emptyrooms = np.concatenate((np_emptyrooms, np_emptyrooms_tmp), axis=0)

    return np_emptyrooms

def get_processing_fullrooms_data_per_set(participants_set, flag_window = False, window = None):
    np_fullrooms = np.empty((0, 52))
    for participant in participants_set:
        np_fullrooms_tmp = get_csi_data_from_participant_to_presencedetection(participant, flag_window, window)
        np_fullrooms = np.concatenate((np_fullrooms, np_fullrooms_tmp), axis=0)

    return np_fullrooms

def split_data_participants_to_presencedetection():
    participants_training = PARTICIPANTS[:RATIO[0]]
    participants_validate = PARTICIPANTS[RATIO[0]:RATIO[0] + RATIO[1]]
    participants_test = PARTICIPANTS[RATIO[0] + RATIO[1]:]

    return participants_training, participants_validate, participants_test

def split_data_emptyrooms_to_presencedetection(train_size, val_size, test_size):
    train_start = 0
    train_end = train_size

    val_start = train_end
    val_end = val_start + val_size

    test_start = val_end
    test_end = test_start + test_size

    train_range = (train_start, train_end)
    val_range = (val_start, val_end)
    test_range = (test_start, test_end)

    return train_range, val_range, test_range



def real_time_presencedetection(window, model_name):
    dict_presencedetection_results = {}  # Diccionario para almacenar los resultados

    # --- Ruta donde guardar/cargar los archivos CSV ---
    PROCESSED_DATA_DIR = '\\processed_data\\presence\\'
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR) # Crea el directorio si no existe

    # Definir los nombres de archivo SOLO para los datos
    data_files = {
        'fullrooms_training': os.path.join(PROCESSED_DATA_DIR, 'fullrooms_training.csv'),
        'fullrooms_validate': os.path.join(PROCESSED_DATA_DIR, 'fullrooms_validate.csv'),
        'fullrooms_test': os.path.join(PROCESSED_DATA_DIR, 'fullrooms_test.csv'),
        'emptyrooms_training': os.path.join(PROCESSED_DATA_DIR, 'emptyrooms_training.csv'),
        'emptyrooms_validate': os.path.join(PROCESSED_DATA_DIR, 'emptyrooms_validate.csv'),
        'emptyrooms_test': os.path.join(PROCESSED_DATA_DIR, 'emptyrooms_test.csv'),
    }

    # Verificar si todos los archivos de datos existen
    all_data_files_exist = True
    for file_path in data_files.values():
        if not os.path.exists(file_path):
            all_data_files_exist = False
            print(f"Archivo de datos de presencia NO encontrado: {file_path}")
            break

    # Si todos los archivos de datos existen, cargarlos
    if all_data_files_exist:
        print(f"Todos los archivos de datos de presencia encontrados en '{PROCESSED_DATA_DIR}'. Cargando...")
        np_fullrooms_training = np.loadtxt(data_files['fullrooms_training'], delimiter=',')
        np_fullrooms_validate = np.loadtxt(data_files['fullrooms_validate'], delimiter=',')
        np_fullrooms_test = np.loadtxt(data_files['fullrooms_test'], delimiter=',')
        np_emptyrooms_training = np.loadtxt(data_files['emptyrooms_training'], delimiter=',')
        np_emptyrooms_validate = np.loadtxt(data_files['emptyrooms_validate'], delimiter=',')
        np_emptyrooms_test = np.loadtxt(data_files['emptyrooms_test'], delimiter=',')
        
        print("Datos de presencia cargados exitosamente. Generando etiquetas...")

        targets_fullrooms_training = labels_generator(1,np_fullrooms_training.shape[0])
        targets_fullrooms_validate = labels_generator(1,np_fullrooms_validate.shape[0])
        targets_fullrooms_test = labels_generator(1,np_fullrooms_test.shape[0])

        targets_emptyrooms_training = labels_generator(0,np_emptyrooms_training.shape[0])
        targets_emptyrooms_validate = labels_generator(0,np_emptyrooms_validate.shape[0])
        targets_emptyrooms_test = labels_generator(0,np_emptyrooms_test.shape[0])

        
        # TODO
        # eliminar es solo para debbugar el codigo
        # descomentar llamadas a modelos
        print('#####################################')
        print('################# Salas llenas')
        print('shape train: ', np_fullrooms_training.shape)
        print('shape val: ', np_fullrooms_validate.shape)
        print('shape test: ', np_fullrooms_test.shape)
        print('shape train label: ', targets_fullrooms_training.shape)
        print('shape val label: ', targets_fullrooms_validate.shape)
        print('shape test label: ', targets_fullrooms_test.shape)
        print('################# Salanas vacias')
        print('shape train: ', np_emptyrooms_training.shape)
        print('shape val: ', np_emptyrooms_validate.shape)
        print('shape test: ', np_emptyrooms_test.shape)
        print('shape train label: ', targets_emptyrooms_training.shape)
        print('shape val label: ', targets_emptyrooms_validate.shape)
        print('shape test label: ', targets_emptyrooms_test.shape)
        print('#####################################')

        '''
        if model_name == 0:
            print('comenzando modelo RANDOM FOREST')
        elif model_name == 1:
            print('comenzando modelo LSTM')
            window = 1
            dict_model_results = get_results_presence_lstm(np_fullrooms_training, np_fullrooms_validate, np_fullrooms_test,
                                                        np_emptyrooms_training, np_emptyrooms_validate, np_emptyrooms_test,
                                                        targets_fullrooms_training, targets_fullrooms_validate, targets_fullrooms_test,
                                                        targets_emptyrooms_training, targets_emptyrooms_validate, targets_emptyrooms_test, 
                                                        window)
        elif model_name == 2:
            print('comenzando modelo MLP - ANN')
            dict_model_results = get_results_presence_mlp(np_fullrooms_training, np_fullrooms_validate, np_fullrooms_test,
                                                        np_emptyrooms_training, np_emptyrooms_validate, np_emptyrooms_test,
                                                        targets_fullrooms_training, targets_fullrooms_validate, targets_fullrooms_test,
                                                        targets_emptyrooms_training, targets_emptyrooms_validate, targets_emptyrooms_test, 
                                                        window)


        if dict_model_results: dict_presencedetection_results.update(dict_model_results)
        else:
            print("\nNo se agregaron resultados al diccionario principal (dict_model_results era None/vacío).")

        '''

    else:

        participants_training, participants_validate, participants_test = split_data_participants_to_presencedetection() # division de cantidad de participantes 
        
        np_fullrooms_training = get_processing_fullrooms_data_per_set(participants_training)
        np_fullrooms_validate = get_processing_fullrooms_data_per_set(participants_validate, True, window)
        np_fullrooms_test = get_processing_fullrooms_data_per_set(participants_test, True, window)

        # TODO
        # ojo creo que esta mal porque no refleja el rango real de datos tomados, 
        # esta tomando el tamaño de la matriz segun lso vectores de amplitud maxima adquirida
        train_range, val_range, test_range = split_data_emptyrooms_to_presencedetection(np_fullrooms_training.shape[0], 
                                                np_fullrooms_validate.shape[0], 
                                                np_fullrooms_test.shape[0])

        np_emptyrooms_training = get_procesing_emptyrooms_data_per_set(train_range)
        np_emptyrooms_validate = get_procesing_emptyrooms_data_per_set(val_range, True, window)
        np_emptyrooms_test = get_procesing_emptyrooms_data_per_set(test_range, True, window)


        targets_fullrooms_training = labels_generator(1,np_fullrooms_training.shape[0])
        targets_fullrooms_validate = labels_generator(1,np_fullrooms_validate.shape[0])
        targets_fullrooms_test = labels_generator(1,np_fullrooms_test.shape[0])

        targets_emptyrooms_training = labels_generator(0,np_emptyrooms_training.shape[0])
        targets_emptyrooms_validate = labels_generator(0,np_emptyrooms_validate.shape[0])
        targets_emptyrooms_test = labels_generator(0,np_emptyrooms_test.shape[0])

        # Guardar los arrays de datos en archivos CSV
        np.savetxt(data_files['fullrooms_training'], np_fullrooms_training, delimiter=',')
        np.savetxt(data_files['fullrooms_validate'], np_fullrooms_validate, delimiter=',')
        np.savetxt(data_files['fullrooms_test'], np_fullrooms_test, delimiter=',')
        np.savetxt(data_files['emptyrooms_training'], np_emptyrooms_training, delimiter=',')
        np.savetxt(data_files['emptyrooms_validate'], np_emptyrooms_validate, delimiter=',')
        np.savetxt(data_files['emptyrooms_test'], np_emptyrooms_test, delimiter=',')
        
        print("Datos de presencia generados y guardados exitosamente. Generando etiquetas...")


        # TODO
        # eliminar es solo para debbugar el codigo
        # descomentar llamadas a modelos
        print('#####################################')
        print('################# Salas llenas')
        print('shape train: ', np_fullrooms_training.shape)
        print('shape val: ', np_fullrooms_validate.shape)
        print('shape test: ', np_fullrooms_test.shape)
        print('shape train label: ', targets_fullrooms_training.shape)
        print('shape val label: ', targets_fullrooms_validate.shape)
        print('shape test label: ', targets_fullrooms_test.shape)
        print('################# Salanas vacias')
        print('shape train: ', np_emptyrooms_training.shape)
        print('shape val: ', np_emptyrooms_validate.shape)
        print('shape test: ', np_emptyrooms_test.shape)
        print('shape train label: ', targets_emptyrooms_training.shape)
        print('shape val label: ', targets_emptyrooms_validate.shape)
        print('shape test label: ', targets_emptyrooms_test.shape)
        print('#####################################')
        
        '''
        if model_name == 0:
            print('comenzando modelo RANDOM FOREST')
        elif model_name == 1:
            print('comenzando modelo LSTM')
            window = 1
            dict_model_results = get_results_presence_lstm(np_fullrooms_training, np_fullrooms_validate, np_fullrooms_test,
                                                        np_emptyrooms_training, np_emptyrooms_validate, np_emptyrooms_test,
                                                        targets_fullrooms_training, targets_fullrooms_validate, targets_fullrooms_test,
                                                        targets_emptyrooms_training, targets_emptyrooms_validate, targets_emptyrooms_test, 
                                                        window)
        elif model_name == 2:
            print('comenzando modelo MLP - ANN')
            dict_model_results = get_results_presence_mlp(np_fullrooms_training, np_fullrooms_validate, np_fullrooms_test,
                                                        np_emptyrooms_training, np_emptyrooms_validate, np_emptyrooms_test,
                                                        targets_fullrooms_training, targets_fullrooms_validate, targets_fullrooms_test,
                                                        targets_emptyrooms_training, targets_emptyrooms_validate, targets_emptyrooms_test, 
                                                        window)


        if dict_model_results: dict_presencedetection_results.update(dict_model_results)
        else:
            print("\nNo se agregaron resultados al diccionario principal (dict_model_results era None/vacío).")

        '''

    return dict_presencedetection_results

###
def presence_results(model_name):
    general_results = {}
    for window in WINDOWS_SIZES_TO_ROUNDS:
        general_results[window] = {}  # Inicializar un diccionario para los resultados de esta ventana
        dict_model_results = real_time_presencedetection(window, model_name)
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

def real_time_identification(group, window, model_name):
    dict_identification_results = {} # diccionario para almacenar resultados para todos los modelos del grupo actual

    # --- Ruta donde guardar/cargar los archivos CSV ---
    PROCESSED_DATA_DIR = '\\processed_data\\identification\\'
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR) # Crea el directorio si no existe

    # Definir los nombres de archivo SOLO para los datos (no para las etiquetas)
    data_files = {
        'train_principal_data': os.path.join(PROCESSED_DATA_DIR, 'train_principal_data.csv'),
        'val_principal_data': os.path.join(PROCESSED_DATA_DIR, 'val_principal_data.csv'),
        'test_principal_data': os.path.join(PROCESSED_DATA_DIR, 'test_principal_data.csv'),
        'train_secondaries_data': os.path.join(PROCESSED_DATA_DIR, 'train_secondaries_data.csv'),
        'val_secondaries_data': os.path.join(PROCESSED_DATA_DIR, 'val_secondaries_data.csv'),
        'test_secondaries_data': os.path.join(PROCESSED_DATA_DIR, 'test_secondaries_data.csv'),
    }

    # Verificar si todos los archivos de datos existen
    all_data_files_exist = True
    for file_path in data_files.values():
        if not os.path.exists(file_path):
            all_data_files_exist = False
            print(f"Archivo de datos de indentificación NO encontrado: {file_path}")
            break

    # Si todos los archivos de datos existen, cargarlos
    if all_data_files_exist:
        print(f"Todos los archivos de datos procesados encontrados en '{PROCESSED_DATA_DIR}'. Cargando...")
        train_principal_data = np.loadtxt(data_files['train_principal_data'], delimiter=',')
        val_principal_data = np.loadtxt(data_files['val_principal_data'], delimiter=',')
        test_principal_data = np.loadtxt(data_files['test_principal_data'], delimiter=',')
        train_secondaries_data = np.loadtxt(data_files['train_secondaries_data'], delimiter=',')
        val_secondaries_data = np.loadtxt(data_files['val_secondaries_data'], delimiter=',')
        test_secondaries_data = np.loadtxt(data_files['test_secondaries_data'], delimiter=',')
        
        print("Datos cargados exitosamente. Generando etiquetas...")

        train_principal_labels = labels_generator(1, train_principal_data.shape[0])
        val_principal_labels = labels_generator(1, val_principal_data.shape[0])
        test_principal_labels = labels_generator(1, test_principal_data.shape[0])

        train_secondaries_labels = labels_generator(0, train_secondaries_data.shape[0])
        val_secondaries_labels = labels_generator(0, val_secondaries_data.shape[0])
        test_secondaries_labels = labels_generator(0, test_secondaries_data.shape[0])

        # TODO
        # eliminar es solo para debbugar el codigo
        # descomentar llamadas a modelos
        print('#####################################')
        print('################# principal')
        print('shape train: ', train_principal_data.shape)
        print('shape val: ', val_principal_data.shape)
        print('shape test: ', test_principal_data.shape)
        print('shape train label: ', train_principal_labels.shape)
        print('shape val label: ', val_principal_labels.shape)
        print('shape test label: ', test_principal_labels.shape)
        print('################# secundario')
        print('shape train: ', train_secondaries_data.shape)
        print('shape val: ', val_secondaries_data.shape)
        print('shape test: ', test_secondaries_data.shape)
        print('shape train label: ', train_secondaries_labels.shape)
        print('shape val label: ', val_secondaries_labels.shape)
        print('shape test label: ', test_secondaries_labels.shape)
        print('#####################################')

        '''
        if model_name == 0:
            print('comenzando modelo RANDOM FOREST')

        elif model_name == 1:
            print('comenzando modelo LSTM')
            dict_model_results = get_results_identification_lstm(train_principal_data, val_principal_data, test_principal_data,
                                        train_secondaries_data, val_secondaries_data, test_secondaries_data,
                                        train_principal_labels, val_principal_labels, test_principal_labels,
                                        train_secondaries_labels, val_secondaries_labels, test_secondaries_labels, window)
        elif model_name == 2:
            print('comenzando modelo MLP - ANN')
        

        identifier = ':'.join(list_group)
        dict_identification_results[identifier] = dict_model_results
        '''

    else:
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

            # Guardar solo los arrays de datos en archivos CSV
            np.savetxt(data_files['train_principal_data'], train_principal_data, delimiter=',')
            np.savetxt(data_files['val_principal_data'], val_principal_data, delimiter=',')
            np.savetxt(data_files['test_principal_data'], test_principal_data, delimiter=',')
            np.savetxt(data_files['train_secondaries_data'], train_secondaries_data, delimiter=',')
            np.savetxt(data_files['val_secondaries_data'], val_secondaries_data, delimiter=',')
            np.savetxt(data_files['test_secondaries_data'], test_secondaries_data, delimiter=',')
            
            print("Datos generados y guardados exitosamente. Generando etiquetas...")


            # TODO
            # eliminar es solo para debbugar el codigo
            # descomentar llamadas a modelos
            print('#####################################')
            print('################# principal')
            print('shape train: ', train_principal_data.shape)
            print('shape val: ', val_principal_data.shape)
            print('shape test: ', test_principal_data.shape)
            print('shape train label: ', train_principal_labels.shape)
            print('shape val label: ', val_principal_labels.shape)
            print('shape test label: ', test_principal_labels.shape)
            print('################# secundario')
            print('shape train: ', train_secondaries_data.shape)
            print('shape val: ', val_secondaries_data.shape)
            print('shape test: ', test_secondaries_data.shape)
            print('shape train label: ', train_secondaries_labels.shape)
            print('shape val label: ', val_secondaries_labels.shape)
            print('shape test label: ', test_secondaries_labels.shape)
            print('#####################################')


            '''
            if model_name == 0:
                print('comenzando modelo RANDOM FOREST')

            elif model_name == 1:
                print('comenzando modelo LSTM')
                dict_model_results = get_results_identification_lstm(train_principal_data, val_principal_data, test_principal_data,
                                            train_secondaries_data, val_secondaries_data, test_secondaries_data,
                                            train_principal_labels, val_principal_labels, test_principal_labels,
                                            train_secondaries_labels, val_secondaries_labels, test_secondaries_labels, window)
            elif model_name == 2:
                print('comenzando modelo MLP - ANN')
            

            identifier = ':'.join(list_group)
            dict_identification_results[identifier] = dict_model_results
            '''

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
            dict_model_results = real_time_identification(group, window, model_name)
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
        WINDOWS_SIZES_TO_ROUNDS, \
        EMPTYROOMS
        #WINDOWS_IDENTIFICATON

    #MODEL_NAMES = ['RF', 'LSTM', 'AE']
    SAMPLES = 500
    PARTICIPANTS = participantes
    PATH_PARTICIPANT_COMPLEX_CSV = 'D:\\ULima\\PosDoc\\code\\dataset_full_csv\\'
    PATH_EMPTYROOM_COMPLEX_CSV = 'D:\\ULima\\PosDoc\\code\\dataset_empty_csv\\'
    PARTICIPANTS_NUMBER = len(participantes)
    POSITIONS_PARTICIPANT = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
    RATIO = [80,20,25]
    EMPTYROOMS = 2125

    WINDOWS_SIZES_TO_ROUNDS = ['1','3','5','11','21','30','40','50','60']
    #WINDOWS_IDENTIFICATON = ['1','3','5','9','11','21']

if __name__ == "__main__":

    # Ejemplos de ejecucion de script
    # Para ejecutar solo la detección de presencia con LSTM (modelo 1):
    # python your_script.py --task presence --model 1
    # Para ejecutar solo la identificación con MLP (modelo 2):
    # python your_script.py --task identification --model 2
    # Para ejecutar ambas tareas con Random Forest (modelo 0):
    # python your_script.py --task all --model 0
    # Para ver la ayuda del script (mostrará las opciones de argumentos):
    # python your_script.py --help

    # 1. Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description="Ejecuta modelos para detección de presencia o identificación.")
    
    parser.add_argument('--task', type=str, required=True,
                        choices=['presence', 'identification', 'all'],
                        help="Define la tarea a ejecutar: 'presence', 'identification' o 'all'.")
    
    parser.add_argument('--model', type=int, default=1, # Valor por defecto como 1 (LSTM)
                        choices=[0, 1, 2],
                        help="Define el modelo a usar: 0=Random Forest, 1=LSTM, 2=MLP/ANN.")

    # 2. Parsear los argumentos de la línea de comandos
    args = parser.parse_args()

    # 3. Llamar a la configuración de variables globales del script
    config_scheme()

    # 4. Usar los argumentos para controlar el flujo
    if args.task == 'presence':
        print(f"Ejecutando solo la tarea de Detección de Presencia con modelo: {args.model}")
        presence_results(args.model)
    elif args.task == 'identification':
        print(f"Ejecutando solo la tarea de Identificación con modelo: {args.model}")
        identification_results(args.model)
    elif args.task == 'all':
        print(f"Ejecutando AMBAS tareas (Detección de Presencia e Identificación) con modelo: {args.model}")
        presence_results(args.model)
        identification_results(args.model)
    else:
        # Esto no debería ocurrir si usas 'choices' correctamente, pero es una buena práctica
        print("Tarea no reconocida. Por favor, elige 'presence', 'identification' o 'all'.")

    print("\nFinalizado con éxito")
