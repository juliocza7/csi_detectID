import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (actual_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))


##################################### PRECENSE DETECTION LSTM MODEL ################################
def presence_lstm_model(np_fullrooms_training, np_fullrooms_validate, np_fullrooms_test,
                            np_emptyrooms_training, np_emptyrooms_validate, np_emptyrooms_test,
                            targets_fullrooms_training, targets_fullrooms_validate, targets_fullrooms_test,
                            targets_emptyrooms_training, targets_emptyrooms_validate, targets_emptyrooms_test):


    train_data = np.concatenate((np_fullrooms_training, np_emptyrooms_training), axis=0)
    val_data = np.concatenate((np_fullrooms_validate, np_emptyrooms_validate), axis=0)
    test_data = np.concatenate((np_fullrooms_test, np_emptyrooms_test), axis=0)

    train_labels = np.concatenate((targets_fullrooms_training, targets_emptyrooms_training))
    val_labels = np.concatenate((targets_fullrooms_validate, targets_emptyrooms_validate))


    # Los LSTM esperan una entrada 3D: (num_samples, timesteps, features)
    # cada fila (de 52 columnas) es una secuencia de 52 "timesteps",
    # donde cada "timestep" tiene 1 característica (la amplitud de una subportadora).

    # Cada subportadora es un "timestep" y su amplitud es la característica (más común)
    # La forma se convierte en (num_samples, 52, 1)
    train_reshaped = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
    val_reshaped = val_data.reshape(val_data.shape[0], val_data.shape[1], 1)
    test_reshaped = test_data.reshape(test_data.shape[0], test_data[1], 1)


    model = Sequential([
        LSTM(units=5, input_shape=(train_reshaped.shape[1], train_reshaped.shape[2])),
        Dense(units=50, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), # Puedes especificar el learning rate aquí
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall', 'f1_score'])

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    history = model.fit(
        train_reshaped, train_labels,
        epochs=100,
        batch_size=32,
        validation_data=(val_reshaped, val_labels),
        callbacks=[early_stopping, reduce_lr]
    )

    return model, history, test_reshaped


def presence_evaluate_lstm_model_test(model, data, labels):
    probabilities = model.predict(data)
    predictions = (probabilities >= 0.5).astype(int).flatten()

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    loss = model.evaluate(data, labels, verbose=0)[0]

    return loss, accuracy, precision, recall, f1

def get_results_presence_lstm(np_fullrooms_training, np_fullrooms_validate, np_fullrooms_test,
                                np_emptyrooms_training, np_emptyrooms_validate, np_emptyrooms_test,
                                targets_fullrooms_training, targets_fullrooms_validate, targets_fullrooms_test,
                                targets_emptyrooms_training, targets_emptyrooms_validate, targets_emptyrooms_test):

    model, history, test_data_reshaped = presence_lstm_model(np_fullrooms_training, np_fullrooms_validate, np_fullrooms_test,
                                                    np_emptyrooms_training, np_emptyrooms_validate, np_emptyrooms_test,
                                                    targets_fullrooms_training, targets_fullrooms_validate, targets_fullrooms_test,
                                                    targets_emptyrooms_training, targets_emptyrooms_validate, targets_emptyrooms_test)

    model_metrics = history.history
    results_dict_of_metrics = {}

    # Training metrics: [loss, accuracy, precision, recall, f1]
    train_results = [
        model_metrics.get('loss')[-1] if 'loss' in model_metrics else None,
        model_metrics.get('accuracy')[-1] if 'accuracy' in model_metrics else None,
        model_metrics.get('precision')[-1] if 'precision' in model_metrics else None,
        model_metrics.get('recall')[-1] if 'recall' in model_metrics else None,
        model_metrics.get('f1_score')[-1] if 'f1_score' in model_metrics else None,
    ]
    results_dict_of_metrics['training'] = train_results

    # Validation metrics: [val_loss, val_accuracy, val_precision, val_recall, val_f1]
    val_results = [
        model_metrics.get('val_loss')[-1] if 'val_loss' in model_metrics else None,
        model_metrics.get('val_accuracy')[-1] if 'val_accuracy' in model_metrics else None,
        model_metrics.get('val_precision')[-1] if 'val_precision' in model_metrics else None,
        model_metrics.get('val_recall')[-1] if 'val_recall' in model_metrics else None,
        model_metrics.get('val_f1_score')[-1] if 'val_f1_score' in model_metrics else None,
    ]
    results_dict_of_metrics['validate'] = val_results

    # Evaluate on the test set to get test metrics including loss
    test_loss, test_accuracy, test_precision, test_recall, test_f1,  = presence_evaluate_lstm_model_test(
        model,
        test_data_reshaped,
        np.concatenate((targets_fullrooms_test, targets_emptyrooms_test), axis=0)
    )

    # Test metrics: [test_loss, test_accuracy, test_precision, test_recall, test_f1]
    test_results = [test_loss, test_accuracy, test_precision, test_recall, test_f1]
    results_dict_of_metrics['test'] = test_results

    return results_dict_of_metrics






###################################### IDENTIFICATION LSTM MODEL ##################################
def identification_lstm_model(train_principal_data, train_secondaries_data, test_principal_data,
                              train_principal_labels, train_secondaries_labels, test_secondaries_data,
                              val_principal_data, val_secondaries_data,
                              val_principal_labels, val_secondaries_labels):

    # Transponer los datos para que las subportadoras sean las "muestras"
    train_data_transposed = np.concatenate((train_principal_data, train_secondaries_data), axis=0).T
    val_data_transposed = np.concatenate((val_principal_data, val_secondaries_data), axis=0).T
    test_data_transposed = np.concatenate((test_principal_data, test_secondaries_data), axis=0).T

    # Reformar los datos transpuestos para la entrada de la LSTM: (muestras, pasos_de_tiempo, características)
    train_data_reshaped = np.reshape(train_data_transposed, (train_data_transposed.shape[0], train_data_transposed.shape[1], 1))
    val_data_reshaped = np.reshape(val_data_transposed, (val_data_transposed.shape[0], val_data_transposed.shape[1], 1))
    test_data_reshaped = np.reshape(test_data_transposed, (test_data_transposed.shape[0], test_data_transposed.shape[1], 1))

    # Convertir las etiquetas a codificación one-hot (aunque para binaria con sigmoid no es estrictamente necesario,
    # pero si tus etiquetas son 0 y 1, binary_crossentropy lo manejará directamente)
    train_labels = np.concatenate((train_principal_labels, train_secondaries_labels))
    val_labels = np.concatenate((val_principal_labels, val_secondaries_labels))

    # Construir el Modelo LSTM
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(train_data_reshaped.shape[1], 1)))
    model.add(Dense(units=1, activation='sigmoid')) # Capa de salida para clasificación binaria

    # Compilar el modelo
    model.compile(optimizer='adam', #Adam()
                  loss='binary_crossentropy',
                  metrics=['accuracy', precision, recall, f1_score])

    # Resumen del modelo
    model.summary()

    # Entrenar el Modelo
    epochs = 10
    batch_size = 32

    history = model.fit(
        train_data_reshaped, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_data_reshaped, val_labels)
    )

    return model, history, test_data_reshaped


# Evaluar el Modelo (ajustado para salida sigmoide)
def identification_evaluate_lstm_model_test(model, data, labels):
    probabilities = model.predict(data)
    predictions = (probabilities >= 0.5).astype(int).flatten()

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    loss = model.evaluate(data, labels, verbose=0)[0]

    return loss, accuracy, precision, recall, f1


def get_results_identification_lstm(train_principal_data, val_principal_data, test_principal_data,
                                        train_secondaries_data, val_secondaries_data, test_secondaries_data,
                                        train_principal_labels, val_principal_labels, test_principal_labels,
                                        train_secondaries_labels, val_secondaries_labels, test_secondaries_labels):
    
    model, history, test_data_reshaped = identification_lstm_model(train_principal_data, train_secondaries_data, test_principal_data,
                                                train_principal_labels, train_secondaries_labels, test_secondaries_data,
                                                val_principal_data, val_secondaries_data,
                                                val_principal_labels, val_secondaries_labels)

    model_metrics = history.history
    results_dict_of_metrics = {}

    # Training metrics: [loss, accuracy, precision, recall, f1]
    train_results = [
        model_metrics.get('loss')[-1] if 'loss' in model_metrics else None,
        model_metrics.get('accuracy')[-1] if 'accuracy' in model_metrics else None,
        model_metrics.get('precision')[-1] if 'precision' in model_metrics else None,
        model_metrics.get('recall')[-1] if 'recall' in model_metrics else None,
        model_metrics.get('f1_score')[-1] if 'f1_score' in model_metrics else None,
    ]
    results_dict_of_metrics['training'] = train_results

    # Validation metrics: [val_loss, val_accuracy, val_precision, val_recall, val_f1]
    val_results = [
        model_metrics.get('val_loss')[-1] if 'val_loss' in model_metrics else None,
        model_metrics.get('val_accuracy')[-1] if 'val_accuracy' in model_metrics else None,
        model_metrics.get('val_precision')[-1] if 'val_precision' in model_metrics else None,
        model_metrics.get('val_recall')[-1] if 'val_recall' in model_metrics else None,
        model_metrics.get('val_f1_score')[-1] if 'val_f1_score' in model_metrics else None,
    ]
    results_dict_of_metrics['validate'] = val_results

    # Evaluate on the test set to get test metrics including loss
    test_loss, test_accuracy, test_precision, test_recall, test_f1,  = identification_evaluate_lstm_model_test(
        model,
        test_data_reshaped,
        np.concatenate((test_principal_labels, test_secondaries_labels), axis=0)
    )

    # Test metrics: [test_loss, test_accuracy, test_precision, test_recall, test_f1]
    test_results = [test_loss, test_accuracy, test_precision, test_recall, test_f1]
    results_dict_of_metrics['test'] = test_results

    return results_dict_of_metrics