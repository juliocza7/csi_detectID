import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam


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

    # Convertir las etiquetas a codificación one-hot
    train_labels_one_hot = to_categorical(np.concatenate((train_principal_labels, train_secondaries_labels), axis=0))
    val_labels_one_hot = to_categorical(np.concatenate((val_principal_labels, val_secondaries_labels), axis=0))

    # Construir el Modelo LSTM
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(train_data_reshaped.shape[1], 1)))
    model.add(Dense(units=2, activation='softmax')) # capa de salida densa con softmax

    # Compilar el modelo
    model.compile(optimizer='adam', #Adam()
                  loss='categorical_crossentropy', #sparse_categorical_crossentropy
                  metrics=['accuracy', precision, recall, f1_score])

    # Resumen del modelo
    model.summary()

    # Entrenar el Modelo
    epochs = 10
    batch_size = 32

    history = model.fit(
        train_data_reshaped, train_labels_one_hot,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_data_reshaped, val_labels_one_hot)
    )
    
    return model, history, test_data_reshaped

# Evaluar el Modelo
def identification_evaluate_lstm_model_test(model, data, labels):
    predictions_one_hot = model.predict(data)
    predictions = np.argmax(predictions_one_hot, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    loss = model.evaluate(data, to_categorical(labels), verbose=0)[0]

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