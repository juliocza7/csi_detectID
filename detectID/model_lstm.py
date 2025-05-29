import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --- Varibles globales de modelo ---

INPUT_SIZE = 1
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1

BATCH_SIZE = 32
EPOCHS = 100 # 10 ide, 100 pre

# --- Configuración de la GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


# Modelo LSTM para detección de presencia (Clasificación Binaria)
class PresenceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PresenceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, output_size)  

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out  # No sigmoid here, BCEWithLogitsLoss will handle it


# --- Modelo LSTM para identificación (Clasificación Binaria) ---
class IdentificationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IdentificationLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.sigmoid = nn.Sigmoid() # Sigmoid sera manejado por  BCEWithLogitsLoss

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        return self.fc(out) 
        #return self.sigmoid(self.fc(out)) # No es necesario sigmoid aqui


# --- 2. Funciones de Entrenamiento y Evaluación ---
def train_model(model, train_loader, val_data, val_labels, 
                criterion, optimizer, epochs, callbacks, 
                window_size=None, save_path='best_model.pth'):
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'precision': [], 'val_precision': [], 'recall': [], 'val_recall': [], 'f1_score': [], 'val_f1_score': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    model.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view_as(outputs))
            #loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float().squeeze(1).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = accuracy_score(all_train_labels, all_train_preds)
        epoch_precision = precision_score(all_train_labels, all_train_preds, zero_division=0)
        epoch_recall = recall_score(all_train_labels, all_train_preds, zero_division=0)
        epoch_f1_score = f1_score(all_train_labels, all_train_preds, zero_division=0)

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)
        history['precision'].append(epoch_precision)
        history['recall'].append(epoch_recall)
        history['f1_score'].append(epoch_f1_score)

        # validacion del modelo
        model.eval()
        val_losses = []
        val_accuracies = []
        val_precisions = []
        val_recalls = []
        val_f1_scores = []

        with torch.no_grad():
            num_val_samples = val_data.shape[0]
            indices = range(0, num_val_samples, window_size) if window_size and window_size > 0 else [0]
            step = window_size if window_size and window_size > 0 else num_val_samples

            for i in indices:
                end = min(i + step, num_val_samples)
                
                #val_inputs_window = torch.tensor(val_data[i:end], dtype=torch.float32).unsqueeze(1).to(device)
                #val_labels_window = torch.tensor(val_labels[i:end], dtype=torch.float32).unsqueeze(1).to(device)
                val_inputs_window = torch.tensor(val_data[i:end], dtype=torch.float32).to(device)
                val_labels_window = torch.tensor(val_labels[i:end], dtype=torch.float32).view(-1, 1).to(device)

                outputs_window = model(val_inputs_window)
                loss_window = criterion(outputs_window, val_labels_window)
                preds_window = (torch.sigmoid(outputs_window) >= 0.5).float().squeeze(1).cpu().numpy()
                labels_window = val_labels_window.squeeze(1).cpu().numpy()

                val_losses.append(loss_window.item())
                val_accuracies.append(accuracy_score(labels_window, preds_window))
                val_precisions.append(precision_score(labels_window, preds_window, zero_division=0))
                val_recalls.append(recall_score(labels_window, preds_window, zero_division=0))
                val_f1_scores.append(f1_score(labels_window, preds_window, zero_division=0))

        epoch_val_loss = np.mean(val_losses) if val_losses else 0.0
        epoch_val_accuracy = np.mean(val_accuracies) if val_accuracies else 0.0
        epoch_val_precision = np.mean(val_precisions) if val_precisions else 0.0
        epoch_val_recall = np.mean(val_recalls) if val_recalls else 0.0
        epoch_val_f1_score = np.mean(val_f1_scores) if val_f1_scores else 0.0

        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)
        history['val_precision'].append(epoch_val_precision)
        history['val_recall'].append(epoch_val_recall)
        history['val_f1_score'].append(epoch_val_f1_score)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}")

        # Guardar el mejor modelo basado en la pérdida de validación
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)
            print(f"Mejor modelo guardado con Val Loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Callbacks (Early Stopping)
        if 'early_stopping' in callbacks and callbacks['early_stopping'] is not None:
            if patience_counter >= callbacks['early_stopping']['patience']:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        # Callbacks (ReduceLROnPlateau - simple implementation)
        if 'reduce_lr' in callbacks and callbacks['reduce_lr'] is not None:
            if epoch > 0 and epoch % callbacks['reduce_lr']['patience'] == 0:
                old_lr = optimizer.param_groups[0]['lr']
                new_lr = old_lr * callbacks['reduce_lr']['factor']
                if new_lr >= callbacks['reduce_lr']['min_lr']:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
                else:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = callbacks['reduce_lr']['min_lr']

    return history



def evaluate_model(model, test_data, test_labels, criterion, window_size):
    model.eval()  # Poner el modelo en modo de evaluación
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    with torch.no_grad():
        num_test_samples = test_data.shape[0]
        for i in range(0, num_test_samples, window_size):
            if i + window_size <= num_test_samples:  # Procesar solo ventanas completas
                test_inputs_window = torch.tensor(test_data[i:i + window_size], dtype=torch.float32).unsqueeze(1).to(device) # Asumiendo (batch, sequence, features)
                test_labels_window = torch.tensor(test_labels[i:i + window_size], dtype=torch.float32).unsqueeze(1).to(device)

                outputs_window = model(test_inputs_window)
                loss_window = criterion(outputs_window, test_labels_window)
                preds_window = (torch.sigmoid(outputs_window) >= 0.5).float().squeeze(1).cpu().numpy()
                labels_window = test_labels_window.squeeze(1).cpu().numpy()

                losses.append(loss_window.item())
                accuracies.append(accuracy_score(labels_window, preds_window))
                precisions.append(precision_score(labels_window, preds_window, zero_division=0))
                recalls.append(recall_score(labels_window, preds_window, zero_division=0))
                f1_scores.append(f1_score(labels_window, preds_window, zero_division=0))

    if not losses:  # Si no hay ventanas completas, devolver None
        print("Warning: No complete windows found for evaluation.")
        return None, None, None, None, None

    final_loss = np.mean(losses)
    final_accuracy = np.mean(accuracies)
    final_precision = np.mean(precisions)
    final_recall = np.mean(recalls)
    final_f1 = np.mean(f1_scores)

    return final_loss, final_accuracy, final_precision, final_recall, final_f1
#################################################################################################################################






###########################################################
# --- Funcion Principal de Detección de Presencia ---
def presence_lstm_model(np_fullrooms_training, np_fullrooms_validate, np_fullrooms_test,
                        np_emptyrooms_training, np_emptyrooms_validate, np_emptyrooms_test,
                        targets_fullrooms_training, targets_fullrooms_validate, targets_fullrooms_test,
                        targets_emptyrooms_training, targets_emptyrooms_validate, targets_emptyrooms_test,
                        window):

    # 1. Concatenación de Datos
    train_data_np = np.concatenate((np_fullrooms_training, np_emptyrooms_training), axis=0)
    val_data_np = np.concatenate((np_fullrooms_validate, np_emptyrooms_validate), axis=0)
   
    train_labels_np = np.concatenate((targets_fullrooms_training, targets_emptyrooms_training)).astype(np.float32)
    val_labels_np = np.concatenate((targets_fullrooms_validate, targets_emptyrooms_validate)).astype(np.float32)
    
    # 2. Reshape de Datos para LSTM (batch_size, sequence_length, features)
    # Asumiendo que tus datos de entrada son (num_samples, timesteps), y INPUT_SIZE=1
    # timesteps sería la longitud de la secuencia para el LSTM
    timesteps = train_data_np.shape[1] # Esto sería la longitud de la secuencia para tu LSTM

    train_reshaped = train_data_np.reshape(train_data_np.shape[0], timesteps, INPUT_SIZE)
    val_reshaped = val_data_np.reshape(val_data_np.shape[0], timesteps, INPUT_SIZE)

    # 3. Conversión a Tensores y DataLoaders para ENTRENAMIENTO
    train_data_tensor = torch.tensor(train_reshaped, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels_np, dtype=torch.float32).unsqueeze(1) # unsqueeze para BCEWithLogitsLoss
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)

    #batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Definición del Modelo
    #HIDDEN_SIZE = 5 # Valor de ejemplo
    #OUTPUT_SIZE = 1
    model = PresenceLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)

    # 5. Definición de la Función de Pérdida y el Optimizador
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # 6. Configuración de Épocas y Callbacks
    #epochs = 100
    callbacks = {
        'early_stopping': {'monitor': 'val_loss', 'patience': 10},
        'reduce_lr': {'monitor': 'val_loss', 'factor': 0.2, 'patience': 5, 'min_lr': 0.0001}
    }


    # 7. Ejecutar Entrenamiento con Validación por Ventanas Fijas
    # IMPORTANTE: val_data_np y val_labels_np se pasan directamente a train_model_windowed
    # También se pasa el window_size
    window_size_val = 10 # Define el tamaño de la ventana para la validación
    history = train_model(model, train_loader, val_reshaped, val_labels_np, 
                          criterion, optimizer, EPOCHS, callbacks, 
                          window_size=window, save_path='best_presence_model_lstm.pth')


    return model, history

def get_results_presence_lstm(np_fullrooms_training, np_fullrooms_validate, np_fullrooms_test,
                              np_emptyrooms_training, np_emptyrooms_validate, np_emptyrooms_test,
                              targets_fullrooms_training, targets_fullrooms_validate, targets_fullrooms_test,
                              targets_emptyrooms_training, targets_emptyrooms_validate, targets_emptyrooms_test,
                              window):

    model, history = presence_lstm_model(np_fullrooms_training, np_fullrooms_validate, np_fullrooms_test,
                                                     np_emptyrooms_training, np_emptyrooms_validate, np_emptyrooms_test,
                                                     targets_fullrooms_training, targets_fullrooms_validate, targets_fullrooms_test,
                                                     targets_emptyrooms_training, targets_emptyrooms_validate, targets_emptyrooms_test,
                                                     window)

    results_dict_of_metrics = {}
    results_dict_of_metrics['training'] = [
        history['loss'][-1], history['accuracy'][-1], history['precision'][-1],
        history['recall'][-1], history['f1_score'][-1]
    ]
    results_dict_of_metrics['validate'] = [
        history['val_loss'][-1], history['val_accuracy'][-1], history['val_precision'][-1],
        history['val_recall'][-1], history['val_f1_score'][-1]
    ]

    # Preparar datos de prueba
    test_data_np = np.concatenate((np_fullrooms_test, np_emptyrooms_test), axis=0)
    test_labels_np = np.concatenate((targets_fullrooms_test, targets_emptyrooms_test)).astype(np.float32)
    test_reshaped = test_data_np.reshape(test_data_np.shape[0], test_data_np.shape[1], INPUT_SIZE)

    # 8. Cargar el Mejor Modelo Guardado para la Evaluación Final
    # Reinstancia el modelo con la misma arquitectura
    best_model = PresenceLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    path_best_model = 'best_presence_model_lstm.pth' # Debe coincidir con save_path en train_model_windowed

    try:
        best_model.load_state_dict(torch.load(path_best_model, map_location=device))
        print(f"Modelo cargado desde: {path_best_model}")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del mejor modelo en {path_best_model}. Asegúrate de que el entrenamiento se completó y el archivo se guardó correctamente.")
        return None, None, None # O manejar el error de otra manera
    

    # 9. Evaluar el Mejor Modelo en el Conjunto de Prueba con Ventanas Fijas
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(
        best_model, test_reshaped, test_labels_np, criterion, window
    )

    results_dict_of_metrics['test'] = [test_loss, test_accuracy, test_precision, test_recall, test_f1]

    return results_dict_of_metrics



###########################################################
# --- Función Principal del Modelo de Identificación ---
def identification_lstm_model(train_principal_data, train_secondaries_data, test_principal_data,
                                    train_principal_labels, train_secondaries_labels, test_secondaries_data,
                                    val_principal_data, val_secondaries_data,
                                    val_principal_labels, val_secondaries_labels, window):

    # Transponer los datos para que las subportadoras sean las "muestras"
    train_data_transposed = np.concatenate((train_principal_data, train_secondaries_data), axis=0).T
    val_data_transposed = np.concatenate((val_principal_data, val_secondaries_data), axis=0).T
    #test_data_transposed = np.concatenate((test_principal_data, test_secondaries_data), axis=0).T

    # Reformar los datos transpuestos para la entrada de la LSTM: (muestras, pasos_de_tiempo, características)
    train_data_reshaped = train_data_transposed.reshape(train_data_transposed.shape[0], train_data_transposed.shape[1], 1)
    val_data_reshaped = val_data_transposed.reshape(val_data_transposed.shape[0], val_data_transposed.shape[1], 1)
    #test_data_reshaped = test_data_transposed.reshape(test_data_transposed.shape[0], test_data_transposed.shape[1], 1)

    # Convertir las etiquetas a tensores de PyTorch
    train_labels = np.concatenate((train_principal_labels, train_secondaries_labels)).astype(np.float32)
    val_labels = np.concatenate((val_principal_labels, val_secondaries_labels)).astype(np.float32)

    train_data_tensor = torch.tensor(train_data_reshaped, dtype=torch.float32).to(device)
    val_data_tensor = torch.tensor(val_data_reshaped, dtype=torch.float32).to(device)
    #test_data_tensor = torch.tensor(test_data_reshaped, dtype=torch.float32).to(device)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1).to(device)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1).to(device)

    # Crear DataLoaders
    #batch_size = 32
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    #val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Construir el Modelo LSTM
    #input_size = 1
    #hidden_size = 64
    #output_size = 1
    model = IdentificationLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)

    # Definir la función de pérdida y el optimizador
    criterion = nn.BCEWithLogitsLoss() #nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenar el Modelo
    #epochs = 10

    # Definir callbacks
    callbacks = {
        'early_stopping': {'patience': 5},
        'reduce_lr': {'factor': 0.5, 'patience': 3, 'min_lr': 1e-5}
    }

    model, history = train_model(model, train_loader, val_data_reshaped, val_labels, 
                                 criterion, optimizer, EPOCHS, callbacks, 
                                 window, 'best_identification_model_lstm') # model, train_loader, val_loader, criterion, optimizer, epochs)

    return model, history


def get_results_identification_lstm(train_principal_data, val_principal_data, test_principal_data,
                                        train_secondaries_data, val_secondaries_data, test_secondaries_data,
                                        train_principal_labels, val_principal_labels, test_principal_labels,
                                        train_secondaries_labels, val_secondaries_labels, test_secondaries_labels, window):

    model, history = identification_lstm_model(
        train_principal_data, train_secondaries_data, test_principal_data,
        train_principal_labels, train_secondaries_labels, test_secondaries_data,
        val_principal_data, val_secondaries_data,
        val_principal_labels, val_secondaries_labels, window)

    results_dict_of_metrics = {}

    # Métricas de entrenamiento
    results_dict_of_metrics['training'] = [
        history['loss'][-1], history['accuracy'][-1], history['precision'][-1],
        history['recall'][-1], history['f1_score'][-1]
    ]

    # Métricas de validación
    results_dict_of_metrics['validate'] = [
        history['val_loss'][-1], history['val_accuracy'][-1], history['val_precision'][-1],
        history['val_recall'][-1], history['val_f1_score'][-1]
    ]

    # Preparar datos de prueba
    test_data = np.concatenate((test_principal_data, test_secondaries_data), axis=0).T
    test_labels = np.concatenate((test_principal_labels, test_secondaries_labels)).astype(np.float32)
    #test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1).to(device)
    #test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
    #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Asegúrate de que estas variables estén definidas con los mismos valores y luego carga el mejor modelo
    best_model = IdentificationLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    path_best_model = 'best_identification_model_lstm.pth'
    best_model_state = torch.load(path_best_model)
    best_model.load_state_dict(best_model_state)
 

    # Evaluar en el conjunto de prueba
    criterion = nn.BCEWithLogitsLoss() #nn.BCELoss()
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(
        best_model, test_data, test_labels, criterion, window
    )

    # Métricas de prueba
    results_dict_of_metrics['test'] = [test_loss, test_accuracy, test_precision, test_recall, test_f1]

    return results_dict_of_metrics