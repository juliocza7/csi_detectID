import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --- Varibles globales de modelo ---

INPUT_SIZE = 52
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1

BATCH_SIZE = 64
EPOCHS = 100 # 10 ide, 100 pre

# --- Configuración de la GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Función para crear ventanas deslizantes
def create_sliding_windows(data, labels, window_size, stride=1):
    X, y = [], []
    for i in range(0, len(data) - window_size + 1, stride):
        X.append(data[i:i+window_size, :])                # (window_size, 52)
        y.append(labels[i + window_size - 1])             # etiqueta del último paso
    return np.array(X), np.array(y)

#################################################################################################################################################
######################################## PRESENCE DETECTION######################################################################################
# Modelo LSTM para detección de presencia (Clasificación Binaria)   
class PresenceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PresenceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        #out = self.relu(self.fc1(out))
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --- 2. Funciones de Entrenamiento y Evaluación ---
def train_model_presence(model, train_loader, val_loader, 
                         criterion, optimizer, epochs, callbacks, 
                         save_path='best_model.pth'):

    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 
               'precision': [], 'val_precision': [], 
               'recall': [], 'val_recall': [], 
               'f1_score': [], 'val_f1_score': []}
    
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
            loss = criterion(outputs, labels.float())
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

        # VALIDACIÓN
        val_loss, val_accuracy, val_precision, val_recall, val_f1_score = evaluate_model_presence(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1_score'].append(val_f1_score)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if 'early_stopping' in callbacks and patience_counter >= callbacks['early_stopping']['patience']:
            print("Early stopping triggered.")
            break

    return model, history

def evaluate_model_presence(model, data_loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    losses = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            losses.append(loss.item() * inputs.size(0))

            preds = (torch.sigmoid(outputs) >= 0.5).float().squeeze(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = np.sum(losses) / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return avg_loss, accuracy, precision, recall, f1



# --- Funcion Principal de Detección de Presencia ---
def presence_lstm_model(np_fullrooms_training, np_fullrooms_validate, np_fullrooms_test,
                        np_emptyrooms_training, np_emptyrooms_validate, np_emptyrooms_test,
                        targets_fullrooms_training, targets_fullrooms_validate, targets_fullrooms_test,
                        targets_emptyrooms_training, targets_emptyrooms_validate, targets_emptyrooms_test,
                        window):

    # 1. Concatenar datos llenos y vacíos
    train_data_np = np.concatenate((np_fullrooms_training, np_emptyrooms_training), axis=0)
    val_data_np = np.concatenate((np_fullrooms_validate, np_emptyrooms_validate), axis=0)
    # test_data_np = np.concatenate((np_fullrooms_test, np_emptyrooms_test), axis=0)  # si quieres test luego

    train_labels_np = np.concatenate((targets_fullrooms_training, targets_emptyrooms_training)).astype(np.float32)
    val_labels_np = np.concatenate((targets_fullrooms_validate, targets_emptyrooms_validate)).astype(np.float32)
    # test_labels_np = np.concatenate((targets_fullrooms_test, targets_emptyrooms_test)).astype(np.float32)

    # 2. Crear ventanas deslizantes para secuencias temporales
    X_train, y_train = create_sliding_windows(train_data_np, train_labels_np, window)
    X_val, y_val = create_sliding_windows(val_data_np, val_labels_np, window)
    # X_test, y_test = create_sliding_windows(test_data_np, test_labels_np, window)

    # 3. Convertir a tensores
    train_data_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_labels_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    val_data_tensor = torch.tensor(X_val, dtype=torch.float32)
    val_labels_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    # test_data_tensor = torch.tensor(X_test, dtype=torch.float32)
    # test_labels_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # 4. Crear DataLoader para entrenamiento y validación
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Definir modelo, criterio y optimizador
    model = PresenceLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Callbacks para early stopping y reducción de lr
    callbacks = {
        'early_stopping': {'monitor': 'val_loss', 'patience': 100},
        'reduce_lr': {'monitor': 'val_loss', 'factor': 0.2, 'patience': 10, 'min_lr': 1e-4}
    }

    # 6. Entrenamiento usando train_loader y val_loader
    model, history = train_model_presence(model, train_loader, val_loader, 
                                   criterion, optimizer, EPOCHS, callbacks, 
                                   save_path='best_presence_model_lstm.pth')

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
                                                     window=window)

    results_dict_of_metrics = {}

    results_dict_of_metrics['training'] = [
    float(history['loss'][-1]),
    float(history['accuracy'][-1]),
    float(history['precision'][-1]),
    float(history['recall'][-1]),
    float(history['f1_score'][-1])
    ]

    results_dict_of_metrics['validate'] = [
    float(history['val_loss'][-1]),
    float(history['val_accuracy'][-1]),
    float(history['val_precision'][-1]),
    float(history['val_recall'][-1]),
    float(history['val_f1_score'][-1])
    ]


    # Cargar el mejor modelo
    best_model = PresenceLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    path_best_model = 'best_presence_model_lstm.pth'
    best_model.load_state_dict(torch.load(path_best_model, map_location=device))
    best_model.eval()
    

    # Preparar datos de prueba
    test_data_np = np.concatenate((np_fullrooms_test, np_emptyrooms_test), axis=0)
    test_labels_np = np.concatenate((targets_fullrooms_test, targets_emptyrooms_test)).astype(np.float32)

    # Crear ventanas deslizantes para test
    X_test, y_test = create_sliding_windows(test_data_np, test_labels_np, window)
    test_data_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_labels_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluar en test_loader
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model_presence(best_model, test_loader, criterion)

    results_dict_of_metrics['test'] = [
    float(test_loss),
    float(test_accuracy),
    float(test_precision),
    float(test_recall),
    float(test_f1)
    ]

    return results_dict_of_metrics



#################################################################################################################################################
######################################## IDENTIFICACION #########################################################################################

# --- Modelo LSTM para identificación (Clasificación Binaria) ---
'''
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
'''

class IdentificationLSTM(nn.Module):
    def __init__(self, input_size=52, hidden_size=64, output_size=1):
        super(IdentificationLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)             # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]               # tomar el último paso temporal
        out = self.fc(out)                # shape: (batch, 1)
        return out

# --- 2. Funciones de Entrenamiento y Evaluación ---
def train_model_identification(model, train_loader, val_loader, 
                 criterion, optimizer, epochs, callbacks, 
                 save_path='best_model.pth'):
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [],
               'precision': [], 'val_precision': [], 'recall': [], 'val_recall': [],
               'f1_score': [], 'val_f1_score': []}
    
    best_val_loss = float('inf')
    patience_counter = 0
    model.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_train_preds, all_train_labels = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float().cpu().numpy()
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

        # ----------- VALIDACIÓN -------------
        model.eval()
        val_losses, all_val_preds, all_val_labels = [], [], []

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                outputs = model(val_inputs)
                loss = criterion(outputs, val_labels)
                preds = (torch.sigmoid(outputs) >= 0.5).float().cpu().numpy()
                labels = val_labels.cpu().numpy()

                val_losses.append(loss.item())
                all_val_preds.extend(preds)
                all_val_labels.extend(labels)

        epoch_val_loss = np.mean(val_losses)
        epoch_val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        epoch_val_precision = precision_score(all_val_labels, all_val_preds, zero_division=0)
        epoch_val_recall = recall_score(all_val_labels, all_val_preds, zero_division=0)
        epoch_val_f1_score = f1_score(all_val_labels, all_val_preds, zero_division=0)

        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)
        history['val_precision'].append(epoch_val_precision)
        history['val_recall'].append(epoch_val_recall)
        history['val_f1_score'].append(epoch_val_f1_score)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}")

        # Guardar el mejor modelo
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)
            print(f"Mejor modelo guardado con Val Loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if 'early_stopping' in callbacks and patience_counter >= callbacks['early_stopping']['patience']:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        if 'reduce_lr' in callbacks and epoch > 0 and epoch % callbacks['reduce_lr']['patience'] == 0:
            old_lr = optimizer.param_groups[0]['lr']
            new_lr = max(old_lr * callbacks['reduce_lr']['factor'], callbacks['reduce_lr']['min_lr'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")

    return model, history




def evaluate_model_identification(model, test_dataset, criterion, batch_size=64):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            preds = (torch.sigmoid(outputs) >= 0.5).float().cpu().numpy()
            labels = labels.cpu().numpy()

            losses.append(loss.item())
            all_preds.extend(preds)
            all_labels.extend(labels)

    final_loss = np.mean(losses)
    final_accuracy = accuracy_score(all_labels, all_preds)
    final_precision = precision_score(all_labels, all_preds, zero_division=0)
    final_recall = recall_score(all_labels, all_preds, zero_division=0)
    final_f1 = f1_score(all_labels, all_preds, zero_division=0)

    return final_loss, final_accuracy, final_precision, final_recall, final_f1


def identification_lstm_model(train_principal_data, train_secondaries_data, test_principal_data,
                              train_principal_labels, train_secondaries_labels, test_secondaries_data,
                              val_principal_data, val_secondaries_data,
                              val_principal_labels, val_secondaries_labels, window):

    # Concatenar datos principal y secundarios verticalmente (en filas)
    train_data = np.concatenate((train_principal_data, train_secondaries_data), axis=0)  # shape (10880, 52)
    val_data = np.concatenate((val_principal_data, val_secondaries_data), axis=0)        # shape (x_val, 52)

    # Concatenar etiquetas igual que los datos
    train_labels = np.concatenate((train_principal_labels, train_secondaries_labels)).astype(np.float32)  # (10880,)
    val_labels = np.concatenate((val_principal_labels, val_secondaries_labels)).astype(np.float32)


    stride = 1
    # Crear ventanas para train y val
    X_train, y_train = create_sliding_windows(train_data, train_labels, window, stride)
    X_val, y_val = create_sliding_windows(val_data, val_labels, window, stride)

    # Convertir a tensores y pasar a dispositivo
    train_data_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    train_labels_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    val_data_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_labels_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    # Crear DataLoaders
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Construir el Modelo LSTM (INPUT_SIZE debe ser 52)
    model = IdentificationLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    #model = IdentificationLSTM(input_size=52, hidden_size=64, output_size=1).to(device)

    # Función pérdida y optimizador
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Callbacks
    callbacks = {
        'early_stopping': {'patience': 100},
        'reduce_lr': {'factor': 0.5, 'patience': 10, 'min_lr': 1e-5}
    }

    # Entrenar modelo (asegúrate que train_model acepte val_loader en vez de val_data y val_labels separados)
    model, history = train_model_identification(model, train_loader, val_loader, 
                                 criterion, optimizer, EPOCHS, callbacks,
                                 save_path='best_identification_model_lstm.pth')

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
    float(history['loss'][-1]),
    float(history['accuracy'][-1]),
    float(history['precision'][-1]),
    float(history['recall'][-1]),
    float(history['f1_score'][-1])
    ]


    # Métricas de validación
    results_dict_of_metrics['validate'] = [
    float(history['val_loss'][-1]),
    float(history['val_accuracy'][-1]),
    float(history['val_precision'][-1]),
    float(history['val_recall'][-1]),
    float(history['val_f1_score'][-1])
    ]


   

    # Asegúrate de que estas variables estén definidas con los mismos valores y luego carga el mejor modelo
    best_model = IdentificationLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    path_best_model = 'best_identification_model_lstm.pth'
    best_model_state = torch.load(path_best_model)
    best_model.load_state_dict(best_model_state)
 

    # Preparar datos para test
    test_data = np.concatenate((test_principal_data, test_secondaries_data), axis=0)
    test_labels = np.concatenate((test_principal_labels, test_secondaries_labels), axis=0)

    stride = 1
    # Crear ventanas
    X_test, y_test = create_sliding_windows(test_data, test_labels, window, stride)

    # Convertir a tensor y dataset
    test_data_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_labels_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)

    # Evaluar en el conjunto de prueba
    criterion = nn.BCEWithLogitsLoss() #nn.BCELoss()
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model_identification(
        best_model, test_dataset, criterion, BATCH_SIZE)

    # Métricas de test
    results_dict_of_metrics['test'] = [
    float(test_loss),
    float(test_accuracy),
    float(test_precision),
    float(test_recall),
    float(test_f1)
    ]

    return results_dict_of_metrics