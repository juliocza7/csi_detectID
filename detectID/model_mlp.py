import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --- Varibles globales de modelo ---

INPUT_SIZE = 1
HIDDEN_SIZE = [128, 64, 32] # 64
OUTPUT_SIZE = 1

BATCH_SIZE = 32
EPOCHS = 100 # 10 ide, 100 pre

# --- Configuración de la GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


class PresenceMLP(nn.Module):
    def __init__(self, input_size_flattened, hidden_layers_sizes, output_size):
        super(PresenceMLP, self).__init__()
        layers = []

        # Capa de entrada a la primera capa oculta
        layers.append(nn.Linear(input_size_flattened, hidden_layers_sizes[0]))
        layers.append(nn.ReLU()) # Función de activación ReLU

        # Capas ocultas intermedias
        for i in range(len(hidden_layers_sizes) - 1):
            layers.append(nn.Linear(hidden_layers_sizes[i], hidden_layers_sizes[i+1]))
            layers.append(nn.ReLU()) # Función de activación ReLU

        # Capa de salida
        layers.append(nn.Linear(hidden_layers_sizes[-1], output_size))

        # Combinar todas las capas en un Sequential
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x llega con forma (batch_size, sequence_length, features_per_timestep)
        # Para MLP, necesitamos aplanarlo a (batch_size, sequence_length * features_per_timestep)
        batch_size = x.size(0)
        x = x.view(batch_size, -1) # El -1 aplana las dimensiones restantes
        return self.network(x)


def train_model(model, train_loader, val_data, val_labels, 
                criterion, optimizer, epochs, callbacks, 
                window_size, save_path='best_model.pth'):
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'precision': [],
                'val_precision': [], 'recall': [], 'val_recall': [], 'f1_score': [], 'val_f1_score': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    model.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        model.train() # Modo entrenamiento
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
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

        # --- Fase de Validación (por ventanas fijas) ---
        model.eval() # Modo evaluación
        val_losses = []
        val_accuracies = []
        val_precisions = []
        val_recalls = []
        val_f1_scores = []

        with torch.no_grad():
            num_val_samples = val_data.shape[0]
            for i in range(0, num_val_samples, window_size):
                if i + window_size <= num_val_samples: # Procesar solo ventanas completas
                    val_inputs_window = torch.tensor(val_data[i:i + window_size], dtype=torch.float32).to(device)
                    val_labels_window = torch.tensor(val_labels[i:i + window_size], dtype=torch.float32).unsqueeze(1).to(device)

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

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}")

        # Guardar el mejor modelo basado en la pérdida de validación
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict() # Guarda el estado (los pesos) del modelo
            torch.save(best_model_state, save_path) # Guarda el estado en un archivo
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
            if epoch > 0 and (epoch + 1) % callbacks['reduce_lr']['patience'] == 0:
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
    model.eval() # Poner el modelo en modo de evaluación
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    with torch.no_grad():
        num_test_samples = test_data.shape[0]
        for i in range(0, num_test_samples, window_size):
            if i + window_size <= num_test_samples: # Procesar solo ventanas completas
                # test_data ya debería tener la forma (batch, sequence, features)
                test_inputs_window = torch.tensor(test_data[i:i + window_size], dtype=torch.float32).to(device)
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

    if not losses:
        print("Warning: No complete windows found for evaluation.")
        return None, None, None, None, None

    final_loss = np.mean(losses)
    final_accuracy = np.mean(accuracies)
    final_precision = np.mean(precisions)
    final_recall = np.mean(recalls)
    final_f1 = np.mean(f1_scores)

    return final_loss, final_accuracy, final_precision, final_recall, final_f1


######################## funciones de presence detection ##############################
def presence_mlp_model(np_fullrooms_training, np_fullrooms_validate, np_fullrooms_test,
                       np_emptyrooms_training, np_emptyrooms_validate, np_emptyrooms_test,
                       targets_fullrooms_training, targets_fullrooms_validate, targets_fullrooms_test,
                       targets_emptyrooms_training, targets_emptyrooms_validate, targets_emptyrooms_test,
                       window_size_val): # Nuevos parámetros para MLP

    # 1. Concatenación de Datos
    train_data_np = np.concatenate((np_fullrooms_training, np_emptyrooms_training), axis=0)
    val_data_np = np.concatenate((np_fullrooms_validate, np_emptyrooms_validate), axis=0)

    train_labels_np = np.concatenate((targets_fullrooms_training, targets_emptyrooms_training)).astype(np.float32)
    val_labels_np = np.concatenate((targets_fullrooms_validate, targets_emptyrooms_validate)).astype(np.float32)

    # 2. Reshape de Datos para LSTM (batch_size, sequence_length, features_per_timestep)
    # Esta es la forma base, incluso para MLP, porque la capa forward de MLP la aplanará
    timesteps = train_data_np.shape[1] # Esto es la longitud de la secuencia (ej: 52)

    train_reshaped = train_data_np.reshape(train_data_np.shape[0], timesteps, INPUT_SIZE) # INPUT_SIZE = La característica por paso de tiempo (ej: 1)
    val_reshaped = val_data_np.reshape(val_data_np.shape[0], timesteps, INPUT_SIZE)

    # 3. Conversión a Tensores y DataLoader para ENTRENAMIENTO
    train_data_tensor = torch.tensor(train_reshaped, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels_np, dtype=torch.float32).unsqueeze(1) # unsqueeze para BCEWithLogitsLoss
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Definición del Modelo MLP
    #output_size = 1 # Para clasificación binaria
    #input_size_flattened = train_data_np.shape[1] * INPUT_SIZE
    input_size_flattened = timesteps * INPUT_SIZE
    model = PresenceMLP(input_size_flattened, HIDDEN_SIZE, OUTPUT_SIZE).to(device)

    # 5. Definición de la Función de Pérdida y el Optimizador
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. Configuración de Épocas y Callbacks
    epochs = 100
    callbacks = {
        'early_stopping': {'monitor': 'val_loss', 'patience': 10},
        'reduce_lr': {'monitor': 'val_loss', 'factor': 0.2, 'patience': 5, 'min_lr': 0.0001}
    }

    # 7. Ejecutar Entrenamiento con Validación por Ventanas Fijas
    history = train_model(model, train_loader, val_reshaped, val_labels_np, 
                                   criterion, optimizer, epochs, callbacks, 
                                   window_size=window_size_val, save_path='best_presence_mlp_model.pth')

    return model, history


def get_results_presence_mlp(np_fullrooms_training, np_fullrooms_validate, np_fullrooms_test,
                              np_emptyrooms_training, np_emptyrooms_validate, np_emptyrooms_test,
                              targets_fullrooms_training, targets_fullrooms_validate, targets_fullrooms_test,
                              targets_emptyrooms_training, targets_emptyrooms_validate, targets_emptyrooms_test,
                              window):

    model, history = presence_mlp_model(np_fullrooms_training, np_fullrooms_validate, np_fullrooms_test,
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
    input_size_flattened = test_data_np.shape[1] * INPUT_SIZE
    best_model = PresenceMLP(input_size_flattened, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    #best_model = PresenceLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    path_best_model = 'best_presence_mlp_model.pth'
    try:
        best_model.load_state_dict(torch.load(path_best_model, map_location=device))
        print(f"Modelo MLP cargado desde: {path_best_model}")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del mejor modelo MLP en {path_best_model}. Asegúrate de que el entrenamiento se completó y el archivo se guardó correctamente.")
        return None, None, None
    

    # 9. Evaluar el Mejor Modelo en el Conjunto de Prueba con Ventanas Fijas
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(
        best_model, test_reshaped, test_labels_np, criterion, window
    )

    results_dict_of_metrics['test'] = [test_loss, test_accuracy, test_precision, test_recall, test_f1]

    return results_dict_of_metrics