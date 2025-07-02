import json
import matplotlib.pyplot as plt
import os
import numpy as np

# Cargar el archivo JSON desde una carpeta anterior
#file_path = os.path.join('..', 'presence_results_LSTM.json') #archivo esta una carpeta antes
file_path = os.path.join('C:\\Users\\jsoto\\code\\csi_detectID\\detectID\\processed_data\\presence_conCPS\\', 'presence_results_LSTM.json')
with open(file_path, 'r') as f:
    data = json.load(f)

# Convertir claves a enteros y ordenar
windows = sorted(map(int, data.keys()))

# Inicializar listas para las 5 métricas de cada conjunto
training_metrics = [[] for _ in range(5)]
validate_metrics = [[] for _ in range(5)]
test_metrics = [[] for _ in range(5)]

# Extraer valores
for w in windows:
    w_str = str(w)
    for i in range(5):
        training_metrics[i].append(data[w_str]['training'][i])
        validate_metrics[i].append(data[w_str]['validate'][i])
        test_metrics[i].append(data[w_str]['test'][i])

# Nombres de métricas (sin incluir Loss para los gráficos generales)
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# Función para graficar métricas (excluyendo Loss)
def plot_metrics(metrics, title):
    plt.figure(figsize=(12, 6))
    for i in range(1, 5):  # Desde índice 1 a 4 (excluye Loss)
        plt.plot(
            windows,
            metrics[i],
            marker='o',
            label=metric_names[i - 1],
            alpha=0.5
        )
    plt.title(f'{title} por Ventana de Tiempo')
    plt.xlabel('Ventana de Tiempo')
    plt.ylabel('Valor')
    plt.xticks(windows)  # Mostrar los valores de ventana como etiquetas
    plt.yticks(np.arange(0, 1.1 + 0.01, 0.1))  # ← Aquí defines el espaciado Y
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




# Graficar métricas sin Loss
plot_metrics(training_metrics, 'Training')
plot_metrics(validate_metrics, 'Validation')
plot_metrics(test_metrics, 'Test')

# Gráfico separado solo para Loss (métrica índice 0)
plt.figure(figsize=(10, 5))
plt.plot(windows, training_metrics[0], marker='o', label='Training Loss', color='blue', alpha=0.5)
plt.plot(windows, validate_metrics[0], marker='o', label='Validation Loss', color='orange', alpha=0.5)
plt.plot(windows, test_metrics[0], marker='o', label='Test Loss', color='green', alpha=0.5)
plt.title('Loss por Ventana de Tiempo')
plt.xlabel('Ventana de Tiempo')
plt.ylabel('Loss')
plt.xticks(windows)  # ← Mostrar los valores de ventana como etiquetas
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
