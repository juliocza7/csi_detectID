import pandas as pd
import matplotlib.pyplot as plt

import torch

print("PyTorch versión:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("Versión CUDA en PyTorch:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No se detectó GPU compatible con CUDA")


print(">>>>>\nCUDA disponible:", torch.cuda.is_available())
print("Versión de CUDA en PyTorch:", torch.version.cuda)
print("GPU detectada:", torch.cuda.get_device_name(0))


# Carga los archivos CSV sin encabezado
archivo1 = pd.read_csv(
    'C:\\Users\\jsoto\\code\\csi_detectID\\detectID\\processed_data\\presence\\fullrooms_training.csv',
    header=None
)
archivo2 = pd.read_csv(
    'C:\\Users\\jsoto\\code\\csi_detectID\\detectID\\processed_data\\presence\\emptyrooms_training.csv',
    header=None
)

# Crear la figura con 2 subplots en una misma fila
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

# Graficar el primer archivo
for col in archivo1.columns:
    axes[0].plot(archivo1.index, archivo1[col], linewidth=0.8)
axes[0].set_title('Subportadoras - Archivo 1')
axes[0].set_xlabel('Tiempo (muestras)')
axes[0].set_ylabel('Amplitud')
axes[0].grid(True)

# Graficar el segundo archivo
for col in archivo2.columns:
    axes[1].plot(archivo2.index, archivo2[col], linewidth=0.8)
axes[1].set_title('Subportadoras - Archivo 2')
axes[1].set_xlabel('Tiempo (muestras)')
axes[1].grid(True)

# Mostrar ticks del eje Y en el segundo gráfico también
axes[1].yaxis.set_tick_params(labelleft=True)

# Ajustar diseño y mostrar
plt.tight_layout()
plt.show()