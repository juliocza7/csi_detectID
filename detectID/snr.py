import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from filtros_old import iq_samples_abs

file_path = './data_example/lleno.csv'  # Reemplaza 'tu_archivo.csv' con el nombre de tu archivo CSV

try:
    df = pd.read_csv(file_path, index_col=0)  # Lee el CSV, asumiendo que la primera columna es el índice

    num_subcarriers = df.shape[1]
    num_samples = df.shape[0]

    def parse_complex_from_str(complex_str):
        try:
            # Evalúa la cadena como una expresión de Python (riesgoso con datos no confiables)
            return complex(eval(complex_str.replace('j', 'j.real').replace('(', '').replace(')', '')))
        except (ValueError, TypeError, NameError):
            return np.nan  # Devuelve NaN si no se puede parsear

    # Aplica la función para convertir las columnas a números complejos
    for col in df.columns:
        df[col] = df[col].apply(parse_complex_from_str)

    # Elimina las columnas donde el parsing falló y se llenaron de NaN
    df = df.dropna(axis=1, how='all')

    df = iq_samples_abs(df) # extrae amplitudes

    num_subcarriers = df.shape[1] # Actualiza el número de subportadoras

    # Calcular la potencia de la señal para cada subportadora (promedio sobre las muestras)
    signal_power = df.apply(lambda col: np.mean(np.abs(col)**2), axis=0)

    # Asumimos que el ruido tiene una potencia constante para todas las subportadoras (puedes ajustarlo)
    noise_power = 1e-6  # Un valor pequeño como ejemplo de potencia de ruido

    # Calcular la relación señal a ruido (SNR) en dB
    snr_db = 10 * np.log10(signal_power / noise_power)

    # Plotear las primeras 64 subportadoras en gráficos individuales
    plt.figure(figsize=(15, 10))
    for i in range(min(64, num_subcarriers)):
        plt.subplot(8, 8, i + 1)
        #plt.plot(range(num_samples), np.abs(df.iloc[:, i])) # Magnitud de la señal
        plt.plot(range(num_samples), df.iloc[:,i]) # Magnitud de la señal
        #plt.title(f'SP: {df.columns[i]} -- SNR: {snr_db.iloc[i]:.2f} dB', fontsize=8)
        plt.title(f'Subportadora: {df.columns[i]}', fontsize=8, pad=1)
        plt.xlabel('amostras', fontsize=6, labelpad=0)
        plt.ylabel('db', fontsize=6, labelpad=0)
        plt.xticks(fontsize=5)
        #plt.yticks(fontsize=2)
        plt.ylim(0,100)
        plt.xlim(0,500)
        # Modificación para labels del eje y de 10 en 10
        #max_y = np.ceil(np.max(np.abs(df.iloc[:, i]))) # Obtiene el valor máximo redondeado hacia arriba
        #plt.yticks(np.arange(0, max_y + 1, 10), fontsize=3.7)
        plt.yticks(np.arange(0, 101, 20), fontsize=4.5) # Usar el límite superior de ylim para las etiquetas
        plt.xticks(np.arange(0, 501, 100), fontsize=4.5) # Usar el límite superior de ylim para las etiquetas
        ax = plt.gca() # Obtener el objeto Axes actual
        ax.tick_params(axis='y', pad=1) # Ajustar el padding de las etiquetas del eje Y
        plt.grid(True)
    plt.tight_layout()
    #plt.suptitle('Magnitud de las Primeras 64 Subportadoras', fontsize=16, y=1.02)
    plt.show()

    # Plotear las subportadoras restantes en grupos de 64 (o menos si quedan menos)
    start_index = 64
    while start_index < num_subcarriers:
        end_index = min(start_index + 64, num_subcarriers)
        num_to_plot = end_index - start_index
        rows = int(np.ceil(np.sqrt(num_to_plot)))
        cols = int(np.ceil(num_to_plot / rows))

        plt.figure(figsize=(15, 10))
        for i in range(num_to_plot):
            subcarrier_index = start_index + i
            plt.subplot(rows, cols, i + 1)
            #plt.plot(range(num_samples), np.abs(df.iloc[:, subcarrier_index])) # Magnitud de la señal
            plt.plot(range(num_samples), df.iloc[:,subcarrier_index]) # Magnitud de la señal
            plt.title(f'SP: {df.columns[subcarrier_index]} -- SNR: {snr_db.iloc[subcarrier_index]:.2f} dB', fontsize=8)
            plt.xlabel('amostras', fontsize=6)
            plt.ylabel('db', fontsize=6)
            plt.xticks(fontsize=5)
            #plt.yticks(fontsize=5)
            plt.ylim(0,100)
            # Modificación para labels del eje y de 10 en 10
            #max_y = np.ceil(np.max(np.abs(df.iloc[:, i]))) # Obtiene el valor máximo redondeado hacia arriba
            #plt.yticks(np.arange(0, max_y + 1, 10), fontsize=3.7)
            plt.yticks(np.arange(0, 101, 10), fontsize=3.7) # Usar el límite superior de ylim para las etiquetas
            ax = plt.gca() # Obtener el objeto Axes actual
            ax.tick_params(axis='y', pad=1) # Ajustar el padding de las etiquetas del eje Y
            plt.grid(True)
        plt.tight_layout()
        #plt.suptitle(f'Magnitud de las Subportadoras {df.columns[start_index]} a {df.columns[end_index-1]}', fontsize=16, y=1.02)
        plt.show()
        start_index = end_index

except FileNotFoundError:
    print(f"El archivo '{file_path}' no fue encontrado.")
except Exception as e:
    print(f"Ocurrió un error al procesar el archivo CSV: {e}")