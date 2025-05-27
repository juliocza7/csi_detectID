import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hampel import hampel
import time
#from filters import hampel_filter, moving_avg_filter


PATH = 'D:\\ULima\\PosDoc\\code\\dataset_full_csv\\001\\'
#PATH = 'D:\\ULima\\PosDoc\\code\\dataset_empty_csv\\'

def hampel_filter_complex(series):

    # 1. Calculate magnitude and phase
    magnitude = np.abs(series)
    phase = np.angle(series)

    # Convert to pandas Series (important for older hampel versions or strict type checks)
    #print('>>>> magnitude')
    magnitude_series = pd.Series(magnitude, index=series.index)
    #print('>>>> phase')
    phase_series = pd.Series(phase, index=series.index)

    # 2. Apply Hampel filter and access the filtered data from the returned object
    # ESTO ES LO QUE NECESITAS CONFIRMAR EN TU ARCHIVO HAMPL.PY:
    # ¿Es '.filtered_data'? ¿Es '.result'? ¿Otro nombre?
    hampel_result_mag = hampel(magnitude_series, window_size=31)
    hampel_result_phase = hampel(phase_series, window_size=31)

    # Suponiendo que el atributo se llama 'filtered_data' (¡confírmalo en tu hampel.py!)
    filtered_magnitude = hampel_result_mag.filtered_data
    filtered_phase = hampel_result_phase.filtered_data

    # Aseguramos que son arrays de numpy para la operación
    filtered_magnitude_np = np.asarray(filtered_magnitude)
    filtered_phase_np = np.asarray(filtered_phase)

    # 3. Combine filtered magnitude and phase back into complex numbers
    filtered_complex_array = filtered_magnitude_np * np.exp(1j * filtered_phase_np)

    # Convert the resulting NumPy array back to a Pandas Series with the original index
    
    return pd.Series(filtered_complex_array, index=series.index)


def moving_avg_filter_complex(series: pd.Series, window: int = 10) -> pd.Series:
    """
    Applies a moving average filter to the real and imaginary parts of a complex Pandas Series separately.

    Args:
        series (pd.Series): A Pandas Series of complex numbers.
        window (int, optional): The window size for the rolling average. Defaults to 10.

    Returns:
        pd.Series: A Pandas Series of complex numbers after applying the moving average.
    """
    # 1. Acceder a las partes real e imaginaria del array NumPy subyacente.
    #    'series.values' te da el array NumPy. Si es complejo, .real y .imag funcionarán.
    real_part_np = series.values.real
    imag_part_np = series.values.imag

    # 2. Convertir estas partes (que son arrays NumPy) de nuevo a Pandas Series
    #    para poder usar el método .rolling() de Pandas y mantener el índice original.
    real_part_pd = pd.Series(real_part_np, index=series.index)
    imag_part_pd = pd.Series(imag_part_np, index=series.index)

    # 3. Aplicar rolling mean a las partes real e imaginaria por separado
    real_mean = real_part_pd.rolling(window=window, min_periods=1, center=True).mean()
    imag_mean = imag_part_pd.rolling(window=window, min_periods=1, center=True).mean()

    # 4. Combinar las medias de nuevo en números complejos
    complex_mean = real_mean + 1j * imag_mean

    return complex_mean

def get_data_from_filters(df_csi_data):
    #df_csi_data = hampel_filter_complex(df_csi_data)
    #df_csi_data = moving_avg_filter(df_csi_data)

    '''
    # Apply Hampel filter to magnitude and phase separately for each subcarrier
    df_csi_data = df_csi_data.apply(hampel_filter_complex)

    # Apply moving average filter
    print('>>>> moving average')
    time.sleep(1)
    df_csi_data = df_csi_data.apply(
        lambda col: col.rolling(window=10, min_periods=1, center=True).mean()
    )
    '''

    df_csi_data = df_csi_data.apply(hampel_filter_complex)
    df_csi_data = df_csi_data.apply(moving_avg_filter_complex) # Usa la función corregida

    return df_csi_data

def get_selected_subcarries(df_csi_data):
    index_drop = [26] + list(range(54, 235))
    df_csi_data = df_csi_data.drop(df_csi_data.columns[index_drop], axis=1)

    return df_csi_data


def remove_multipath_delay():
    # --- Parámetros de Simulación ---
    num_subcarriers = 30  # Número de subportadoras OFDM
    num_samples = 500     # Número de muestras de tiempo (ej. cada 200 ms)
    sampling_frequency_csi = 1 / 0.0002 # Frecuencia de muestreo de CSI, si cada muestra es 200ms, entonces 1/0.2 = 5Hz.
                                    # El texto original habla de 2-5 Hz para movimiento de boca,
                                    # pero el IFFT/FFT opera sobre el dominio de las subportadoras para el retardo.
                                    # Para el IFFT, la "frecuencia de muestreo" en el dominio del tiempo
                                    # se relaciona con el ancho de banda del canal.
                                    # Usaremos una frecuencia de muestreo implícita para el dominio de retardo.

    # Umbral de retardo para eliminar multipath (en segundos, el paper usa 500 ns)
    delay_threshold_ns = 500
    delay_threshold_samples = int(delay_threshold_ns * (num_subcarriers / (20 * 10**6)) * 10**-9) # Ajuste a muestras de IFFT

    # --- Generación de datos de CSI simulados ---
    # Simulamos datos de CSI complejos.
    # La CSI real tiene variaciones de amplitud y fase debido al canal.
    # Aquí creamos un CSI base y le añadimos algunas "trayectorias" simuladas.

    # CSI base (ej. canal de línea de vista directa)
    csi_data = np.ones((num_samples, num_subcarriers), dtype=complex) * (1 + 0.1j)

    # Simular algunas trayectorias múltiples
    # Trayectoria 1: Retardo corto (útil, ~50 ns)
    delay_short_ns = 50
    amplitude_short = 0.5
    phase_short = np.pi / 4

    # Trayectoria 2: Retardo medio (útil, ~200 ns)
    delay_medium_ns = 200
    amplitude_medium = 0.3
    phase_medium = -np.pi / 2

    # Trayectoria 3: Retardo largo (a eliminar, ~600 ns)
    delay_long_ns = 600
    amplitude_long = 0.2
    phase_long = np.pi / 8

    # Función para añadir una trayectoria al CSI (simulación simplificada)
    def add_multipath_to_csi(csi, delay_ns, amplitude, phase, subcarrier_freqs_hz):
        # La fase de la trayectoria múltiple depende de la frecuencia y el retardo
        # H_mp(f) = A * e^(-j * 2 * pi * f * tau)
        tau = delay_ns * 1e-9  # Convertir nanosegundos a segundos
        for i in range(num_subcarriers):
            # f es la frecuencia de la subportadora (aquí simplificada como índice)
            # Esto es una simplificación; en la realidad, las frecuencias de las subportadoras están espaciadas.
            # Asumimos un ancho de banda total para el IFFT/FFT, por ejemplo 20 MHz para Wi-Fi.
            # Las frecuencias de las subportadoras se extenderían a lo largo de este ancho de banda.
            # Para la simulación, podemos usar el índice de la subportadora o una aproximación.
            # Usaremos el índice para un cálculo simple de fase.
            # Para una simulación más precisa, necesitarías las frecuencias exactas de las subportadoras.
            csi[:, i] += amplitude * np.exp(-1j * (2 * np.pi * subcarrier_freqs_hz[i] * tau + phase))
        return csi

    # Frecuencias simuladas para las subportadoras (simplificación)
    # En Wi-Fi, 30 subportadoras están dentro de un canal de 20 MHz
    # Podemos usar un rango de frecuencias para el IFFT.
    # La resolución del IFFT en tiempo (retardo) depende del ancho de banda total.
    # Si el ancho de banda es B y hay N subportadoras, la resolución temporal es 1/B.
    # Aquí usaremos un rango simplificado para demostración.
    subcarrier_spacing = 20e6 / num_subcarriers # Ancho de banda de 20 MHz dividido por 30 subportadoras
    subcarrier_freqs_hz = np.linspace(-10e6, 10e6, num_subcarriers) # Simula un rango de 20MHz centrado en 0

    csi_data = add_multipath_to_csi(csi_data, delay_short_ns, amplitude_short, phase_short, subcarrier_freqs_hz)
    csi_data = add_multipath_to_csi(csi_data, delay_medium_ns, amplitude_medium, phase_medium, subcarrier_freqs_hz)
    csi_data = add_multipath_to_csi(csi_data, delay_long_ns, amplitude_long, phase_long, subcarrier_freqs_hz)


    # --- Procesamiento de Multipath ---
    csi_processed = np.zeros_like(csi_data, dtype=complex)
    power_delay_profiles = [] # Para almacenar los perfiles de retardo antes de filtrar
    power_delay_profiles_filtered = [] # Para almacenar los perfiles de retardo después de filtrar

    # El eje de tiempo (retardo) para los perfiles de potencia
    # La resolución de retardo del IFFT está dada por 1 / Ancho_de_banda_total
    # Si consideramos un ancho de banda de 20 MHz para 30 subportadoras, la resolución es 1/20MHz = 50 ns
    # Entonces, el número de "bins" de retardo es igual al número de subportadoras para el IFFT.
    delay_resolution_ns = 1 / (max(subcarrier_freqs_hz) - min(subcarrier_freqs_hz) + subcarrier_spacing) * 1e9 # En ns
    if delay_resolution_ns == np.inf: # Evitar división por cero si num_subcarriers es muy bajo o frecuencias no bien definidas
        delay_resolution_ns = 50 # Un valor por defecto razonable para simulacion

    delay_axis_ns = np.arange(num_subcarriers) * delay_resolution_ns

    # Ajustar el umbral de retardo al número de bins de IFFT
    delay_threshold_bins = int(delay_threshold_ns / delay_resolution_ns)
    if delay_threshold_bins < 0:
        delay_threshold_bins = 0
    print(f"Resolución de retardo del IFFT: {delay_resolution_ns:.2f} ns/bin")
    print(f"Umbral de retardo: {delay_threshold_ns} ns, correspondiente a {delay_threshold_bins} bins.")


    for i in range(num_samples):
        # 1. Aplicar IFFT a la CSI de las subportadoras para obtener el perfil de retardo de potencia
        # np.fft.ifft(csi_data[i, :]) opera sobre la fila de CSI para una muestra de tiempo
        power_delay_profile = np.fft.ifft(csi_data[i, :])
        power_delay_profiles.append(np.abs(power_delay_profile)) # Guardar magnitud para visualización

        # 2. Eliminar componentes de multipath con retardo mayor al umbral
        # Esto implica poner a cero los bins en el perfil de retardo que exceden el umbral
        filtered_profile = power_delay_profile.copy()
        # Poner a cero los componentes de retardo más allá del umbral
        # Note: Para IFFT, los retardos se mapean simétricamente, pero los reales están en la primera mitad
        # Se eliminan los bins con índice mayor o igual al umbral
        filtered_profile[delay_threshold_bins:] = 0

        power_delay_profiles_filtered.append(np.abs(filtered_profile)) # Guardar magnitud filtrada

        # 3. Convertir el perfil de retardo filtrado de vuelta a CSI usando FFT
        csi_processed[i, :] = np.fft.fft(filtered_profile)

    # --- Visualización de Resultados ---

    plt.figure(figsize=(15, 10))

    # 1. Visualizar un Perfil de Retardo de Potencia (ejemplo de la primera muestra)
    plt.subplot(2, 2, 1)
    plt.plot(delay_axis_ns, power_delay_profiles[0], label='Original PDP')
    plt.axvline(x=delay_threshold_ns, color='r', linestyle='--', label=f'Umbral {delay_threshold_ns} ns')
    plt.title('Perfil de Retardo de Potencia (PDP) - Original')
    plt.xlabel('Retardo (ns)')
    plt.ylabel('Magnitud')
    plt.grid(True)
    plt.legend()

    # 2. Visualizar el Perfil de Retardo de Potencia Filtrado
    plt.subplot(2, 2, 2)
    plt.plot(delay_axis_ns, power_delay_profiles_filtered[0], label='Filtrado PDP')
    plt.axvline(x=delay_threshold_ns, color='r', linestyle='--', label=f'Umbral {delay_threshold_ns} ns')
    plt.title('Perfil de Retardo de Potencia (PDP) - Filtrado')
    plt.xlabel('Retardo (ns)')
    plt.ylabel('Magnitud')
    plt.grid(True)
    plt.legend()

    # 3. Visualizar la Magnitud de CSI de una subportadora antes y después del procesamiento
    plt.subplot(2, 2, 3)
    subcarrier_to_plot = 15 # Elegir una subportadora para ver su evolución
    plt.plot(np.abs(csi_data[:, subcarrier_to_plot]), label='CSI Original (Magnitud)')
    plt.plot(np.abs(csi_processed[:, subcarrier_to_plot]), label='CSI Procesado (Magnitud)')
    plt.title(f'Magnitud de CSI para la Subportadora {subcarrier_to_plot+1}')
    plt.xlabel('Muestra de Tiempo')
    plt.ylabel('Magnitud')
    plt.grid(True)
    plt.legend()

    # 4. Visualizar la Fase de CSI de una subportadora antes y después del procesamiento
    plt.subplot(2, 2, 4)
    plt.plot(np.angle(csi_data[:, subcarrier_to_plot]), label='CSI Original (Fase)')
    plt.plot(np.angle(csi_processed[:, subcarrier_to_plot]), label='CSI Procesado (Fase)')
    plt.title(f'Fase de CSI para la Subportadora {subcarrier_to_plot+1}')
    plt.xlabel('Muestra de Tiempo')
    plt.ylabel('Fase (radianes)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # --- Verificación del Impacto ---
    print("\n--- Impacto del Filtro en el CSI (ejemplo para la primera muestra) ---")
    print("CSI Original (magnitud) para la subportadora 0:", np.abs(csi_data[0, 0]))
    print("CSI Procesado (magnitud) para la subportadora 0:", np.abs(csi_processed[0, 0]))
    print("CSI Original (fase) para la subportadora 0:", np.angle(csi_data[0, 0]))
    print("CSI Procesado (fase) para la subportadora 0:", np.angle(csi_processed[0, 0]))

    # Para una verificación más cuantitativa, podrías comparar la desviación estándar
    # o la variación de las señales filtradas vs no filtradas.

if __name__ == "__main__":

    remove_multipath_delay()
    '''
    for i in range(1, 18):
        df_csi_data = pd.read_csv(PATH + str(i) + '.csv') # comentar si se usa datos csi del pcap
        
        #df_csi_data = get_selected_subcarries(df_csi_data)

        df_csi_data = df_csi_data.drop(['Unnamed: 0'], axis=1)
        df_csi_data = df_csi_data.apply(lambda col: col.apply(lambda val: complex(val.strip('()'))))

        df_csi_data = get_data_from_filters(df_csi_data)

        df_csi_data.to_csv('D:\\ULima\\PosDoc\\code\\full\\' + str(i) + '.csv', index=False)
        #df_csi_data.to_csv('D:\\ULima\\PosDoc\\code\\empty\\' + str(i) + '.csv', index=False)
    '''