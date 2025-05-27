import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, lfilter, stft
from hampel import hampel
from sklearn.decomposition import PCA
from scipy.fftpack import fft, fftfreq
#from ssqueezepy import ssq_cwt, ssq_stft, extract_ridges
#from fastdtw import fastdtw
#from scipy.spatial.distance import euclidean
import random

#from dtaidistance import dtw
#from dtaidistance import dtw_visualisation as dtwvis


def iq_samples_abs(series):
    abs_series = {}
    for key in series.keys():
        for i in range(len(series[key])):
            valor = np.abs(series[key][i])
            if valor == 0:
                valor = -1 * 100
            else:
                valor = 20 * np.log10(valor)
            if key in abs_series:
                abs_series[key] = np.append(abs_series[key], valor)
            else:
                abs_series[key] = np.array(valor)

    abs_series = pd.DataFrame(abs_series)

    return abs_series

def iq_samples_angle(series):
    angle_series = {}
    for key in series.keys():
        for i in range(len(series[key])):
            valor = np.angle(series[key][i])

            if key in angle_series:
                angle_series[key] = np.append(angle_series[key], valor)
            else:
                angle_series[key] = np.array(valor)

    angle_series = pd.DataFrame(angle_series)

    return angle_series

def remove_avg(series):
    removed_avg = {}

    for key in series.keys():
        removed_avg[key] = series[key] - series[key].mean()

    removed_avg = pd.DataFrame(removed_avg)

    return removed_avg

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


def hampel_filter(series):
    filtered = {}
    for key in series.keys():
        filtered[key] = hampel(series[key], window_size=31, n=3, imputation=True)
    filtered = pd.DataFrame(filtered)

    return filtered

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


def moving_avg_filter(series):
    moving_avg = {}

    for key in series.keys():
        moving_avg[key] = series[key].rolling(window=10, min_periods=1, center=True).mean()

    moving_avg = pd.DataFrame(moving_avg)
    return moving_avg


def band_pass_filter(series):
    fs = 30.0 #7.64
    t = 1.0 / fs
    lowcut = 0.2
    highcut = 10.0 #0.4
    n = len(series)
    xf = np.linspace(0.0, 1.0 / (2.0 * t), n // 2)
    b, a = butter(5, [lowcut / (fs / 2), highcut / (fs / 2)], 'band', analog=False, output='ba')

    bandpass_samples_filter = {}
    for key in series.keys():
        bandpass_samples_filter[key] = lfilter(b, a, series[key])

    bandpass_samples_filter = pd.DataFrame(bandpass_samples_filter)

    return bandpass_samples_filter


def csi_pca(series):
    series = series.reset_index()
    for subcarrier in series.keys():
        for sample in range(len(series[subcarrier])):
            if np.isnan(series[subcarrier][sample]) or np.isinf(series[subcarrier][sample]):
                series[subcarrier][sample] = 0

    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(series)

    principal_components = pd.DataFrame(data=principal_components, columns=['PCA'])

    return principal_components


def n_max(n, series):
    maximos = {}
    for i in range(n):
        maior = -1
        f = -1
        for i in range(len(series)//2):
            if not i in maximos and series[i] >= maior:
                maior = series[i]
                f = i
        maximos[f] = maior

    return maximos


def csi_fft(series):
    # FFT
    yf = fft(series)
    xf = fftfreq(yf.size, 0.13)

    n_maximos = n_max(4, np.abs(yf))

    print("dicionario de maximos:")
    print(n_maximos)

    freqs = []
    for i in list(n_maximos.keys()):
        freqs.append(xf[i])
    print(freqs)
    freq = np.mean(freqs)

    print("\n\n\n\n")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Frequencia: ", freq, " taxa de respiracao: ", 60 * freq)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("\n\n\n\n")

    # rr = np.zeros(len(xf))
    # rr[maximo] = np.max(np.abs(yf))
    fig, ax = plt.subplots()
    plt.plot(xf, np.abs(yf))
    # plt.stem(xf, rr, linefmt='red')
    ax.set(xlabel='Frequencies (Hz)', ylabel='dB', title='FFT')
    plt.xlim(0, 1)
    plt.show()


def csi_sftf(series):
    f, t, zxx = stft(series, 1 / 0.13)
    amp = np.max(np.abs(zxx))
    plt.pcolormesh(t, f, np.abs(zxx), vmin=0, vmax=amp, shading='gouraud')
    plt.title('STFT')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(0, 0.7)
    plt.xlabel('Time [sec]')
    plt.show()

def plot(series, title):
    f, ax = plt.subplots()
    plt.plot(series)
    ax.set(xlabel='Samples', ylabel='dB', title=title)
    plt.show()


# def viz(x, tx, wx):
#     plt.imshow(np.abs(wx), aspect='auto')
#     plt.show()
#     plt.imshow(np.abs(tx), aspect='auto')
#     plt.show()


def viz(x, tf, ridge_idxs, yticks=None, ssq=False, transform='cwt', show_x=False, scale=1):
    if show_x:
        f, ax = plt.subplots()
        plt.plot(x)
        ax.set(xlabel="Time [samples]", ylabel="Signal Amplitude [A.U.]", title="x(t)")
        plt.show()

    f, ax = plt.subplots()

    ax.set(ylabel="Frequency scales [1/Hz]", title="abs({}{}) w/ ridge_idxs")
    if scale == 1:
        plt.ylim(50, 150)
    else:
        plt.ylim(0, 100)
    plt.imshow(np.abs(tf))
    plt.plot(ridge_idxs, 'r')
    plt.show()