import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import seaborn as sns
from scipy.stats import entropy
from scipy.stats import zscore
from sklearn.decomposition import PCA
from detectID import get_selected_subcarries
from filters import hampel_filter, moving_avg_filter, iq_samples_abs


def calcular_outliers_zscore(df, umbral=3):

    z_scores = df.apply(zscore)
    z_scores_abs = z_scores.abs()
    outliers = (z_scores_abs > 3)
    porcentaje_outliers = outliers.sum().sum() / df.size * 100

    return float(porcentaje_outliers)

def cargar_archivos(path, num_archivos=17, modo='primero'):
    """
    Carga y procesa archivos CSV desde una carpeta.

    Parámetros:
        path (str): Ruta a la carpeta con archivos CSV.
        num_archivos (int): Número de archivos a cargar.
        modo (str): 'primero', 'ultimo' o 'aleatorio'.

    Retorna:
        np.array con forma (num_archivos, muestras, 51)
    """
    archivos = [
        os.path.join(path, f) for f in os.listdir(path)
        if f.endswith('.csv')
    ]
    archivos = sorted(archivos)  # ordenar alfabéticamente

    if modo == 'primero':
        archivos_csv = archivos[:num_archivos]
    elif modo == 'ultimo':
        archivos_csv = archivos[-num_archivos:]
    elif modo == 'aleatorio':
        if num_archivos > len(archivos):
            raise ValueError("num_archivos excede la cantidad de archivos disponibles")
        archivos_csv = random.sample(archivos, num_archivos)
    else:
        raise ValueError("El parámetro 'modo' debe ser 'primero', 'ultimo' o 'aleatorio'.")

    señales = []

    print('>>>>>>>> archivos nombres: ', [os.path.splitext(os.path.basename(f))[0] for f in archivos_csv])

    #lista_porcentajes = []

    for archivo in archivos_csv:
        df = pd.read_csv(archivo)

        # Filtrar subportadoras (usa tu propia lógica)
        #df = get_selected_subcarries(df)  # Devuelve 51 columnas

        # Eliminar columna innecesaria si existe
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        # Convertir strings a números complejos
        df = df.apply(lambda col: col.apply(lambda val: complex(val.strip('()')) if isinstance(val, str) else val))

        df = iq_samples_abs(df)
        #lista_porcentajes.append(calcular_outliers_zscore(df))
        #df = hampel_filter(df)     
        #df = moving_avg_filter(df)

        # Guardar como matriz real de amplitudes
        #señal = np.abs(df.values)  # shape: (muestras, 51)
        señales.append(df.values)

    #print('lista de % de outliers: ', lista_porcentajes)

    return np.array(señales)  # shape: (num_archivos, muestras, 51)

def graficar_comparativa_17_archivos(señales_full, señales_empty, titulo_full='Salas Llenas', titulo_empty='Salas Vacías'):
    fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharey=True)

    # Primer gráfico: señales con persona
    for i in range(len(señales_full)):
        promedio = np.mean(señales_full[i], axis=0)
        axes[0].plot(promedio, label=f'Archivo {i+1}')
    axes[0].set_title(titulo_full)
    axes[0].set_xlabel('Subportadora')
    axes[0].set_ylabel('Amplitud promedio')
    axes[0].grid(True)
    axes[0].legend(fontsize='small')

    # Segundo gráfico: señales sin persona
    for i in range(len(señales_empty)):
        promedio = np.mean(señales_empty[i], axis=0)
        axes[1].plot(promedio, label=f'Archivo {i+1}')
    axes[1].set_title(titulo_empty)
    axes[1].set_xlabel('Subportadora')
    axes[1].grid(True)
    axes[1].legend(fontsize='small')

    plt.suptitle('Comparativa de Promedios por Subportadora entre Salas Llenas y Vacías', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def graficar_diferencia_promedio(salas_llenas, salas_vacias):
    prom_llenas = np.mean(salas_llenas, axis=(0, 1))  # promedio global por subportadora
    prom_vacias = np.mean(salas_vacias, axis=(0, 1))
    diferencia = prom_llenas - prom_vacias

    plt.figure(figsize=(12, 6))
    plt.plot(prom_llenas, label='Salas llenas', linewidth=2)
    plt.plot(prom_vacias, label='Salas vacías', linewidth=2)
    plt.plot(diferencia, label='Diferencia', linestyle='--', color='black')
    plt.title('Comparación de amplitud promedio por subportadora')
    plt.xlabel('Subportadora')
    plt.ylabel('Amplitud promedio')
    plt.legend()
    plt.grid(True)
    plt.show()

def graficar_diferencia_potencia(salas_llenas, salas_vacias):
    # Potencia por muestra: cuadrado de cada valor
    pot_llenas = np.mean(np.square(salas_llenas), axis=(0, 1))  # shape: (52,)
    pot_vacias = np.mean(np.square(salas_vacias), axis=(0, 1))  # shape: (52,)
    diferencia_pot = pot_llenas - pot_vacias

    plt.figure(figsize=(12, 6))
    plt.plot(pot_llenas, label='Potencia - Salas llenas', linewidth=2)
    plt.plot(pot_vacias, label='Potencia - Salas vacías', linewidth=2)
    plt.plot(diferencia_pot, label='Diferencia de potencia', linestyle='--', color='black')
    plt.title('Comparación de potencia promedio por subportadora')
    plt.xlabel('Subportadora')
    plt.ylabel('Potencia promedio')
    plt.legend()
    plt.grid(True)
    plt.show()

def graficar_diferencia_maximo(salas_llenas, salas_vacias):
    # salas_llenas y salas_vacias tienen forma (num_archivos, muestras, subportadoras)
    
    # Maximo por archivo y subportadora (axis=1 sobre muestras)
    maximos_llenas = np.max(salas_llenas, axis=1)  # shape: (num_archivos, subportadoras)
    maximos_vacias = np.max(salas_vacias, axis=1)
    
    # Promedio de máximos sobre archivos (axis=0)
    prom_max_llenas = np.mean(maximos_llenas, axis=0)  # shape: (subportadoras,)
    prom_max_vacias = np.mean(maximos_vacias, axis=0)
    
    # Diferencia entre máximos promedio
    diferencia = prom_max_llenas - prom_max_vacias

    plt.figure(figsize=(12, 6))
    plt.plot(prom_max_llenas, label='Salas llenas (máximo promedio)', linewidth=2)
    plt.plot(prom_max_vacias, label='Salas vacías (máximo promedio)', linewidth=2)
    plt.plot(diferencia, label='Diferencia', linestyle='--', color='black')
    plt.title('Comparación de amplitud máxima promedio por subportadora')
    plt.xlabel('Subportadora')
    plt.ylabel('Amplitud máxima promedio')
    plt.legend()
    plt.grid(True)
    plt.show()


def graficar_fft_comparativa(señales_full, señales_empty, titulo_full='Salas Llenas', titulo_empty='Salas Vacías'):
    fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharey=True)

    # FFT para sala llena
    for i in range(len(señales_full)):
        señal = señales_full[i]
        fft = np.abs(np.fft.fft(señal, axis=0))  # FFT por subportadora
        fft_prom = np.mean(fft, axis=0)
        axes[0].plot(fft_prom, label=f'Archivo {i+1}')
    axes[0].set_title(f'FFT promedio por subportadora - {titulo_full}')
    axes[0].set_xlabel('Subportadora')
    axes[0].set_ylabel('Magnitud FFT')
    axes[0].legend(fontsize='small')
    axes[0].grid(True)

    # FFT para sala vacía
    for i in range(len(señales_empty)):
        señal = señales_empty[i]
        fft = np.abs(np.fft.fft(señal, axis=0))  # FFT por subportadora
        fft_prom = np.mean(fft, axis=0)
        axes[1].plot(fft_prom, label=f'Archivo {i+1}')
    axes[1].set_title(f'FFT promedio por subportadora - {titulo_empty}')
    axes[1].set_xlabel('Subportadora')
    axes[1].legend(fontsize='small')
    axes[1].grid(True)

    plt.suptitle('Comparativa FFT promedio por Subportadora entre Salas Llenas y Vacías', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def medir_concentracion_fft(lista_senales):
    """
    Calcula medidas de concentración espectral para una lista de señales.

    Retorna:
        - concentración_media: media de la proporción de energía en las subportadoras más energéticas
        - entropía_media: medida de dispersión espectral (menor = más concentrado)
        - curtosis_media: mide el "pico" de la distribución FFT (mayor = más concentración)
    """
    concentraciones = []
    entropias = []
    curtosis_list = []

    for señal in lista_senales:
        fft = np.abs(np.fft.fft(señal, axis=0))  # FFT por subportadora
        fft_prom = np.mean(fft, axis=0)

        # 1. Proporción de energía en el top 10% más fuerte
        energia_total = np.sum(fft_prom)
        top_10 = np.sort(fft_prom)[-int(0.1 * len(fft_prom)):]
        energia_top10 = np.sum(top_10)
        concentracion = energia_top10 / energia_total
        concentraciones.append(concentracion)

        # 2. Entropía espectral (mayor entropía = más disperso)
        p = fft_prom / np.sum(fft_prom)
        entropia = -np.sum(p * np.log2(p + 1e-12))  # suma con suavizado
        entropias.append(entropia)

        # 3. Curtosis (mayor curtosis = más concentración en el centro)
        mean = np.mean(fft_prom)
        std = np.std(fft_prom)
        kurt = np.mean(((fft_prom - mean) / std) ** 4)
        curtosis_list.append(kurt)

    return {
        'concentracion_media_top10': np.mean(concentraciones),
        'entropia_media': np.mean(entropias),
        'curtosis_media': np.mean(curtosis_list)
    }

def graficar_concentracion_fft(resultados_full, resultados_empty):
    etiquetas = ['Concentración Top 10%', 'Entropía espectral', 'Curtosis']
    valores_full = [
        resultados_full['concentracion_media_top10'],
        resultados_full['entropia_media'],
        resultados_full['curtosis_media']
    ]
    valores_empty = [
        resultados_empty['concentracion_media_top10'],
        resultados_empty['entropia_media'],
        resultados_empty['curtosis_media']
    ]

    x = np.arange(len(etiquetas))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, valores_full, width, label='Salas llenas', color='steelblue')
    bars2 = ax.bar(x + width/2, valores_empty, width, label='Salas vacías', color='orange')

    ax.set_ylabel('Valor de la métrica')
    ax.set_title('Comparación de concentración FFT entre salas llenas y vacías')
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Mostrar valores en las barras
    for bar in bars1 + bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def matriz_entropia_fft(señales):
    """
    Devuelve una matriz de entropía espectral: (n_archivos, n_subportadoras)
    """
    entropias = []
    for señal in señales:
        fft = np.abs(np.fft.fft(señal, axis=0))
        fft_norm = fft / np.sum(fft, axis=0, keepdims=True)  # normalizar
        entr = entropy(fft_norm + 1e-12, axis=0)  # (n_subportadoras,)
        entropias.append(entr)
    return np.array(entropias)


def graficar_heatmap_entropia(matriz_llenas, matriz_vacias):
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True, constrained_layout=True)

    # Mapa de calor - Salas Llenas
    sns.heatmap(matriz_llenas, ax=axes[0], cmap="YlGnBu", cbar_kws={"label": "Entropía"})
    axes[0].set_title("Entropía espectral - Salas Llenas")
    axes[0].set_xlabel("Subportadora")
    axes[0].set_ylabel("Archivo")

    # Mapa de calor - Salas Vacías
    sns.heatmap(matriz_vacias, ax=axes[1], cmap="YlOrRd", cbar_kws={"label": "Entropía"})
    axes[1].set_title("Entropía espectral - Salas Vacías")
    axes[1].set_xlabel("Subportadora")

    fig.suptitle("Comparativa de Entropía espectral por Subportadora", fontsize=16)
    plt.show()



def calcular_metricas_por_muestra(matriz_3d, metodo='promedio'):
    """
    matriz_3d: ndarray de shape (archivos, muestras, subportadoras)
    metodo: 'promedio', 'maxima', o 'potencia'
    return: array de 500 valores (uno por muestra de tiempo)
    """
    if metodo == 'promedio':
        # Promedia sobre subportadoras y luego sobre archivos
        resultado = np.mean(matriz_3d, axis=(0, 2))  # shape: (500,)
    elif metodo == 'maxima':
        # Máximo absoluto por muestra, luego promedio sobre archivos
        resultado = np.max(np.abs(matriz_3d), axis=2)  # shape: (17, 500)
        resultado = np.mean(resultado, axis=0)         # shape: (500,)
    elif metodo == 'potencia':
        # Potencia = cuadrado y promedio sobre subportadoras, luego promedio sobre archivos
        potencia = np.mean(matriz_3d**2, axis=2)  # shape: (17, 500)
        resultado = np.mean(potencia, axis=0)     # shape: (500,)
    else:
        raise ValueError("Método no válido. Usa 'promedio', 'maxima' o 'potencia'")
    
    return resultado


def graficar_metricas_misma_figura(pot_llenas, pot_vacias, title):
    plt.figure(figsize=(12, 5))
    plt.plot(pot_llenas, label='Salas Llenas', color='blue')
    plt.plot(pot_vacias, label='Salas Vacías', color='orange')
    plt.title(title)
    plt.xlabel("Muestra (tiempo)")
    plt.ylabel("Magnitud")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def aplicar_pca_y_concatenar(matriz_3d):
    """
    matriz_3d: ndarray shape (17 archivos, 500 muestras, 234 subportadoras)
    retorna: vector concatenado de PCA (17*500,)
    """
    pca = PCA(n_components=1)
    lista_pca = []

    for i in range(matriz_3d.shape[0]):  # por cada archivo
        data = matriz_3d[i]  # shape (500, 234)
        # Ajustar PCA y transformar
        señal_pca = pca.fit_transform(data)  # shape (500, 1)
        señal_pca = señal_pca.flatten()      # shape (500,)
        lista_pca.append(señal_pca)
    
    # Concatenar todas las señales en secuencia
    return np.concatenate(lista_pca)  # shape (17*500,)


def graficar_pca(pca_llenas, pca_vacias):
    # Graficar
    plt.figure(figsize=(16, 6))

    plt.plot(pca_llenas, label='Salas llenas', color='blue')
    plt.plot(pca_vacias, label='Salas vacías', color='orange')

    # Opcional: marcar separación entre archivos con líneas verticales
    for i in range(1, 17):
        plt.axvline(x=i*500, color='gray', linestyle='--', alpha=0.5)

    plt.title("Señales PCA concatenadas de 17 archivos (500 muestras c/u)")
    plt.xlabel("Tiempo concatenado (muestras)")
    plt.ylabel("Componente PCA 1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==== RUTINA PRINCIPAL ====
participante = f"{random.randint(1, 125):03}"
#participante = '007'
PATH_FULL = 'C:\\Users\\jsoto\\code\\dataset_full_csv\\' + participante + '\\'
PATH_EMPTY = 'C:\\Users\\jsoto\\code\\dataset_empty_csv\\'

# Cargar datos
print('>>>>>> participante: ',participante)
salas_llenas = cargar_archivos(PATH_FULL, num_archivos=17, modo='ultimo') # primero, ultimo, aleatorio
salas_vacias = cargar_archivos(PATH_EMPTY, num_archivos=17, modo='aleatorio')

print('shape salas llenas: ',salas_llenas.shape)
print('shape salas vacías: ',salas_vacias.shape)

# Aplicar a cada conjunto
pca_llenas = aplicar_pca_y_concatenar(salas_llenas)
pca_vacias = aplicar_pca_y_concatenar(salas_vacias)

graficar_pca(pca_llenas, pca_vacias)




prom_llenas = calcular_metricas_por_muestra(salas_llenas, metodo='promedio')
prom_vacias = calcular_metricas_por_muestra(salas_vacias, metodo='promedio')
graficar_metricas_misma_figura(prom_llenas, prom_vacias, 'promedio')


max_llenas = calcular_metricas_por_muestra(salas_llenas, metodo='maxima')
max_vacias = calcular_metricas_por_muestra(salas_vacias, metodo='maxima')
graficar_metricas_misma_figura(max_llenas, max_vacias, 'maxima')

pot_llenas = calcular_metricas_por_muestra(salas_llenas, metodo='potencia')
pot_vacias = calcular_metricas_por_muestra(salas_vacias, metodo='potencia')
graficar_metricas_misma_figura(pot_llenas, pot_vacias, 'potencia')




'''
# Visualizaciones
print('graficando las 17 señales de salas llenas y vacías...')
graficar_comparativa_17_archivos(salas_llenas, salas_vacias)

print('graficando diferencia promedio...')
graficar_diferencia_promedio(salas_llenas, salas_vacias)

print('graficando diferencia de potencia...')
graficar_diferencia_potencia(salas_llenas, salas_vacias)

print('graficnado diferencia de ampliud maxima...')
graficar_diferencia_maximo(salas_llenas, salas_vacias)

print('graficando magnitud de fft...')
graficar_fft_comparativa(salas_llenas, salas_vacias)


res_full = medir_concentracion_fft(salas_llenas)
res_empty = medir_concentracion_fft(salas_vacias)

graficar_concentracion_fft(res_full, res_empty)

print(">> Salas llenas:")
for k, v in res_full.items():
    print(f"{k}: {v:.4f}")

print("\n>> Salas vacías:")
for k, v in res_empty.items():
    print(f"{k}: {v:.4f}")


# Obtener matriz de entropía
matriz_llenas = matriz_entropia_fft(salas_llenas)
matriz_vacias = matriz_entropia_fft(salas_vacias)

graficar_heatmap_entropia(matriz_llenas, matriz_vacias)
'''
