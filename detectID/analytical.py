import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import seaborn as sns
import time
import re
from scipy.stats import entropy
from scipy.stats import zscore
from sklearn.decomposition import PCA
from detectID import get_selected_subcarries
from filters import hampel_filter, moving_avg_filter, iq_samples_abs, moving_median_filter
from sklearn.covariance import MinCovDet
from scipy.stats import chi2

random.seed(time.time())

def extraer_numero(nombre_archivo):
    # Busca el primer número en el nombre del archivo
    numeros = re.findall(r'\d+', nombre_archivo)
    return int(numeros[0]) if numeros else -1

def calcular_outliers_zscore(df, umbral=3):

    z_scores = df.apply(zscore)
    z_scores_abs = z_scores.abs()
    outliers = (z_scores_abs > 3)
    porcentaje_outliers = outliers.sum().sum() / df.size * 100

    return float(porcentaje_outliers)

def calcular_outliers_rc(df, alpha=0.99):
    mcd = MinCovDet().fit(df)
    mahal_dist = mcd.mahalanobis(df)

    # Usamos la distribución chi-cuadrado para el umbral (p=0.975 típico)
    umbral = chi2.ppf(alpha, df.shape[1])
    outliers = mahal_dist > umbral

    porcentaje_outliers = outliers.sum() / df.shape[0] * 100
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
    #archivos = sorted(archivos)  # ordenar alfabéticamente
    archivos = sorted(archivos, key=extraer_numero)

    if modo == 'primero':
        archivos_csv = archivos[:num_archivos]
    elif modo == 'ultimo':
        archivos_csv = archivos[-num_archivos:]
    elif modo == 'aleatorio':
        if num_archivos > len(archivos):
            raise ValueError("num_archivos excede la cantidad de archivos disponibles")

        # Seleccionar aleatoriamente el índice del primer archivo
        max_inicio = len(archivos) - num_archivos
        inicio = random.randint(0, max_inicio)
        archivos_csv = archivos[inicio:inicio + num_archivos]
    else:
        raise ValueError("El parámetro 'modo' debe ser 'primero', 'ultimo' o 'aleatorio'.")
    
    '''
    elif modo == 'aleatorio':
        if num_archivos > len(archivos):
            raise ValueError("num_archivos excede la cantidad de archivos disponibles")
        archivos_csv = random.sample(archivos, num_archivos)
    '''

    señales = []

    print('>>>>>>>> archivos nombres: ', [os.path.splitext(os.path.basename(f))[0] for f in archivos_csv])

    lista_porcentajes_zscore = []
    lista_porcentajes_rc = []

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
        sns.heatmap(df.corr(), cmap='coolwarm')
        
        lista_porcentajes_zscore.append(calcular_outliers_zscore(df))
        lista_porcentajes_rc.append(calcular_outliers_rc(df))
        
        #df = hampel_filter(df) 
        '''
        df = moving_median_filter(df)
        media_columnas = df.mean(axis=0)  # resultado: vector de 234 valores
        plt.figure(figsize=(14,8))
        plt.plot(media_columnas, alpha=0.3)  # todas las columnas con transparencia
        plt.title('Señales filtradas con mediana móvil (todas las subportadoras)')
        plt.xlabel('Índice de fila')
        plt.ylabel('Valor filtrado')
        plt.grid(True)
        plt.show()
        '''
        
        df = moving_avg_filter(df)
        '''
        media_columnas = df.mean(axis=0)  # resultado: vector de 234 valoresedia_columnas
        plt.figure(figsize=(14,8))
        plt.plot(media_columnas, alpha=0.3)  # todas las columnas con transparencia
        plt.title('Señales filtradas con media móvil (todas las subportadoras)')
        plt.xlabel('Índice de fila')
        plt.ylabel('Valor filtrado')
        plt.grid(True)
        plt.show()
        '''

        # Guardar como matriz real de amplitudes
        #señal = np.abs(df.values)  # shape: (muestras, 51)
        señales.append(df.values)

    #print('\nlista de % de outliers zscore: ', lista_porcentajes_zscore)
    print("\nlista de % de outliers zscore:".join(f"{v:.4f}" for v in lista_porcentajes_zscore))
    #print('lista de % de outliers rc: ', lista_porcentajes_rc)
    print("\nlista de % de outliers rc:".join(f"{v:.4f}" for v in lista_porcentajes_rc))

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
def graficar_diferencia_maximo(salas_llenas, salas_vacias, indices_reales=None):
    """
    Graficar máximos promedio por subportadora (relleno o vacío) y su diferencia.
    
    - salas_llenas / salas_vacias: shape (archivos, muestras, subportadoras)
    - indices_reales: opcional, lista con los índices originales de subportadoras. Si no se da, usa 0, 1, 2, ...
    """
    max_llenas = np.max(salas_llenas, axis=1)   # (archivos, subportadoras)
    max_vacias = np.max(salas_vacias, axis=1)

    prom_llenas = np.mean(max_llenas, axis=0)
    prom_vacias = np.mean(max_vacias, axis=0)
    diferencia = prom_llenas - prom_vacias

    n_subs = prom_llenas.shape[0]
    x_labels = indices_reales if indices_reales is not None else np.arange(n_subs)

    plt.figure(figsize=(12, 6))
    plt.plot(prom_llenas, label='Salas llenas', linewidth=2)
    plt.plot(prom_vacias, label='Salas vacías', linewidth=2)
    plt.plot(diferencia, label='Diferencia', linestyle='--', color='black')

    plt.xticks(ticks=np.arange(n_subs), labels=x_labels, rotation=90)

    plt.title('Comparación de amplitud máxima promedio por subportadora')
    plt.xlabel('Índice de subportadora')
    plt.ylabel('Amplitud máxima promedio')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
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



def matriz_entropia_fft_filtrada(señales, umbral_percentil=70):
    """
    Calcula entropía espectral y retorna solo las subportadoras con mayor entropía promedio.

    Parámetros:
    - señales: array de shape (n_archivos, n_muestras, n_subportadoras)
    - umbral_percentil: percentil de corte para conservar las subportadoras más entropicas

    Retorna:
    - señales_filtradas: array reducido (n_archivos, n_muestras, n_subs_filtradas)
    - indices_subs_filtradas: indices de subportadoras seleccionadas
    """
    entropias = []

    for señal in señales:  # señal shape: (n_muestras, n_subportadoras)
        fft = np.abs(np.fft.fft(señal, axis=0))
        fft_norm = fft / np.sum(fft, axis=0, keepdims=True)
        entr = entropy(fft_norm + 1e-12, axis=0)  # (n_subportadoras,)
        entropias.append(entr)

    entropias = np.array(entropias)  # shape: (n_archivos, n_subportadoras)
    entropia_promedio = np.mean(entropias, axis=0)

    # Filtrar subportadoras con mayor entropía (por percentil)
    umbral = np.percentile(entropia_promedio, umbral_percentil)
    indices_filtrados = np.where(entropia_promedio >= umbral)[0]

    # Filtrar las señales en base a esas subportadoras
    señales_filtradas = señales[:, :, indices_filtrados]  # reduce subportadoras

    return señales_filtradas, indices_filtrados



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



def graficar_heatmap_entropia_subportadoras(entropia_llenas, entropia_vacias, indices_subs):
    """
    Grafica dos heatmaps lado a lado donde:
    - Eje X: subportadoras seleccionadas (índices reales)
    - Eje Y: número de sala (archivo)
    - Color: entropía espectral
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    # Heatmap - Salas llenas
    im0 = axes[0].imshow(entropia_llenas, aspect='auto', cmap='YlGnBu', origin='lower')
    axes[0].set_title("Entropía por Subportadora - Salas Llenas")
    axes[0].set_xlabel("Índice de Subportadora")
    axes[0].set_ylabel("Sala (archivo)")
    axes[0].set_xticks(np.arange(len(indices_subs)))
    axes[0].set_xticklabels(indices_subs, rotation=90)
    fig.colorbar(im0, ax=axes[0], label="Entropía")

    # Heatmap - Salas vacías
    im1 = axes[1].imshow(entropia_vacias, aspect='auto', cmap='YlOrRd', origin='lower')
    axes[1].set_title("Entropía por Subportadora - Salas Vacías")
    axes[1].set_xlabel("Índice de Subportadora")
    axes[1].set_xticks(np.arange(len(indices_subs)))
    axes[1].set_xticklabels(indices_subs, rotation=90)
    fig.colorbar(im1, ax=axes[1], label="Entropía")

    plt.suptitle("Heatmap de Entropía por Sala y Subportadora", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
    plt.figure(figsize=(16, 6))
    
    plt.plot(pca_llenas, label='Salas llenas', color='blue')
    plt.plot(pca_vacias, label='Salas vacías', color='orange')

    # Líneas verticales cada 500 muestras
    num_archivos = len(pca_llenas) // 500
    for i in range(1, num_archivos):
        plt.axvline(x=i * 500, color='gray', linestyle='--', alpha=0.5)

    # Ticks del eje X cada 500 muestras
    plt.xticks(ticks=np.arange(0, len(pca_llenas) + 1, 500))

    plt.title("Señales PCA concatenadas de 17 archivos (500 muestras c/u)")
    plt.xlabel("Tiempo concatenado (muestras)")
    plt.ylabel("Componente PCA 1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def graficar_maximos_por_archivo(salas_llenas, salas_vacias, indices_reales=None):
    """
    Grafica los máximos por subportadora de cada archivo individual para salas llenas y vacías.

    - salas_llenas / salas_vacias: shape (archivos, muestras, subportadoras)
    - indices_reales: lista opcional con los índices originales de subportadoras.
    """
    max_llenas = np.max(salas_llenas, axis=1)   # (archivos, subportadoras)
    max_vacias = np.max(salas_vacias, axis=1)

    n_archivos, n_subs = max_llenas.shape
    x_labels = indices_reales if indices_reales is not None else np.arange(n_subs)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # --- Salas llenas ---
    for i in range(n_archivos):
        axes[0].plot(x_labels, max_llenas[i], label=f'Archivo {i+1}')
    axes[0].set_title("Máximos por archivo - Salas Llenas")
    axes[0].set_xlabel("Índice de Subportadora")
    axes[0].set_ylabel("Amplitud máxima")
    axes[0].grid(True)
    axes[0].legend(loc='upper right', fontsize='small', ncol=2)
    axes[0].set_xticks(x_labels)
    axes[0].tick_params(axis='x', rotation=90)

    # --- Salas vacías ---
    for i in range(n_archivos):
        axes[1].plot(x_labels, max_vacias[i], label=f'Archivo {i+1}')
    axes[1].set_title("Máximos por archivo - Salas Vacías")
    axes[1].set_xlabel("Índice de Subportadora")
    axes[1].grid(True)
    axes[1].legend(loc='upper right', fontsize='small', ncol=2)
    axes[1].set_xticks(x_labels)
    axes[1].tick_params(axis='x', rotation=90)

    fig.suptitle("Comparación de amplitudes máximas por subportadora (archivo por archivo)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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


graficar_maximos_por_archivo(salas_llenas, salas_vacias)

res_full = medir_concentracion_fft(salas_llenas)
res_empty = medir_concentracion_fft(salas_vacias)

graficar_concentracion_fft(res_full, res_empty)

print("\n>> Salas llenas:")
for k, v in res_full.items():
    print(f"{k}: {v:.4f}")

print("\n>> Salas vacías:")
for k, v in res_empty.items():
    print(f"{k}: {v:.4f}")





# Obtener matriz de entropía
matriz_llenas = matriz_entropia_fft(salas_llenas)
matriz_vacias = matriz_entropia_fft(salas_vacias)

graficar_heatmap_entropia(matriz_llenas, matriz_vacias)

# 1. Filtrar señales con alta entropía
señales_llenas_filtradas, subs_llenas = matriz_entropia_fft_filtrada(salas_llenas, umbral_percentil=65)
señales_vacias_filtradas, subs_vacias = matriz_entropia_fft_filtrada(salas_vacias, umbral_percentil=65)

# 2. Obtener subportadoras comunes
indices_comunes = np.intersect1d(subs_llenas, subs_vacias)
print("Subportadoras comunes seleccionadas:", indices_comunes)

# 3. Convertir índices absolutos a relativos en cada conjunto
rel_idx_llenas = [i for i, val in enumerate(subs_llenas) if val in indices_comunes]
rel_idx_vacias = [i for i, val in enumerate(subs_vacias) if val in indices_comunes]

# 4. Filtrar matrices 3D con índices relativos
salas_llenas_comunes = señales_llenas_filtradas[:, :, rel_idx_llenas]
salas_vacias_comunes = señales_vacias_filtradas[:, :, rel_idx_vacias]

# 5. Calcular entropía de esas subportadoras comunes
ent_llenas = matriz_entropia_fft(salas_llenas_comunes)
ent_vacias = matriz_entropia_fft(salas_vacias_comunes)

# 6. Graficar heatmap con etiquetas reales
graficar_heatmap_entropia_subportadoras(ent_llenas, ent_vacias, indices_comunes)

graficar_diferencia_promedio(salas_llenas_comunes, salas_vacias_comunes)
graficar_diferencia_potencia(salas_llenas_comunes, salas_vacias_comunes)
graficar_diferencia_maximo(salas_llenas_comunes, salas_vacias_comunes, indices_comunes)



