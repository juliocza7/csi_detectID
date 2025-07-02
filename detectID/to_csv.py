import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_script_dir, '..')
sys.path.append(project_root)

from csiexplorer.decoders import interleaved as decoder

diretorio = 'C:\\Users\\jsoto\\dataset\\scans_ds2\\001\\'

'''
contenido = os.listdir(diretorio)
print('Quant_file: ', len(contenido), '\n')
name_fil = 0

for i in contenido:
    print('obteniendo dados CSI del pcap: ', i)
    samples = decoder.read_pcap(diretorio + i)
    csi_data = samples.get_pd_csi()

    name_fil += 1
    print('creando archivo csv: ', name_fil)
    csi_data.to_csv('C:\\Users\\jsoto\\dataset\\fullroom_2000\\' + str(name_fil) + '.csv', sep=',')
'''

contenido = os.listdir(diretorio)
print('Quant_file: ', len(contenido), '\n')

for i, filename in enumerate(contenido):
    print('obteniendo dados CSI del pcap:', filename)
    
    # Leer datos CSI
    samples = decoder.read_pcap(os.path.join(diretorio, filename))
    csi_data = samples.get_pd_csi()
    
    # Obtener el número antes del primer "_"
    nombre_csv = filename.split('_')[0] + '.csv'
    print('creando archivo csv:', nombre_csv)

    # Guardar CSV (ejemplo, ajusta esto a tu método de guardado)
    csi_data.to_csv('C:\\Users\\jsoto\\dataset\\fullroom_2000\\' + nombre_csv, sep=',')

    