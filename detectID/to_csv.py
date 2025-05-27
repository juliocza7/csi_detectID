import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_script_dir, '..')
sys.path.append(project_root)

from csiexplorer.decoders import interleaved as decoder

diretorio = 'D:\\ULima\\PosDoc\\code\\dataset_empty_csv\\500_coletas_vazias\\'

contenido = os.listdir(diretorio)
print('Quant_file: ', len(contenido), '\n')
name_fil = 1710

for i in contenido:
    print('obteniendo dados CSI del pcap: ', i)
    samples = decoder.read_pcap(diretorio + i)
    csi_data = samples.get_pd_csi()

    name_fil += 1
    print('creando archivo csv: ', name_fil)
    csi_data.to_csv('D:\\ULima\\PosDoc\\code\\dataset_empty_csv\\csv_newemptyrooms\\' + str(name_fil) + '.csv', sep=',')