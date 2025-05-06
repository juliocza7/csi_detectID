import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from decoders import interleaved as decoder
import os
import time


diretorio = 'D:\\ULima\\PosDoc\\code\\csi_detectID\\1_2022_04_28_-_15_17_32_bw_80_ch_36.pcap'

samples = decoder.read_pcap(diretorio)
csi_data = samples.get_pd_csi()

csi_data.to_csv('D:\\ULima\\PosDoc\\code\\csi_detectID\\lleno.csv', sep=',')

'''
contenido = os.listdir(diretorio)
print('Quant_file: ', len(contenido), '\n')
cont = 0

for i in contenido:

    name_fil = i[0:i.find("_2023_")]
    print('file #: ', name_fil)

    samples = decoder.read_pcap(diretorio + i)
    csi_data = samples.get_pd_csi()
    #os.rename(diretorio + i, diretorio + str(seq) + '.pcap')
    #time.sleep(1)
    
    csi_data.to_csv('D:\\ULima\\PosDoc\\code\\csi_detectID\\' + name_fil + '.csv', sep=',')
    cont +=1
'''