import sys
import time
# import importlib
import config
#from analysis.dataAnalysis import analyze  # , julio_analysis
from plotters.AmpPhaPlotter import Plotter  # Amplitude and Phase plotter
import decoders.interleaved as decoder
import numpy as np
# decoder = importlib.import_module(f'decoders.{config.decoder}')  # This is also an import


def string_is_int(s):
    """
    Check if a string is an integer
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    pcap_filename = sys.argv[1]

    if '.pcap' not in pcap_filename:
        pcap_filename += '.pcap'

    try:
        samples = decoder.read_pcap(pcap_filename)
    except FileNotFoundError:
        print(f'File {pcap_filename} not found.')
        exit(-1)

    if config.plot_samples:
        plotter = Plotter(samples.bandwidth)

    csi_data = samples.get_pd_csi()
    analyze(csi_data)
