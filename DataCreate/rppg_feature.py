import numpy as np
import os
import matplotlib.pyplot as plt
import biosppy
import pyhrv.tools as tools
import pyhrv.time_domain as td
import pyhrv.hrv as hrv
from scipy.signal import argrelextrema
import pandas as pd
import scipy.stats as st
import glob
from tqdm import tqdm
import pyhrv.frequency_domain as fd
import numpy as np

def read_text_file(path, filename):
    array = []
    with open(path + filename, "r") as f:
        for line in f:
            line = line.strip()
            line = line.split()[1][2:-6]
            array.append(line)
        array = array[1:]
        for i in range(len(array)):
            try:
                if float(array[i]) > 5:
                    array[i] = float(array[i-1])
                else:
                    array[i] = float(array[i])
            except:
                array[i] = float(array[i-1])
    return np.array(array)


def mk_df_WESAD(merge):
    PPG_np = merge[:3000]
    PPG_signal, _ = biosppy.signals.ppg.ppg(PPG_np, sampling_rate=25, show=False)[1:3]

    PPG_peaks = argrelextrema(PPG_signal, np.greater, order=20)
    PPG_peaks_sq = np.squeeze(PPG_peaks)
    real_time = PPG_peaks_sq * 0.04
    PPG_nni = tools.nn_intervals(real_time)

    # ar method
    PPG_ratio = fd.welch_psd(PPG_nni, show=False)
    LF, HF = PPG_ratio['fft_norm']
    # hr
    HR = td.hr_parameters(PPG_nni)
    #SDNN
    sdnn = td.sdnn(PPG_nni)

    return [HR['hr_mean'], HR['hr_std'], sdnn['sdnn'], LF, HF, PPG_ratio['fft_ratio']]
    
if __name__ == "__main__":   
    # 변수 선언    
    path = 'C:\\Users\\hyeon\\Desktop\\임베디드\\data\\PGH\\' 
    filename = 'PGH1.txt'
    df = pd.DataFrame(columns=['HR', 'HR_std', 'SDNN', 'LF', 'HF', 'LF/HF'])
    ppg = read_text_file(path, filename)
    
    for i in range(6):
        df.loc[i] = mk_df_WESAD(ppg[3000*i:3000*(i+1)])
    df.to_csv(filename[:-4]+'.csv')