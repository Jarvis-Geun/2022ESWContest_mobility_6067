import biosppy
import pyhrv.tools as tools
import pyhrv.time_domain as td
from scipy.signal import argrelextrema
import pyhrv.frequency_domain as fd
import numpy as np


def hrv_analysis(rppg):
    print(rppg)
    # np.save("./rppgsignal.npy", rppg)
    PPG_signal, _ = biosppy.signals.ppg.ppg(rppg, sampling_rate=25, show=False)[1:3]
    
    PPG_peaks = argrelextrema(PPG_signal, np.greater, order=12)
    PPG_peaks_sq = np.squeeze(PPG_peaks)
    real_time = PPG_peaks_sq * 0.04
    PPG_nni = tools.nn_intervals(real_time)

    # ar method
    PPG_ratio = fd.welch_psd(PPG_nni, show=False)
    LF, HF = PPG_ratio['fft_norm']
    # hr
    HR = td.hr_parameters(PPG_nni)
    # SDNN
    sdnn = td.sdnn(PPG_nni)
    
    ret = np.array([HR['hr_mean'], HR['hr_std'], sdnn['sdnn'], LF, HF, PPG_ratio['fft_ratio']])
    
    return ret
