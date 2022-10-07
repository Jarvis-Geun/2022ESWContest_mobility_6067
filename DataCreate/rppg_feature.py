import os
import biosppy
import pyhrv.tools as tools
import pyhrv.time_domain as td

from scipy.signal import argrelextrema
import pandas as pd

from tqdm import tqdm
import pyhrv.frequency_domain as fd
import numpy as np
import sys
from IPython.display import display
import time

def read_text_file(path):
    array = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            line = line.split()[1][2:-6]
            array.append(line)
        array = array[1:]
        for i in range(len(array)):
            try:
                if float(array[i]) > 5:
                    array[i] = float(array[i - 1])
                else:
                    array[i] = float(array[i])
            except:
                array[i] = float(array[i - 1])
    return np.array(array)


def mk_df_WESAD(merge):
    PPG_np = merge
    PPG_signal, _ = biosppy.signals.ppg.ppg(PPG_np, sampling_rate=25, show=False)[1:3]

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

    return [HR['hr_mean'], HR['hr_std'], sdnn['sdnn'], LF, HF, PPG_ratio['fft_ratio']]


if __name__ == "__main__":

    if not os.path.exists("./data"):
        os.mkdir("./data")

    final_path = "./data/RppgFeature.csv"
    data_path = "../get_PPG_GTD"
    folders = ["PGH", "geun", "jho", "phj", "hwang"]

    final_df = pd.DataFrame(columns=['HR', 'HR_std', 'SDNN', 'LF', 'HF', 'LF/HF'])
    start = time.time()

    for folder in tqdm(sorted(folders)):
        try:
            print(" =========== Folder: {} ===========".format(folder))
            ppgData = [folder+str(i)+".txt" for i in range(1, 11)]
            if folder == 'hwang':
                ppgData = ppgData[:-1]
            for data in tqdm(ppgData):
                ppg = read_text_file(os.path.join(*[data_path, folder, data]))
                df = pd.DataFrame(columns=['HR', 'HR_std', 'SDNN', 'LF', 'HF', 'LF/HF'])
                for i in range(6):
                    df.loc[i] = mk_df_WESAD(ppg[3000 * i:3000 * (i + 1)])
                final_df = pd.concat([final_df, df], axis=0)

        except FileNotFoundError:
            print("===== Check your data folder & file name =====")
            print("[FileNotFoundError] {}".format(os.path.join(*[data_path, folder, data])))
            sys.exit()

    final_df.reset_index().to_csv(final_path, index=False)
    display(final_df.info())
    print("processing time: {:.4f} sec".format(time.time() - start))
