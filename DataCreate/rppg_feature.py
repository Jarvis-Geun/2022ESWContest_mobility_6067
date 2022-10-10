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

def read_text_file(ppg_file, video_file):
    ppg_df = pd.read_csv(ppg_file, delimiter=' ')
    video_df = pd.read_csv(video_file, delimiter=' ')
    ppg_df.drop('0', axis=1, inplace=True)
    ppg_df.columns = ['ppg', 'time']
    video_df.drop('0', axis=1, inplace=True)
    video_df.columns = ['time']

    ppg_df.iloc[:, 1] = np.round(ppg_df.iloc[:, 1], 2)
    video_df.iloc[:, 0] = np.round(video_df.iloc[:, 0], 2)
    ppg_df = ppg_df.drop_duplicates('time')

    df = pd.merge(video_df,ppg_df, how='inner',on='time')
    ppg_list = df['ppg']
    nums = []
    for i in ppg_list:
        try:
            temp = float(i)
            nums.append(temp)
        except:
            nums.append(float(i))
        
    return np.array(nums)


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
                ppg = read_text_file(os.path.join(data_path, folder, data+".txt"), os.path.join(data_path, folder, data+'_frame_time.txt'))
                df = pd.DataFrame(columns=['HR', 'HR_std', 'SDNN', 'LF', 'HF', 'LF/HF'])
                for i in range(6):
                    df.loc[i] = mk_df_WESAD(ppg[3000 * i:3000 * (i + 1)])
                final_df = pd.concat([final_df, df], axis=0)

        except FileNotFoundError:
            print("===== Check your data folder & file name =====")
            print("[FileNotFoundError] {}".format(os.path.join(*[data_path, folder, data+".txt"])))
            sys.exit()

    final_df.reset_index().to_csv(final_path, index=False)
    display(final_df.info())
    print("processing time: {:.4f} sec".format(time.time() - start))
