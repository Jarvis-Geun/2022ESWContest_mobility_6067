import numpy as np
import pandas as pd
import sys
import os

if __name__ == "__main__":

    if not os.path.exists("./data"):
        os.mkdir("./data")

    final_path = "./data/Label.csv"
    data_path = "../get_PPG_GTD"
    folders = ["PGH", "PMG", "PHJ", "JHO", "HGT"]
    result = []

    for folder in sorted(folders):
        try:
            LabelData = [folder + str(i) + "_label.txt" for i in range(1, 11)]
            for data in sorted(LabelData):
                txt_file = os.path.join(*[data_path, folder, data])

                with open(txt_file, 'r') as f:
                    numbers = f.readlines()
                start = float(numbers[0].strip("\n"))
                end = float(numbers[1].strip("\n"))
                interval = round((end-start)/5, 6)
                for i in range(6):
                    result.append([round(start+(interval*i), 4), data])
        except FileNotFoundError:
            print("===== Check your data folder & file name =====")
            print("[FileNotFoundError] {}".format(os.path.join(*[data_path, folder, data])))
            sys.exit()

    pd.DataFrame(result, columns=['FatigueScore', 'path']).to_csv(final_path, index=False)