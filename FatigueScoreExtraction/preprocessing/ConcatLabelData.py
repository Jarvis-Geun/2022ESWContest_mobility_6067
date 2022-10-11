import numpy as np
import pandas as pd
import sys
import os
from IPython.display import display
import time

def GetResult(path, result):

    with open(path, 'r') as f:
        numbers = f.readlines()

    try:
        start = float(numbers[0].split(",")[0])
        end = float(numbers[1].split(",")[0])
    except ValueError:
        start = float(numbers[0].strip("\n"))
        end = float(numbers[1].strip("\n"))

    interval = round((end - start) / 5, 6)
    for i in range(6):
        result.append(round(start + (interval * i), 4))

    return result

if __name__ == "__main__":

    if not os.path.exists("../data"):
        os.mkdir("../data")

    final_path = "../data/Label.csv"
    data_path = "../../get_PPG_GTD"
    folders = ["PGH", "geun", "jho", "phj", "hwang"]
    result = []

    start = time.time()
    for folder in sorted(folders):
        try:
            LabelData = [folder + str(i) + "_label.txt" for i in range(1, 11)]
            if folder == 'hwang':
                LabelData = LabelData[:-1]
            for data in LabelData:
                txt_file = os.path.join(*[data_path, folder, data])
                result = GetResult(txt_file, result)

        except FileNotFoundError:
            print("===== Check your data folder & file name =====")
            print("[FileNotFoundError] {}".format(os.path.join(*[data_path, folder, data])))
            sys.exit()

    final_csv = pd.DataFrame(result, columns=['FatigueScore'])
    final_csv.to_csv(final_path, index=False)
    display(final_csv.info())
    print("processing time: {:.4f} sec".format(time.time() - start))