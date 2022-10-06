import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import  display

import facial_feature
import rppg_feature
import trainANN
import train

if __name__ == '__main__':

    ## Get dataset(csv format)
    if not os.path.exists("./data"):
        os.mkdir("./data")

    rppg_final_path = "./data/RppgFeature.csv"
    label_final_path = "./data/Label.csv"
    facial_final_path = "./data/FacialFeature.csv"

    data_path = "../get_PPG_GTD"
    folders = ["PGH", "PMG", "PHJ", "JHO", "HGT"]

    rppg_final_df = pd.DataFrame(columns=['HR', 'HR_std', 'SDNN', 'LF', 'HF', 'LF/HF'])
    facial_final_df = pd.DataFrame(columns=['PERCLOSE', 'Excessive Blink', 'Yawn'])
    label_results = []

    a_start = time.time()
    for folder in tqdm(sorted(folders)):
        try:
            print(" =========== Folder: {} ===========".format(folder))
            for i in tqdm(range(1, 11)):
                start = time.time()
                ppg_fname = folder+str(i)+".txt"
                vid_fname = folder+str(i)+".avi"
                label_fname = folder + str(i) + "_label.txt"

                rppg_df = pd.DataFrame(columns=['HR', 'HR_std', 'SDNN', 'LF', 'HF', 'LF/HF'])
                facial_df = pd.DataFrame(columns=['PERCLOSE', 'Excessive Blink', 'Yawn'])

                ppg = rppg_feature.read_text_file(os.path.join(*[data_path, folder, ppg_fname]))
                vid = facial_feature.OneVidProcessing(os.path.join(*[data_path, folder, vid_fname]))
                with open(label_fname, 'r') as f:
                    numbers = f.readlines()
                start = float(numbers[0].strip("\n"))
                end = float(numbers[1].strip("\n"))
                interval = round((end-start)/5, 6)

                for i in range(6):
                    rppg_df.loc[i] = rppg_feature.mk_df_WESAD(ppg[3000 * i:3000 * (i + 1)])
                    label_results.append(round(start + (interval * i), 4))

                facial_final_df = pd.concat([facial_final_df, facial_df], axis=0)
                rppg_final_df = pd.concat([rppg_final_df, rppg_df], axis=0)

                print("[fileNumber]: {}, [Processing Time]: {:.4f}".format(i, time.time() - start))

        except FileNotFoundError:
            print("===== Check your data folder & file name =====")
            print("[FileNotFoundError] {}".format(os.path.join(*[data_path, folder, folder+str(i)])))
            sys.exit()

    label_final_df = pd.DataFrame(label_results, columns=['FatigueScore'])
    label_final_df.to_csv(label_final_path, index=False)
    facial_final_df.reset_index().to_csv(facial_final_path, index=False)
    rppg_final_df.reset_index().to_csv(rppg_final_path, index=False)

    print("processing time(Making CSV dataset): {:.4f} sec".format(time.time() - a_start))

    ## Training
    if len(sys.argv) > 1:
        if sys.argv[1] == '--pytorch':
            seed = 42
            epochs = 100
            n_folds = 10
            trainANN.set_seed(seed)
            real = True  # Using real dataset or not

            # dataset for training
            dataset = trainANN.FatigueDataset(real)
            results = trainANN.AnnTraining(dataset, epochs, n_folds)

            print("===== result =====")
            final_mse = 0
            best_fold = 0
            best_mse = 1e4
            for key, value in results.items():
                print(f"fold : {key}, train_loss: {value['train_loss']}, val_loss: {value['valid_loss']}")
                final_mse += value['valid_loss']
                if value['valid_loss'] < best_mse:
                    best_mse = value['valid_loss']
                    best_fold = key

            print("\n10 Kfold mean MSE : {}".format(final_mse / 10))
            print("best fold : {}, best_mse: {}".format(best_fold, best_mse))

        elif sys.argv[2] == '--sklearn':
            seed = 42
            train.set_seed(seed)
            real = True  # Using real dataset or not

            # Read csv file & Concatenate DataFrame
            if real:
                try:
                    RppgFeatureCsv = pd.read_csv("./data/RppgFeature.csv")
                    FacialFeatureCsv = pd.read_csv("./data/FacialFeature.csv")
                    y_train = np.array(pd.read_csv("./data/Label.csv")['FatigueScore'])
                except FileNotFoundError:
                    csv_list = [i for i in os.listdir("./data") if ".csv" in i]
                    print(" === csv file list ===")
                    for i in csv_list:
                        print(i)
                    sys.exit()
                x_train = np.concatenate((RppgFeatureCsv.values, FacialFeatureCsv.values), axis=1)

            else:
                from sklearn.datasets import load_boston

                Dataset = load_boston()
                x_train = Dataset['data']
                y_train = Dataset['target']

            start = time.time()
            train.RegressorTraining(x_train, y_train)
            print("Training Time: {:.4f}\n".format(time.time() - start))

            print(" ======= GridSearch Result ======= ")
            display(pd.read_csv("./model/GridSearchCvResult.csv"))

        else:
            print("====== check your argv[1] parameter ========")
            print("--pytorch : Neural Network training using torch.nn.linear")
            print("--sklearn : Machine Learning Regression training using scikit-learn")
            sys.exit()
