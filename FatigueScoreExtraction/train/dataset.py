import sys
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

class FatigueDataset(Dataset):
    def __init__(self, real: bool):
        super(FatigueDataset, self).__init__()

        if real:
            try:
                self.RppgFeatureCsv = pd.read_csv("../data/RppgFeature.csv")
                self.FacialFeatureCsv = pd.read_csv("../data/FacialFeature.csv")
                self.y_train = pd.read_csv("../data/Label.csv").values.reshape(-1, 1)
            except FileNotFoundError:
                csv_list = [i for i in os.listdir("../data") if ".csv" in i]
                print(" === csv file list ===")
                for i in csv_list:
                    print(i)
                sys.exit()
            self.x_train = np.concatenate((self.RppgFeatureCsv.values, self.FacialFeatureCsv.values), axis=1)

        else:
            from sklearn.datasets import load_boston
            boston_dataset = load_boston()
            self.x_train = boston_dataset['data']
            self.y_train = boston_dataset['target'].reshape(-1, 1)

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        data = {
            'features': torch.FloatTensor(self.x_train[idx]),
            'label': torch.FloatTensor(self.y_train[idx])
        }
        return data

if __name__ == "__main__":
    real = True
    dataset = FatigueDataset(real)
    print("x_train shape : {}, y_train shape: {}".format(dataset.x_train.shape, dataset.y_train.shape))

