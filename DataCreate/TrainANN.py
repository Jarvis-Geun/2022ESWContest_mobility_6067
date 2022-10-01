import sys
import pandas as pd
import numpy as np
import time

import random
import warnings
import os
from sklearn.model_selection import train_test_split, KFold

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return


class LinearModel(nn.Module):
    def __init__(self, input_features, output_features):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(input_features, 4)
        self.linear2 = nn.Linear(4, output_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        output = self.linear2(x)
        return output


class FatigueDataset(Dataset):
    def __init__(self, real: bool):
        super(FatigueDataset, self).__init__()

        if real:
            try:
                self.RppgFeatureCsv = pd.read_csv("./data/RppgFeature.csv")
                self.FacialFeatureCsv = pd.read_csv("./data/FacialFeature.csv")
                self.y_train = np.array(pd.read_csv("./data/Label.csv")).reshape(-1, 1)
            except FileNotFoundError:
                csv_list = [i for i in os.listdir("./data") if ".csv" in i]
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


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def AnnTraining(dataset, epochs, n_folds):
    input_features = dataset.x_train.shape[1]
    output_features = dataset.y_train.shape[1]

    loss_fn = nn.MSELoss()
    kfold = KFold(n_splits=n_folds, shuffle=True)
    best_mse = 1e4
    best_model_path = "./models/best_mse_model.pth"
    results = dict()
    start = time.time()

    print("Pytorch Linear Regression Model Training Start")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):

        print("========================")
        print("FOLD : {}".format(fold + 1))
        print("========================")

        train_subsampler = SubsetRandomSampler(train_idx)
        valid_subsampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset=dataset, batch_size=len(train_idx), sampler=train_subsampler)
        valid_loader = DataLoader(dataset=dataset, batch_size=len(val_idx), sampler=valid_subsampler)

        model = LinearModel(input_features, output_features)
        model.apply(reset_weights)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            model.train()
            train_loss = []
            for Data in train_loader:
                x_train = Data['features']
                y_train = Data['label']

                optimizer.zero_grad()
                output = model(x_train)
                loss = loss_fn(output, y_train)

                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            valid_loss = []
            model.eval()
            with torch.no_grad():
                for Data in valid_loader:
                    x_valid = Data['features']
                    y_valid = Data['label']

                    output = model(x_valid)
                    loss = loss_fn(output, y_valid)
                    valid_loss.append(loss.item())

            if epoch % 10 == 0:
                print(f"epoch: {epoch}, train_loss: {np.mean(train_loss):.4f}, valid_loss: {np.mean(valid_loss):.4f}")

        results[fold + 1] = {"train_loss": np.round(np.mean(train_loss), 4),
                             "valid_loss": np.round(np.mean(valid_loss), 4)}

        if not os.path.exists("./models"):
            os.mkdir("./models")
        model_name = "./models/{}_fold_model_state_dict.pth".format(fold + 1)
        torch.save(model.state_dict(), model_name)
        if np.mean(valid_loss) < best_mse:
            best_mse = np.mean(valid_loss)
            torch.save(model.state_dict(), best_model_path)

    print("\nTraining Time: {:.4f}\n".format(time.time() - start))
    return results


if __name__ == "__main__":

    # set parameter
    seed = 42
    epochs = 100
    n_folds = 10
    set_seed(seed)
    real = False  # Using real dataset or not

    # dataset for training
    dataset = FatigueDataset(real)
    results = AnnTraining(dataset, epochs, n_folds)

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
