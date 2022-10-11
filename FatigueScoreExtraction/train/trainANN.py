import numpy as np
import time

import random
import warnings
import os
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

import model
import dataset


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return

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
    best_model_path = "../models/best_mse_model.pth"
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

        model = model.LinearModel(input_features, output_features)
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

            if epoch % 100 == 0:
                print(f"epoch: {epoch}, train_loss: {np.mean(train_loss):.4f}, valid_loss: {np.mean(valid_loss):.4f}")

        results[fold + 1] = {"train_loss": np.round(np.mean(train_loss), 4),
                             "valid_loss": np.round(np.mean(valid_loss), 4)
                             }

        if not os.path.exists("../models"):
            os.mkdir("../models")
        model_name = "../models/{}_fold_model_state_dict.pth".format(fold + 1)
        torch.save(model.state_dict(), model_name)
        if np.mean(valid_loss) < best_mse:
            best_mse = np.mean(valid_loss)
            torch.save(model.state_dict(), best_model_path)

    print("\nTraining Time: {:.4f}\n".format(time.time() - start))
    return results

if __name__ == "__main__":

    # set parameter
    seed = 42
    epochs = 1000
    n_folds = 7
    set_seed(seed)
    real = True  # Using real dataset or not

    # dataset for training
    dataset = dataset.FatigueDataset(real)
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

    print("\n{} Kfold mean MSE : {}".format(n_folds, final_mse / n_folds))
    print("best fold : {}, best_mse: {}".format(best_fold, best_mse))