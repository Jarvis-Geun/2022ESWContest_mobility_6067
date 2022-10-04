from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
import random
from Deepphys_dataset import *
from model import *
import os
from preprocesss import *
import pandas as pd
import scipy.signal

DEVICE = torch.device("cuda:3")


def save_checkpoint(ep, net, opt, filename):
    state = {'Epoch': ep,
             'State_dict': net.state_dict(),
             'optimizer': opt.state_dict()
             }
    torch.save(state, filename)


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def pooling(path, num_frames):
    data = open(path)
    lines = data.readlines()
    signal = []
    for line in range(len(lines)):
        if len(signal) > num_frames*40-1:
            break

        lines[line] = lines[line].strip()
        signal.append(float(lines[line]))

    resampled_signal = scipy.signal.resample(signal, num_frames)
    label = np.array(resampled_signal)

    delta_label = []
    for i in range(label.shape[0] - 1):
        delta_label.append(label[i + 1] - label[i])

    data.close()
    npy = np.array(delta_label)

    return npy


def train():
    model.train()
    train_loss_epoch = []

    for i in range(len(train_data)):
        raw_video = Deepphys_preprocess_Video(train_data[i])
        BP_path = train_data[i].split('video')[0] + 'signal' + train_data[i].split('video')[1] + '/BP_mmHg.txt'
        BP_np = pooling(BP_path, raw_video.shape[0]+1)
        dataset = DeepPhysDataset(raw_video[:,:,:,3:], raw_video[:,:,:,:3], BP_np)
        Train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)

        for itr, data in enumerate(Train_loader):
            inputs, target = data
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())

    return np.mean(train_loss_epoch)


def test():
    model.eval()
    test_loss_epoch = []

    with torch.no_grad():
        for i in range(len(test_data)):
            raw_video = Deepphys_preprocess_Video(test_data[i])
            BP_path = test_data[i].split('video')[0] + 'signal' + test_data[i].split('video')[1] + '/BP_mmHg.txt'
            BP_np = pooling(BP_path, raw_video.shape[0] + 1)
            dataset = DeepPhysDataset(raw_video[:, :, :, 3:], raw_video[:, :, :, :3], BP_np)
            Test_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)

            for itr, data in enumerate(Test_loader):
                inputs, target = data
                output = model(inputs)
                loss = loss_func(output, target)
                test_loss_epoch.append(loss.item())

    return np.mean(test_loss_epoch)


data_path = '/hdd1/MMSE-HR/video/'

video_path_list = []
for (root, directories, files) in os.walk(data_path):
    for d in directories:
        if 'T' in d:
            d_path = os.path.join(root, d)
            video_path_list.append(str(d_path))

#random.shuffle(video_path_list)
data_list = sorted(video_path_list)
slicing = int((len(video_path_list))*0.8)
train_data = video_path_list[:slicing]
test_data = video_path_list[slicing:]

BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EPOCH = 100

record_name = '_100401'

model = DeepPhys().to(DEVICE)
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

min_loss = 1000000
record_list = []
start_time = datetime.now()
for epoch in range(EPOCH):
    train_loss = train()
    test_loss = test()
    print(f'| EPOCH:{epoch + 1} | train loss:{train_loss:.5f} | test loss:{test_loss:.5f}')

    record_list.append([epoch + 1, train_loss, test_loss])
    df = pd.DataFrame(record_list, columns=["Epoch", "train loss", 'test_loss'])
    df.to_excel('./record/Deepphys' + '_BS' + str(BATCH_SIZE) + '_LR' + str(LEARNING_RATE) + record_name + '.xlsx')

    if test_loss < min_loss:
        save_checkpoint(epoch, model, optimizer,
                        './state_dict/Deepphys_stdt' + '_BS' + str(BATCH_SIZE) +
                        '_LR' + str(LEARNING_RATE) + record_name +'.tar')
        min_loss = test_loss
        print('Model Statedictionary Saved')

duration = datetime.now() - start_time
print('test data list : ', test_data)
print('Train finished, duration : ', duration)
#
# model.apply(weight_reset)
#
# checkpoint = torch.load('./state_dict/Deepphys_stdt' + '_BS' + str(BATCH_SIZE) +
#                         '_LR' + str(LEARNING_RATE) + record_name +'.tar')
# model.load_state_dict(checkpoint['State_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
#
# test_loss = test()