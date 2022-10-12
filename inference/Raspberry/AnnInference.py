import time
from model import model
import torch
import numpy as np
import threading
from collections import deque
import serial
import sys
import os

Fatigue_score = 0.0
stress = 0.0
sdnn = 0.0
lf = 0.0
hf = 0.0
lf_hf = 0.0
hr = 0.0

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import serial_server

def main():
    '''
    from jetson to Raspberry
    '''
    global Fatigue_score, stress, lf, hf, lf_hf, sdnn, hr

    root_path = '/home/ubuntu'

    q = deque([])
    if not os.path.exists(root_path + "/data"):
        os.mkdir(root_path + "/data")
    DataPath = root_path + "/data/data.txt"
    server_port = serial.Serial(
        port="/dev/ttyS0",
        baudrate=9600,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )

    t = threading.Thread(target=serial_server.serial_server, args=(server_port, DataPath, q))
    t.start()
    Model = model.LinearModel(9, 1)
    state_dict = "./state_dict/best_mse_model.pth"
    Model.load_state_dict(torch.load(state_dict))

    while True:
        time.sleep(5)
        with open(DataPath, 'r') as f:
            data = f.readlines()
        if len(data[-1]) == 0:
            data = data[-2]
        else:
            data = data[-1]

        data = np.array(list(map(float, data.split(","))))
        hr = data[0]
        sdnn = data[2]
        lf = data[3]
        hf = data[4]
        lf_hf = data[5]
        print(data, len(data))

        try:
            if len(data) == 9:
                data = torch.FloatTensor(data)
                Fatigue_score = Model(data)
                print("Fatigue_score : ", Fatigue_score)
        except:
            pass
    return

if __name__ == "__main__":
    main()