import os
import sys
import time
import model
import torch
import numpy as np
import threading
from collections import deque

Fatigue_score = 0.0
stress = 0.0
sdnn = 0.0
lf = 0.0
hf = 0.0
lf_hf = 0.0
hr = 0.0

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Jetson import serial_server

def throw(data):
    print(data)
    return


def main():
    '''
    from jetson to Raspberry
    '''
    global Fatigue_score, stress, lf, hf, lf_hf, sdnn, hr

    q = deque([])
    if not os.path.exists("./data"):
        os.mkdir("./data")
    DataPath = "./data/data.txt"
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
    state_dict = "./best_mse_model.pth"
    Model.load_state_dict(state_dict)

    while True:
        if len(q) >= 1:
            with open(DataPath, 'r') as f:
                data = f.readlines()[-1].strip()

            data = torch.FloatTensor(np.array(list(map(float, data))))
            Fatigue_score = Model(data)
            pring(data)
            q.popleft()

    return


if __name__ == "__main__":
    main()
