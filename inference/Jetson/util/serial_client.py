import serial
import random
import time


# shape of features : 9
# type of features : list(float)
def serial_client(client_port, features):
    i = 0
    while (True):
        for i in range(len(features)):
            if i < len(features) - 1:
                send_msg = str(features[i]) + ','
                send_msg = send_msg
            elif i == len(features) - 1:
                send_msg = str(features[i]) + '\n'

            client_port.write(send_msg.encode('utf-8'))
