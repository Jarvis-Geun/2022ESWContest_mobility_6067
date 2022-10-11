import serial
import random

print('serial ' + serial.__version__)

# shape of features : 9
# type of features : list(float)
def serial_client(client_port, features):
    i = 1
    while (True):
        for i in range(len(features)):
            if i < len(features) - 1:
                send_msg = str(features[i]).encode('utf-8') + b','
            elif i == len(features) - 1:
                send_msg = b'\n'
            
            # features = str(random.random()).encode('utf-8')
            client_port.write(send_msg)

if __name__=="__main__":
    # Set a PORT Number & baud rate
    PORT = '/dev/ttyTHS1'
    BaudRate = 9600

    client_port = serial.Serial(PORT, BaudRate)