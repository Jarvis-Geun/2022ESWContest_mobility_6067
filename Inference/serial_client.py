import serial
import random

print('serial ' + serial.__version__)

# Set a PORT Number & baud rate
PORT = '/dev/ttyTHS1'
BaudRate = 9600

ARD= serial.Serial(PORT, BaudRate)

i = 1
while (True):
    features = str(random.random()).encode('utf-8')

    if i % 10 == 0:
        features += b'\n'

    ARD.write(features)

    i += 1
