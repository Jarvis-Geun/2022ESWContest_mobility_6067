'''
아두이노에 연결된 PPG 센서(PSL iPPG2C)의 PPG 값을
젯슨나노-아두이노 간의 Serial 통신을 사용하여 획득한다.
'''

# Install "pyserial" : python3 -m pip install pyserial
import serial
import time

def ppg_from_serial(py_serial):
    if py_serial.readable():
        # read line by line if inputs are available
        ppg = py_serial.readline()
        # return first 5 values(except \r\n) and current time => ex. 1.569, current_time
        return ppg[:len(ppg)-1].decode(), round(time.time(), 4)

if __name__ == "__main__":
    py_serial = serial.Serial(
        port='/dev/ttyACM0',
        # Baud rate (speed of communication)
        baudrate=9600,
    )

    while True:
        print(ppg_from_serial(py_serial))