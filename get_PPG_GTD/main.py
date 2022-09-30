import time
import signal
import cv2
import threading
import argparse
import numpy as np
import serial
import os

from camera.JetsonCamera import Camera
from camera.Focuser import Focuser
from camera.Autofocus import FocusState, doFocus

exit_ = False
def sigint_handler(signum, frame):
    global exit_
    exit_ = True

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)

def parse_cmdline():
    parser = argparse.ArgumentParser(description='Arducam IMX519 Autofocus Demo.')

    parser.add_argument('-i', '--i2c-bus', type=int, nargs=None, required=False, default=7,
                        help='Set i2c bus, for A02 is 6, for B01 is 7 or 8, for Jetson Xavier NX it is 9 and 10.')

    parser.add_argument('-v', '--verbose', action="store_true", help='Print debug info.')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='path of rgb and ppg data')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='name of video and ppg')
    parser.add_argument('-t', '--time', type=int, required=True,
                        help='time of video')

    return parser.parse_args()


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory ' + directory)


if __name__ == "__main__":
    py_serial = serial.Serial(
        port='/dev/ttyACM0',
        # Baud rate (speed of communication)
        baudrate=9600,
    )

    args = parse_cmdline()
    camera = Camera()
    focuser = Focuser(args.i2c_bus)
    focuser.verbose = args.verbose

    focusState = FocusState()
    focusState.verbose = args.verbose
    doFocus(camera, focuser, focusState)

    path = args.path
    name = args.name
    len_time = args.time

    createFolder(path)

    w = 640
    h = 360
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("{}/{}.avi".format(path, name), fourcc, fps, (w, h))

    # for fps
    start = time.time()
    frame_count = 0

    # for elapsed time
    start_time = time.time()

    i = 0
    f = open("{}/{}.txt".format(path, name), 'w')

    while not exit_:
        print("elapsed time : {} seconds".format(time.time() - start_time))

        if time.time() - start_time > len_time:
            print("{} seconds end".format(len_time))
            break
        frame = camera.getFrame(2000)

        if py_serial.readable():
            ppg = py_serial.readline()
            f.write(str(i) + ' ' + str(ppg) + '\n')

            cv2.imshow("frame", frame)
            out.write(frame)
            i += 1
            print("frame count : ", i)

        key = cv2.waitKey(1)
        if key == ord('q'):
            exit_ = True
        if key == ord('f'):
            if focusState.isFinish():
                focusState.reset()
                doFocus(camera, focuser, focusState)
            else:
                print("Focus is not done yet.")

        frame_count += 1
        if time.time() - start >= 1:
            print("{}fps".format(frame_count))
            start = time.time()
            frame_count = 0

    camera.close()
    f.close()
