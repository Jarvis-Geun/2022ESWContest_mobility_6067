import time
import signal
import cv2
import threading
import argparse
import numpy as np
import serial

from JetsonCamera import Camera
from Focuser import Focuser
from Autofocus import FocusState, doFocus
from ppg_from_serial import ppg_from_serial

exit_ = False
def sigint_handler(signum, frame):
    global exit_
    exit_ = True

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)

def parse_cmdline():
    parser = argparse.ArgumentParser(description='Arducam IMX519 Autofocus Demo.')

    parser.add_argument('-i', '--i2c-bus', type=int, nargs=None, required=True,
                        help='Set i2c bus, for A02 is 6, for B01 is 7 or 8, for Jetson Xavier NX it is 9 and 10.')

    parser.add_argument('-v', '--verbose', action="store_true", help='Print debug info.')

    return parser.parse_args()

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

    w = 640
    h = 360
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("output.avi", fourcc, fps, (w, h))

    start = time.time()
    start_time = time.time()
    frame_count = 0

    i = 0
    f = open("ppg_data.txt", 'w')

    while not exit_:
        print("elapsed time : ", time.time()-start_time)

        if time.time()-start_time > 60:
            print("60 seconds end")
            break
        frame = camera.getFrame(2000)

#        if time.time() - start < 3:
#            continue
        if py_serial.readable():
            ppg = py_serial.readline()
            f.write(str(i) + ' ' + str(ppg) + '\n')
            print("ppg : ", ppg)

            cv2.imshow("Test", frame)
#            cv2.imwrite("frames/{}.png".format(i), frame)
            out.write(frame)
            i += 1
            print("i : ", i)

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
