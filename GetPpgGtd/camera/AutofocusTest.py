import time
import signal
import cv2
import threading
import argparse
import numpy as np

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

    parser.add_argument('-i', '--i2c-bus', type=int, nargs=None, required=True,
                        help='Set i2c bus, for A02 is 6, for B01 is 7 or 8, for Jetson Xavier NX it is 9 and 10.')

    parser.add_argument('-v', '--verbose', action="store_true", help='Print debug info.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_cmdline()
    camera = Camera()
    focuser = Focuser(args.i2c_bus)
    focuser.verbose = args.verbose

    focusState = FocusState()
    focusState.verbose = args.verbose
    doFocus(camera, focuser, focusState)

    start = time.time()
    frame_count = 0

    i = 0
    img_memory = []
    while not exit_:
        frame = camera.getFrame(2000)

        cv2.imshow("Test", frame)
#        cv2.imwrite("frames/{}.png".format(i), frame)
        img_memory.append(frame)
        print("len(img_memory) : ", len(img_memory))

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
        
        i += 1

    memory = np.array(img_memory)
    np.save("ppg/frames/memory", memory)
    print("memory.shape : ", memory.shape)
    camera.close()
