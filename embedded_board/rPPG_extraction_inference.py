from sklearn.preprocessing import minmax_scale
import cv2
import numpy as np
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from model import *
from scipy.signal import butter, lfilter
import time
import argparse

from camera.JetsonCamera import Camera
from camera.Focuser import Focuser
from camera.Autofocus import FocusState, doFocus


def parse_cmdline():
    parser = argparse.ArgumentParser(description='Get RGB video and PPG for training Deep Learning model')
    parser.add_argument('-i', '--i2c-bus', type=int, nargs=None, required=False, default=7,
                        help='Set i2c bus, for A02 is 6, for B01 is 7 or 8, for Jetson Xavier NX it is 9 and 10.')
    parser.add_argument('-v', '--verbose', action="store_true", help='Print debug info.')
    return parser.parse_args()


def generate_Floatimage(frame):
    '''
    :param frame: roi frame
    :return: float value frame [0 ~ 1.0]
    '''
    dst = img_as_float(frame)
    dst = cv2.cvtColor(dst.astype('float32'), cv2.COLOR_BGR2RGB)
    dst[dst > 1] = 1
    dst[dst < 0] = 0
    return dst


def generate_MotionDifference(prev_frame, crop_frame):
    '''
    :param prev_frame: previous frame
    :param crop_frame: current frame
    :return: motion diff frame
    '''
    # motion input
    motion_input = (crop_frame - prev_frame) / (crop_frame + prev_frame)
    # TODO : need to diminish outliers [ clipping ]
    # motion_input = motion_input / np.std(motion_input)
    # TODO : do not divide each D frame, modify divide whole video's unit standard deviation
    return motion_input


def normalize_Image(frame):
    '''
    :param frame: image
    :return: normalized_frame
    '''
    return frame / np.std(frame)


def preprocess_Image(prev_frame, crop_frame):
    '''
    :param prev_frame: previous frame
    :param crop_frame: current frame
    :return: motion_differnceframe, normalized_frame
    '''
    return generate_MotionDifference(prev_frame, crop_frame), normalize_Image(prev_frame)


def video_normalize(channel):
    channel /= np.std(channel)
    return channel


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

args = parse_cmdline()

model = DeepPhys().to(DEVICE)

checkpoint = torch.load('Deepphys_stdt_BS128_LR1.0_100805.tar', map_location='cuda:0')
model.load_state_dict(checkpoint['State_dict'])

### get CSI camera ###
camera = Camera()
focuser = Focuser(args.i2c_bus)
focuser.verbose = args.verbose

focusState = FocusState()
focusState.verbose = args.verbose
doFocus(camera, focuser, focusState)


'''
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
'''

width = 640
height = 360

raw_video = np.empty((36, 36, 6))
prev_frame = None

lowcut = 0.7
highcut = 2.5
Fs = 25

prev_time = 0

rPPG = np.array([0 for i in range(100)])
fig = plt.figure(figsize=(8, 2))
while True:
#    ret, frame = cap.read()
    frame = camera.getFrame(2000)
    current_time = time.time() - prev_time

    if (frame is not None) and (current_time > 1. / Fs):
        ## pre-processing, thread 1
        crop_frame_ = frame[:, int(width / 2) - int(height / 2 + 1):int(height / 2) + int(width / 2), :]
        crop_frame = cv2.resize(crop_frame_, dsize=(36, 36), interpolation=cv2.INTER_AREA)
        crop_frame = generate_Floatimage(crop_frame)
        if prev_frame is None:
            prev_frame = crop_frame
            continue
        raw_video[:, :, :3], raw_video[:, :, -3:] = preprocess_Image(prev_frame, crop_frame)
        prev_frame = crop_frame

        ## rPPG extraction, thread 2
        appearance_data = torch.tensor(np.transpose(raw_video[:, :, 3:], (2, 0, 1))).float().to(DEVICE)
        motion_data = torch.tensor(np.transpose(raw_video[:, :, :3], (2, 0, 1))).float().to(DEVICE)
        inputs = torch.stack([appearance_data, motion_data], dim=0)
        output = model(torch.unsqueeze(inputs, 0))
        print(output.item())

        ## plotting
        rPPG = rPPG[1:]
        rPPG = np.append(rPPG, output.item())
        # rPPG = butter_bandpass_filter(rPPG, lowcut, highcut, Fs, order=5)
        # rPPG = minmax_scale(rPPG)
        # peaks = argrelextrema(rPPG, np.greater, order=7)
        plt.plot(rPPG)
        # plt.plot(peaks[0], rPPG[peaks], 'x')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        plt.tight_layout()

        fig_ = figure_to_array(fig)
        fig_ = cv2.cvtColor(np.uint8(fig_), cv2.COLOR_RGBA2BGR)

        face_img = cv2.resize(crop_frame_, dsize=(200, 200), interpolation=cv2.INTER_AREA)

        full = np.hstack((face_img, fig_))
        cv2.imshow('frame', full)
        # cv2.imshow("fig", fig_)
        plt.cla()

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        if key == ord('f'):
            if focusState.isFinish():
                focusState.reset()
                doFocus(camera, focuser, focusState)
            else:
                print("Focus is not done yet.")

# cap.release()
cv2.destroyAllWindows()
