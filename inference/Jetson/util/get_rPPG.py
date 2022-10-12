import cv2
import numpy as np
from skimage.util import img_as_float
from scipy.signal import butter, lfilter
from sklearn.preprocessing import minmax_scale
import os
from tqdm import tqdm
import natsort
from ..model import *


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


def get_rPPG(path):
    img_path_list = natsort.natsorted(os.listdir(path))
    img_path_list = img_path_list[125:-125]

    width = 640
    height = 360
    raw_video = np.empty((36, 36, 6))
    prev_frame = None

    lowcut = 0.7
    highcut = 2.5
    Fs = 25

    model = DeepPhys().to(DEVICE)
    checkpoint = torch.load('../model/deepphys_model.tar', map_location='cuda:0')
    model.load_state_dict(checkpoint['State_dict'])

    rPPG = ([])
    for i in tqdm(range(len(img_path_list))):
        frame = cv2.imread(os.path.join(path, img_path_list[i]))

        ## pre-processing, thread 1
        crop_frame = frame[:, int(width / 2) - int(height / 2 + 1):int(height / 2) + int(width / 2), :]
        crop_frame = cv2.resize(crop_frame, dsize=(36, 36), interpolation=cv2.INTER_AREA)
        crop_frame = generate_Floatimage(crop_frame)
        if prev_frame is None:
            prev_frame = crop_frame
            continue
        raw_video[:, :, :3], raw_video[:, :, -3:] = preprocess_Image(prev_frame, crop_frame)
        prev_frame = crop_frame
        raw_video[:, :, 3] = video_normalize(raw_video[:, :, 3])

        ## rPPG extraction, thread 2
        appearance_data = torch.tensor(np.transpose(raw_video[:, :, 3:], (2, 0, 1))).float().to(DEVICE)
        motion_data = torch.tensor(np.transpose(raw_video[:, :, :3], (2, 0, 1))).float().to(DEVICE)
        inputs = torch.stack([appearance_data, motion_data], dim=0)
        output = model(torch.unsqueeze(inputs, 0))

        rPPG = np.append(rPPG, output.item())

    rPPG = butter_bandpass_filter(rPPG, lowcut, highcut, Fs, order=6)
    rPPG = minmax_scale(rPPG)

    return rPPG
