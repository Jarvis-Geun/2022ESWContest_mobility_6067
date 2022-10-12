import os

import cv2
import numpy as np
from skimage.util import img_as_float
# from test import plot_graph_from_image,get_graph_from_image
from sklearn import preprocessing
from tqdm import tqdm

def Deepphys_preprocess_Video(path):
    '''
    :param path: dataset path
    :param flag: face detect flag
    :return: [:,:,:0-2] : motion diff frame
             [:,:,:,3-5] : normalized frame
    '''

    total_img = sorted(os.listdir(path))
    raw_video = np.empty((len(total_img) - 1, 36, 36, 6))
    prev_frame = None
    height = 1392
    width = 1040
    j = 0

    for image in range(len(total_img)):
        f_path = os.path.join(path, total_img[image])
        frame = cv2.imread(f_path)
        crop_frame = frame[int(height / 2) - int(width / 2 + 1):int(width / 2) + int(height / 2), :, :]
        crop_frame = cv2.resize(crop_frame, dsize=(36, 36), interpolation=cv2.INTER_AREA)
        crop_frame = generate_Floatimage(crop_frame)
        if prev_frame is None:
            prev_frame = crop_frame
            continue
        raw_video[j, :, :, :3], raw_video[j, :, :, -3:] = preprocess_Image(prev_frame, crop_frame)
        prev_frame = crop_frame
        j += 1
    raw_video[:, :, :, 3] = video_normalize(raw_video[:, :, :, 3])
    return raw_video

    # cap = cv2.VideoCapture(path)
    # frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # raw_video = np.empty((frame_total - 1, 36, 36, 6))
    # prev_frame = None
    # j = 0
    #
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if frame is None:
    #         break
    #     else:
    #         crop_frame = frame[:, :, int(height / 2) - int(width / 2 + 1):int(width / 2) + int(height / 2)]
    #
    #     crop_frame = cv2.resize(crop_frame, dsize=(36, 36), interpolation=cv2.INTER_AREA)
    #     crop_frame = generate_Floatimage(crop_frame)
    #
    #     if prev_frame is None:
    #         prev_frame = crop_frame
    #         continue
    #     raw_video[j, :, :, :3], raw_video[j, :, :, -3:] = preprocess_Image(prev_frame, crop_frame)
    #     prev_frame = crop_frame
    #     j += 1
    # raw_video[:, :, :, 3] = video_normalize(raw_video[:, :, :, 3])
    # cap.release()
    #
    # return raw_video


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
