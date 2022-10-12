import cv2
import os
import numpy as np
import dlib
import serial

import util.get_facial_feature
import util.get_rppg_feature
import util.get_rPPG
from util.save_frames import save_frames
from util.serial_client import serial_client
from camera.JetsonCamera import Camera
from camera.Focuser import Focuser
from camera.Autofocus import FocusState, doFocus


# Jetson Nano
client_port = serial.Serial('/dev/ttyTHS1', 9600)

'''
# Raspberry Pi 4
server_port = serial.Serial(
    port="/dev/ttyS0",
    baudrate=9600,
)
'''

path = ''
image_folder = ''
image_list = []

'''
# Get CSI camera
camera = Camera()
7 : i2c bus
focuser = Focuser(7)

focusState = FocusState()
doFocus(camera, focuser, focusState)
'''

# 얼굴 landmark를 찾기 위한 객체 생성
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# 필요없으면 스킵
def return_image_list(path, image_folder):
    image_list  = []
    for filename in os.listdir(path + image_folder):
        ext = filename.split('.')[-1]
        if ext == 'png':
            image_list.append(filename)
    return image_list


# USB camera
cap = cv2.VideoCapture(1)
width = 640
height = 360
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# for saving frames
cnt = 0
while True:
    '''카메라에서 3000장의 이미지를 받아와서 image_list에 이미지 이름 저장'''
    root_dir = '/home/jetson/'
    
    '''
    # CSI camera
    frame = camera.getFrame(2000)
    '''

    val, frame = cap.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

    if frame is not None:
        save_frames(root_dir, frame, cnt)

    cnt += 1
    if cnt < 3250:
        continue

    path = root_dir + 'frames'

    print('Get facial feature')
    facial_features = get_facial_feature.facial_feature(path, detector, predictor)

    print('Start rPPG signal Extraction')
    rppg = get_rPPG.get_rPPG(path)

    rppg_features = get_rppg_feature.hrv_analysis(rppg[:1000])

    # feature 합치기
    input = np.append(facial_features, rppg_features)

    # 라즈베리 파이에 전송
    serial_client(client_port, input)

    # 모델 예측이 끝난 후에 저장한 프레임 제거 (메모리 절약)
    os.system("rm -rf {}".format(path))