from imutils import face_utils
import dlib
import cv2
import math
import pandas as pd
import numpy as np
import natsort

def len_xy(p1, p2):
    ret = ((p2[0] - p1[0])**2) + ((p2[1] - p1[1])**2)
    return math.sqrt(ret)

def facial_feature(path):
    global predictor, detector
    ex_blink, blink, frame_cnt = 0, 0, 0
    yawn = False
    blink_cnt, yawn_cnt = 0, 0
    image_list = natsort.natsorted(os.listdir(path))
    image_list = image_list[125:-125]
    for i in image_list:
        df = pd.DataFrame(columns=['PERCLOSE', 'Excessive Blink', 'Yawn'])
        gray = cv2.imread(path + i, cv2.IMREAD_GRAYSCALE)

        # 영상에 얼굴이 없는 경우 continue
        if  len(detector(gray, 0)) == 0: continue
        else:
            frame_cnt += 1
            rect = detector(gray, 0)[0]
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            ear = (len_xy(shape[37], shape[41]) + len_xy(shape[38], shape[40])) / (2*len_xy(shape[36], shape[39]))
            mar = len_xy(shape[51], shape[57]) / len_xy(shape[48], shape[54])
            
            if ear <= 0.18:
                blink_cnt += 1
                blink += 1
                if blink_cnt >= 25: # 1초 이상 눈을 감는 경우 -> excessive blink
                    ex_blink += 1
                    blink_cnt = 0
            else:
                blink_cnt = 0
            if mar >= 0.6:
                yawn_cnt += 1
                if yawn_cnt >= 50: # 2초 이상 입을 벌리면 하품을 하고 있다 판단
                    yawn = True
            else:
                yawn_cnt = 0
    facial_feature = np.array([blink/frame_cnt, ex_blink, yawn])
    return facial_feature
