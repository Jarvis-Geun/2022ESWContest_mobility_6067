from imutils import face_utils
import dlib
import cv2
import math
import pandas as pd
import numpy as np

import os
import sys
import time
from tqdm import tqdm
from IPython.display import display

# 두 점 사이 거리 구하기
def len_xy(p1, p2):
    ret = ((p2[0] - p1[0])**2) + ((p2[1] - p1[1])**2)
    return math.sqrt(ret)

# dlib으로 landmark 구하기
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

def OneVidProcessing(path):
    # 변수 초기화
    ex_blink, blink, frame_cnt = 0, 0, 0
    yawn = False
    blink_cnt, yawn_cnt = 0, 0

    cap = cv2.VideoCapture(path)
    n_Frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []
    # 첫 번재 프래임 버리기 (첫 번째 ppg 데이터에 오류가 너무 많음)
    _, _ = cap.read()

    if cap.isOpened():
        with tqdm(total = n_Frame - 1) as pbar:
            while True:
                ret, image = cap.read()
                if frame_cnt == 3000: # 3000 frame 마다 데이터 저장 + 초기화
                    print(len(results) + 1, ':', blink/frame_cnt, ex_blink, yawn)
                    results.append([blink/frame_cnt, ex_blink, yawn])
                    # df = df.append({'PERCLOSE' : blink/frame_cnt, 'Excessive Blink' : ex_blink, 'Yawn': yawn}, ignore_index = True)
                    ex_blink, blink, frame_cnt = 0, 0, 0
                    yawn = False
                    if len(results) == 6: break # 한 영상에 Feature 6개

                if ret:
                    # cv2.imshow("ori", image)
                    frame_cnt += 1
                    # print(frame_cnt)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    if len(detector(gray, 0)) == 0: continue

                    else:
                        rect = detector(gray, 0)[0]
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # cv2.imshow("landmark", image)
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

                    cv2.waitKey(1)
                    pbar.update(1)
                    pbar.set_description("frame: {}".format(frame_cnt))
                else:
                    break

    return_df = pd.DataFrame(np.array(results), columns=['PERCLOSE', 'Excessive Blink', 'Yawn'])
    return return_df


if __name__ == "__main__":

    if not os.path.exists("../data"):
        os.mkdir("../data")

    data_path = "../../get_PPG_GTD"
    a_start = time.time()
    if len(sys.argv) == 1:
        final_path = "../data/FacialFeature.csv"
        folders = ["PGH", "geun", "jho", "phj", "hwang"]
        final_df = pd.DataFrame(columns=['PERCLOSE', 'Excessive Blink', 'Yawn'])

        for folder in tqdm(sorted(folders)):
            try:
                print(" =========== Folder: {} ===========".format(folder))
                videos = [folder+str(i)+".avi" for i in range(1, 11)]
                if folder == 'hwang':
                    videos = videos[:-1]
                for data in videos:
                    start = time.time()
                    df = OneVidProcessing(os.path.join(*[data_path, folder, data]))
                    final_df = pd.concat([final_df, df], axis=0)
                    print("[VidName] {}, [Processing Time]: {:.4f}".format(data, time.time()-start))

            except FileNotFoundError:
                print("===== Check your data folder & file name =====")
                print("[FileNotFoundError] {}".format(os.path.join(*[data_path, folder, data])))
                sys.exit()

        final_df.reset_index().to_csv(final_path, index=False)
        display(final_df.info())

    else:
        if sys.argv[1] == 'concat':
            folders = ["PGH", "geun", "jho", "phj", "hwang"]
            final_path = "../data/FacialFeature.csv"
            final_df = pd.DataFrame(columns=['PERCLOSE', 'Excessive Blink', 'Yawn'])
            for folder in sorted(folders):
                file_path = "../data/{}_FacialFeature.csv".format(folder)
                df = pd.read_csv(file_path)
                final_df = pd.concat([final_df, df], axis = 0)

            final_df = final_df.drop(labels="index", axis = 1).reset_index(drop=True)
            final_df.to_csv(final_path, index=False)
            display(final_df)
            sys.exit()

        print("Processing Folder: {}".format(sys.argv[1:]))
        for folder_name in sys.argv[1:]:
            print(" =========== Folder: {} ===========".format(folder_name))
            final_df = pd.DataFrame(columns=['PERCLOSE', 'Excessive Blink', 'Yawn'])
            final_path = "../data/{}_FacialFeature.csv".format(folder_name)
            try:
                videos = [folder_name + str(i) + ".avi" for i in range(1, 11)]
                if folder_name == 'hwang':
                    videos = videos[:-1]
                for data in tqdm(videos):
                    start = time.time()
                    df = OneVidProcessing(os.path.join(*[data_path, folder_name, data]))
                    final_df = pd.concat([final_df, df], axis=0)
                    print("[VidName] {}, [Processing Time]: {:.4f}".format(data, time.time()-start))

            except FileNotFoundError:
                print("===== Check your data folder & file name =====")
                print("[FileNotFoundError] {}".format(os.path.join(*[data_path, folder_name, data])))
                sys.exit()

            final_df.reset_index().to_csv(final_path, index=False)
            display(final_df.info())

    print("processing time: {:.4f} sec".format(time.time() - a_start))