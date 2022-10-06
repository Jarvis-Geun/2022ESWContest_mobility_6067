from imutils import face_utils
import dlib
import cv2
import math
import pandas as pd

data_path =  'C:\\Users\\hyeon\\Desktop\\임베디드\\data' 
filename =  'video.mp4'
save_path = 'C:\\Users\\hyeon\\Desktop\\임베디드\\feature' 

# 변수 초기화
ex_blink, blink, frame_cnt = 0, 0, 0
yawn = False
blink_cnt, yawn_cnt = 0, 0

# 두 점 사이 거리 구하기
def len_xy(p1, p2):
    ret = ((p2[0] - p1[0])**2) + ((p2[1] - p1[1])**2)
    return math.sqrt(ret)

# dlib으로 landmark 구하기
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

df = pd.DataFrame(columns=['PERCLOSE', 'Excessive Blink', 'Yawn'])
cap = cv2.VideoCapture(data_path + filename)

# 첫 번재 프래임 버리기 (첫 번째 ppg 데이터에 오류가 너무 많음)
_, _ = cap.read()

if cap.isOpened():
    while True:
        ret, image = cap.read()
        if frame_cnt == 3000: # 3000 frame 마다 데이터 저장 + 초기화
            print(len(df) +1, ':', blink/frame_cnt, ex_blink, yawn)
            df = df.append({'PERCLOSE' : blink/frame_cnt, 'Excessive Blink' : ex_blink, 'Yawn': yawn}, ignore_index = True)
            ex_blink, blink, frame_cnt = 0, 0, 0
            yawn = False
            if len(df) == 6: break # 한 영상에 Feature 6개
        
        if ret:
            # cv2.imshow("ori", image)
            frame_cnt += 1
            print(frame_cnt)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if  len(detector(gray, 0)) == 0: continue

            else:
                rect = detector(gray, 0)[0]
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # cv2.imshow("landmark", image)
                ear = (len_xy(shape[37], shape[41]) + len_xy(shape[38], shape[40])) / (2*len_xy(shape[36], shape[39]))
                mar = len_xy(shape[51], shape[57]) / len_xy(shape[48], shape[54])
                if ear <= 0.2:
                    blink_cnt += 1
                    blink += 1
                    if blink_cnt >= 25: # 1초 이상 눈을 감는 경우 -> excessive blink
                        ex_blink += 1
                        blink_cnt += 0
                else:
                    blink_cnt = 0
                        
                if mar >= 0.6:
                    yawn_cnt += 1
                    if yawn_cnt >= 50: # 2초 이상 입을 벌리면 하품을 하고 있다 판단
                        yawn = True
                else:
                    yawn_cnt = 0

            cv2.waitKey(1)
        else:
            break


print(df)
df.to_csv(filename[:-4] + '.csv')