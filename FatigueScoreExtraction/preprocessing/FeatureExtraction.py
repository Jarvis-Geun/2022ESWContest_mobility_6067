import dlib
import cv2
import math

def distance(pair:dict):
    point1 = pair["point1"]
    point2 = pair["point2"]
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def eyelid(eye_landmark):

    pair1 = {
        "point1" : eye_landmark[1],
        "point2" : eye_landmark[5]
    }

    pair2 = {
        "point1": eye_landmark[2],
        "point2": eye_landmark[4]
    }

    pair3 = {
        "point1": eye_landmark[0],
        "point2": eye_landmark[3]
    }
    EAR = (distance(pair1) + distance(pair2)) / (2*distance(pair3))
    return  EAR

def yawning(mouth_outline_landmark):

    pair1 = {
        "point1": mouth_outline_landmark[0],
        "point2": mouth_outline_landmark[6]
    }

    pair2 = {
        "point1": mouth_outline_landmark[3],
        "point2": mouth_outline_landmark[9]
    }

    MAR = distance(pair2)/distance(pair1)
    return MAR

def main(vid_path):
    ALL = list(range(0, 68))
    RIGHT_EYEBROW = list(range(17, 22))
    LEFT_EYEBROW = list(range(22, 27))
    RIGHT_EYE = list(range(36, 42))
    LEFT_EYE = list(range(42, 48))
    NOSE = list(range(27, 36))
    MOUTH_OUTLINE = list(range(48, 61))
    MOUTH_INNER = list(range(61, 68))
    JAWLINE = list(range(0, 17))

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    vid = cv2.VideoCapture(vid_path)
    delay = int(1000/30)
    while True:
        ret, img = vid.read()
        if ret is None:
            break

        img = cv2.resize(img, dsize=(640,480), interpolation = cv2.INTER_AREA)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_detector = detector(gray_img, 1)
        cnt = 0
        for face in face_detector:
            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0,255,255), 3)
            landmarks = predictor(img, face)

            left_eye_landmark = []
            right_eye_landmark = []
            mouth_outline_landmark = []
            for idx, p in enumerate(landmarks.parts()):
                if 36<=idx<=68:
                    cv2.circle(img, (p.x, p.y), 2, (0,255,0), -1)
                    if 36<=idx<42:
                       right_eye_landmark.append([p.x, p.y])
                    elif 42<=idx<48:
                       left_eye_landmark.append([p.x, p.y])
                    elif 48<=idx<61:
                       mouth_outline_landmark.append([p.x, p.y])
                    else:
                       continue

            left_EAR = eyelid(left_eye_landmark)
            right_EAR = eyelid(right_eye_landmark)
            EAR_mean = (left_EAR + right_EAR) / 2
            MAR = yawning(mouth_outline_landmark)

            EAR_txt = "EAR : {:.4f}".format(EAR_mean)
            MAR_txt = "MAR : {:.4f}".format(MAR)
            cv2.putText(img, EAR_txt, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)
            cv2.putText(img, MAR_txt, (50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)

        cv2.imshow('result', img)
        key = cv2.waitKey(delay)
        if key == 27:
            break
    return

if __name__ == "__main__":

    vid_path = "../data/FaceSampleVid.mp4"
    main(vid_path)