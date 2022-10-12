import cv2
import argparse
import os


parser = argparse.ArgumentParser(description='Video to frames')
parser.add_argument('-n', '--name', type=str, required=True,
                    help='name of RGB video')
args = parser.parse_args()
name = args.name


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory ' + directory)

createFolder(name)

cap = cv2.VideoCapture("{}.avi".format(name))
frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("frame count : ", frame_cnt)

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow("{}".format(name), frame)
    cv2.imwrite("{}/{}.png".format(name, i), frame)
    print("{}.png".format(i))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    i += 1
cap.release()
cv2.destroyAllWindows()
