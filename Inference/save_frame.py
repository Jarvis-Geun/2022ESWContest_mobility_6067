from venv import create
import cv2
import os


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory.' + directory)


def save_frames(path, frame, cnt):
    directory = path + '/frames'
    createFolder(directory)

    cv2.imwrite("{}/frames/{}.png".format(path, cnt), frame)
    print("{}.png".format(cnt))