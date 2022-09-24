# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

from ppg_from_serial import ppg_from_serial
import cv2
import serial
import time

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def show_camera():
    window_title = "CSI Camera"

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            f = open("ppg.txt", 'w')
            i = 0
            while True:
                ret_val, frame = video_capture.read()

                ### get "ppg" from serial ###
                s1 = time.time()
                ppg, ppg_time = ppg_from_serial()
                e1_time = time.time()
                e1 = time.time() - s1

                ### save "ppg" in text ###
                f.write(ppg + ' ' + ppg_time + '\n')
                
                s2 = time.time()
                cv2.imwrite("get_PPG_GTD/frames/{}.png".format(i), frame)
                print("delay between serial and frame\n", time.time() - e1_time)
                e2 = time.time() - s2

                print("serial elapsed time : {}\nframe elapsed : {}".format(e1, e2))

                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break
                
                i += 1
        finally:
            f.close()
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    py_serial = serial.Serial(
    port='/dev/ttyACM0',
    # Baud rate (speed of communication)
    baudrate=9600,
    )

    show_camera()