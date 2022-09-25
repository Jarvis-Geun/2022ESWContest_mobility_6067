import cv2

cap = cv2.VideoCapture("output.avi")
frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("frame count : ", frame_cnt)

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    cv2.imwrite("frames/{}.png".format(i), frame)
    print("{}.png".format(i))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    i += 1
cap.release()
cv2.destroyAllWindows()
