import time

import cv2 as cv

cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
fourcc = cv.VideoWriter_fourcc(*'MJPG')
start = -6
# saveVideoPath = 'video_' + str(start)+'.avi'
# out = cv.VideoWriter(saveVideoPath, fourcc, 30.0, (640, 480))
while cam.isOpened():
    saveVideoPath = 'video_' + str(start) + '.avi'
    out = cv.VideoWriter(saveVideoPath, fourcc, 30.0, (640, 480))

    ret, frame = cam.read()
    show = frame.copy()
    # show = cv.flip(show, 1, dst=None)
    # show = cv.rectangle(show, (450, 50), (1550, 1000), (152, 54, 255), 4, 4)
    # cam.set(3, 1920)  # width=1920
    # cam.set(4, 1080)  # height=1080
    # # method 2:
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cv.namedWindow("test", cv.WINDOW_FREERATIO)
    # show = cv.rectangle(frame, (450, 50), (1550, 1000), (152, 54, 255), 4, 4)
    cv.imshow("test", show)
    if not ret:
        break
    key = cv.waitKey(1) & 0xFF

    # out.write(frame)

    if key == 32:
        num = 0
        # press ESC to escape (ESC ASCII value: 27)
        while num <= 90:
            rets, frames = cam.read()
            cv.waitKey(1)
            out.write(frames)
            num = num+1
        start = start + 2
    elif key == 27:
        # press ESC to escape (ESC ASCII value: 27)
        print("Escape hit, closing...")
        break
    else:
        pass

# np_cali_res = np.array(cali_res)
# np.save('cali_theta_data.npy', np_cali_res)

cam.release()
out.release()
cv.destroyAllWindows()


