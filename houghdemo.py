import cv2
import numpy as np


# 两个回调函数
def HoughLinesP(minLineLength):
    tempIamge = src.copy()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=minLineLength, minLineLength=180, maxLineGap=20)
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(tempIamge, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print(lines)
    cv2.imshow(window_name, tempIamge)


# 临时变量
minLineLength = 20

# 全局变量
# minLINELENGTH = 20
window_name = "HoughLines Demo"

# 读入图片，模式为灰度图，创建窗口
src = cv2.imread("opencv_frame_5.png")
src = src[0:480, 1:640]

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
print(gray.shape)
img = cv2.GaussianBlur(gray, (3, 3), 0)
edges = cv2.Canny(img, 50, 150, apertureSize=3)
cv2.namedWindow(window_name)

# 初始化
HoughLinesP(minLineLength)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
