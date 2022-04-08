import cv2.cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

POINT = 25

h1 = []
h2 = []
sum1 = 0
sum2 = 0
img0 = cv.imread('data/opencv_frame_4_0.png', 0)
frame = img0[0:1080, 3:1919]
cv.namedWindow("test", cv.WINDOW_FREERATIO)
cv.imshow("test", frame)
frame = cv.bitwise_not(src=frame)
ret1, src = cv.threshold(frame, 28, 0, cv.THRESH_TOZERO)
# cv.namedWindow("test", cv.WINDOW_FREERATIO)
# cv.imshow("test", src)
test = cv.medianBlur(src, 3)
cv.namedWindow("test", cv.WINDOW_FREERATIO)
cv.imshow("test", test)

for i in range(0, 1916):
    for j in range(0, 1080):
        sum1 = sum1 + test[j, i]
    h1.append(sum1)
    sum1 = 0

for i in range(0, 1080):
    for j in range(0, 1916):
        sum2 = sum2 + test[i, j]
    h2.append(sum2)
    sum2 = 0

x0 = np.linspace(0, 1915, 1916)
x1 = np.linspace(0, 1079, 1080)

y0 = frame[10][0:1916]
y1 = h1
y2 = h2
plt.figure(1)
plt.title("gray value sum of each column of the image")
plt.xlabel("pixel")#x轴上的名字
plt.ylabel("sum of gray value")#y轴上的名字
plt.plot(x0, y1)
plt.figure(2)
plt.title("gray value sum of each row of the image")
plt.plot(x1, y2)
plt.xlabel("pixel")#x轴上的名字
plt.ylabel("sum of gray value")#y轴上的名字
plt.show()

cv.waitKey()
cv.destroyAllWindows()


