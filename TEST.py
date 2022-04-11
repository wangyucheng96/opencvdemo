import math
from collections import deque

from sklearn.metrics import mean_squared_error

from fit import *
from testfile import edgetest
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import mat4py
# import scipy.io as scio
# # res = np.zeros([8000, 6000, 1], np.uint8)
# # mat4py.loadmat('data/cross_img_20220119_12V.mat')
# b = scio.loadmat('data/cross_img_20220119_12V.mat')
# print(b)
from testfile.edgetest import *

ks = np.load('npdata/k.npy')
k_x = ks[0][0]
k_y = ks[0][1]
theta = ks[0][2]
p = ks[0][3]
h0 = ks[0][4]
v_0 = ks[0][5]

num1 = deque()
num2 = deque()
num3 = deque()
num4 = deque()

x_num1 = []
x_num2 = []
x_num3 = []
x_num4 = []
# h1 = deque()
# h2 = deque()
# sum1 = 0
# sum2 = 0
# image = cv.imread("opencv_frame_4_0.png", 0)
# frame = image[0:1080, 3:1919]
# frame = cv.flip(frame, 1, dst=None)
# show = frame.copy()
# show = cv.rectangle(show, (450, 50), (1550, 1000), (152, 54, 255), 4, 4)
# cv.namedWindow("original_image_1", cv.WINDOW_FREERATIO)
# cv.imshow("original_image_1", show)
# frame = cv.bitwise_not(src=frame)
# # ret2, dst2 = cv.threshold(frame, 0, 255, cv.THRESH_OTSU)
# # cv.imshow("otsu", dst2)
# # for x in range(frame.shape[0]):   # 图片的高
# #     for y in range(frame.shape[1]):   # 图片的宽
# #         if frame[x, y] <= 15:
# #             frame[x, y] = 1
# frame0 = frame[0:500, 943:973]
# plt.hist(frame0.ravel(), 256, [0, 256])
# plt.show()
# ret1, img11 = cv.threshold(frame, 39, 0, cv.THRESH_TOZERO)
# print(ret1)
# # test = img_d_noise(img11)
# # ret2, test = cv.threshold(img11, 30, 255, cv.THRESH_BINARY)
# test = cv.medianBlur(img11, 3)
# cv.imshow("OORR", img11)
# cv.imshow("test", test)
# # lie
# for i in range(0, 1916):
#     for j in range(0, 1080):
#         sum1 = sum1 + test[j, i]
#     h1.append(sum1)
#     sum1 = 0
# # hang
# for i in range(0, 1080):
#     for j in range(0, 1916):
#         sum2 = sum2 + test[i, j]
#     h2.append(sum2)
#     sum2 = 0
# # for i in range(0, 1917):
# #         h2 = h2 + frame[:, i]
# v = np.argmax(h1)  # v = 311
# v0 = np.argmax(h2)  # v0 = 2
# print(v)
# print(v0)
# frameT = np.transpose(img11)
# for i in range(2, 402):
#     dst1 = gray_weight(frame, i, v)
#     dst2 = gray_weight(frameT, i, v0)
#     if np.isnan(dst1) or np.isnan(dst2):
#         continue
#     num1.append(dst1)
#     num2.append(dst2)

img = cv.imread('opencv_frame_4_1.png', 0)
img = img[0:1080, 3:1919]
# cv.imshow("0", img)
frame = cv.medianBlur(img, 3)
frame = cv.GaussianBlur(frame,(3,3),0)
# cv.imshow("i", img)
# frame = img
# gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
frame = cv.bitwise_not(src=img)
# cv.imshow("gray", gray_img)

# 2.canny边缘检测
canny = cv.Canny(frame, 50, 130)
# canny = cv.Pre
prewitt = edgetest.prewitt(frame)
cv.imshow("canny", prewitt)
a, b, c, d = find_zone(prewitt)
t1 = find_t(frame, a, b)
ret, img11 = cv.threshold(frame, t1, 0, cv.THRESH_TOZERO)
print("thresh", t1)

img_t = np.transpose(img11)

# if v0 - 25 < 2 or v - 25 < 2:
#     print("too edge")
for i in range(0, 500):
    dst1 = gray_weight_latest(img11, i, a, b)
    # dst2 = gray_weight2(frameT, i, v0)
    if np.isnan(dst1):
        continue
    x_num1.append(i)
    num1.append(dst1)
for i in range(0, 500):
    # dst1 = gray_weight(src, i, v)
    dst2 = gray_weight_latest(img_t, i, c, d)
    if np.isnan(dst2):
        continue
    x_num2.append(i)
    num2.append(dst2)
# for i in range(v0+50, v0+450):
#     dst3 = gray_weight(frame, i, v)
#     # dst4 = gray_weight2(frameT, i, v0)
#     if np.isnan(dst3):
#         continue
#     num3.append(dst3)
#     # num2.append(dst2)
#
# for i in range(v+10, v+410):
#     # dst1 = gray_weight(frame, i, v)
#     dst4 = gray_weight_wide(frameT, i, v0)
#     dst5 = gray_weight_wide2(frameT, i, v0)
#     if np.isnan(dst4) or np.isnan(dst5):
#         continue
#     dst6 = (dst4 + dst5)/2
#     num4.append(dst6)
door1 = d + 450
door2 = b + 410
if d+450 >= 1079:
    door1 = 1079
for i in range(d + 50, door1):
    dst3 = gray_weight_latest(img11, i, a, b)
    # dst4 = gray_weight2(frameT, i, v0)
    if np.isnan(dst3):
        continue
    x_num3.append(i)
    num3.append(dst3)
if b+410 >= 1916:
    door2 = 1916
for i in range(b + 20, door2):
    # dst1 = gray_weight(frame, i, v)
    dst4 = gray_weight_latest(img_t, i, c, d)
    # dst5 = gray_weight_wide2(frameT, i, v0)
    if np.isnan(dst4):
        continue
    x_num4.append(i)
    num4.append(dst4)

point1 = len(num4)
print(point1)
# for i in range(2, 402):
#     dst1 = fit_arctan(frame, i, v)
#     dst2 = fit_arctan2(frame, i, v0)
#     num1.append(dst1)
#     num2.append(dst2)
space1 = np.linspace(0, 399, 400)
space2 = np.linspace(0, 299, 300)
# space3 = np.linspace(0, 499, 500)
# space4 = np.linspace(0, point1-1, point1)
num2 = num2+num4
num1 = num1+num3
x_num2 = x_num2 + x_num4
x_num1 = x_num1 + x_num3
plt.plot(x_num1, num1, 'o')
# plt.ylim(950, 960)
plt.show()

plt.plot(x_num2, num2, 'o')
# plt.ylim(640, 660)
plt.show()
pointer2 = len(num2)
print(pointer2)
pointer3 = len(num1)
space3 = np.linspace(0, pointer3-1, pointer3)
space4 = np.linspace(0, pointer2-1, pointer2)
a, b, popt1 = fit_line(num2, x_num2)
c, d, popt2 = fit_line(num1, x_num1)
# c = 1/c
# d = -d/c
vi = []
vi_1 = []
num2_1 = []
predict = []
predict_1 = []

vs_sum = 0
vs_sum_1 = 0
for i in range(0, len(num2)):
    # vi.append(a*i+b - num2[i])
    # predict.append(line_fit(i, *popt1))
    predict.append(a*x_num2[i]+b)
    # vs = line_fit(i, *popt1) - num2[i]
    vs = predict[i] - num2[i]
    vs_sum = vs_sum + vs**2
    vi.append(vs)
xi = np.linspace(0, len(num2)-1, len(num2))
y_predict = a*xi + b
theta_0 = (vs_sum/(len(num2)-1))**0.5
print(len(predict))
print(theta_0)
print(a, b)
print(c, d)
mse = mean_squared_error(num2, predict)
rmse = np.sqrt(mse)
plt.figure(0)
plt.plot(x_num2, vi, 'o')
# plt.ylim(650, 660)
plt.show()

# plt.plot(xi, num2, 'o')
# plt.plot(xi, y_predict)
# plt.ylim(640, 660)
# plt.show()
print("MSE :", mse)
print("RMSE :", rmse)

predict_y = []
vs_sum_y = 0
vi_y = []
vi_y_1 = []
num_1_1 = []
predict_y_1 = []
vs_sum_y_1 = 0
for i in range(0, len(num1)):
    # vi.append(a*i+b - num2[i])
    # predict.append(line_fit(i, *popt1))
    predict_y.append(c*x_num1[i]+d)
    # vs = line_fit(i, *popt1) - num2[i]
    vs = predict_y[i] - num1[i]
    vs_sum_y = vs_sum_y + vs**2
    vi_y.append(vs)
xi_y = np.linspace(0, len(num1)-1, len(num1))
y_predict_y = c*xi_y + d
theta_1 = (vs_sum_y/(len(num1)-1))**0.5
print(len(predict_y))
print(theta_1)
mse_y = mean_squared_error(num1, predict_y)
rmse_y = np.sqrt(mse_y)
plt.figure(1)
plt.plot(x_num1, vi_y, 'o')
# plt.ylim(950, 960)
plt.show()

# plt.plot(xi_y, num1, 'o')
# plt.plot(xi_y, y_predict_y)
# plt.ylim(950, 960)
# plt.show()
print("MSE_y :", mse_y)
print("RMSE_y :", rmse_y)

for i in range(0, len(num2)):
    # vi.append(a*i+b - num2[i])
    vs_1 = predict[i] - num2[i]
    num2_1.append(num2[i])
    predict_1.append(predict[i])
    if vs_1 > 1*theta_0 or vs_1 < -1*theta_0:
        num2_1.pop()
        predict_1.pop()
        continue
    vs_sum_1 = vs_sum_1 + vs_1 ** 2
    vi_1.append(vs_1)
theta_v = (vs_sum_1/(len(num2_1)-1))**0.5
print(theta_v)

for i in range(0, len(num1)):
    # vi.append(a*i+b - num2[i])
    vs_1 = predict_y[i] - num1[i]
    num_1_1.append(num1[i])
    predict_y_1.append(predict_y[i])
    if vs_1 > 1*theta_1 or vs_1 < -1*theta_1:
        num_1_1.pop()
        predict_y_1.pop()
        continue
    vs_sum_y_1 = vs_sum_y_1 + vs_1 ** 2
    vi_y_1.append(vs_1)
theta_v = (vs_sum_y_1/(len(num_1_1)-1))**0.5
print(theta_v)


pointer_2 = len(num2_1)
pointer_3 = len(num_1_1)
space3 = np.linspace(0, pointer_3 - 1, pointer_3)
space4 = np.linspace(0, pointer_2 - 1, pointer_2)
a1, b1, popt1 = fit_line(num2_1, space4)
c1, d1, popt2 = fit_line(num_1_1, space3)
xi = np.linspace(0, len(num2_1)-1, len(num2_1))
xi_y_1 = np.linspace(0, len(num_1_1)-1, len(num_1_1))

y_predict_1 = a1*xi + b1
y_predict_2 = c1*xi_y_1 + d1

mse_1 = mean_squared_error(num2_1, predict_1)
rmse_1 = np.sqrt(mse_1)

mse_1_y = mean_squared_error(num_1_1, predict_y_1)
rmse_1_y = np.sqrt(mse_1_y)

plt.figure(2)
plt.plot(xi, num2_1, 'o')
plt.plot(xi, y_predict_1)
plt.ylim(640, 660)
plt.show()
print("MSE :", mse_1)
print("RMSE :", rmse_1)
# x = (d-b)/(a-c)
# y =

plt.figure(3)
plt.plot(xi_y_1, num_1_1, 'o')
plt.plot(xi_y_1, y_predict_2)
plt.ylim(950, 960)
plt.show()
print("MSE :", mse_1_y)
print("RMSE :", rmse_1_y)
x = (c * b + d) / (1 - a * c)
y = (a * d + b) / (1 - a * c)

mean_num2 = np.mean(num2_1)
S = 0
for yt in num2_1:
    S = S + (yt - mean_num2)**2

print(S)
# y = a * x + b
f1 = np.zeros((2, 1))
f1[0][0] = x
f1[1][0] = y
ki = np.zeros((2, 2))
ki[0][0] = k_x
ki[0][1] = k_x*p-k_y*theta
ki[1][0] = k_x*theta
ki[1][1] = k_x*theta*p - k_y

v_h = np.dot(ki, f1)
v_h[0][0] = v_h[0][0] + h0
v_h[1][0] = v_h[1][0] + v_0
print(str(x)+ ', '+ str(y))
print(v_h)
v_res = np.deg2rad(v_h[0][0]/3600)
h_res = np.deg2rad(v_h[1][0]/3600)
print(str(v_res)+ ', '+ str(h_res))
# vi = []
# vs_sum = 0
# for i in range(0, len(num2)):
#     # vi.append(a*i+b - num2[i])
#     vs = line_fit(i, *popt1) - num2[i]
#     vs_sum = vs_sum + vs**2
#     vi.append(vs)
# xi = np.linspace(0, len(num2)-1, len(num2))
# theta = (vs_sum/(len(num2)-1))**0.5
# print(theta)
plt.plot(xi, vi_1)
plt.show()
num1.clear()
num2.clear()
num3.clear()
num4.clear()

# frame[int(y), int(x)] = 255
# frame[int(y)+1, int(x)] = 255
# frame[int(y), int(x)+1] = 255
# frame[int(y)-1, int(x)] = 255
# frame[int(y), int(x)-1] = 255
center = (int(x), int(y))
point_size = 1
# point_color = (0, 0, 255) # BGR
thickness = 8 # 可以为 0 、4、8

# cv.circle(frame, center, 3, point_color, thickness)

cv.namedWindow("point", cv.WINDOW_FREERATIO)
cv.imshow("point", frame)
# cv.imwrite("data/res_p.png", frame)

res_f = cv.cvtColor(frame, cv.COLOR_GRAY2RGB);

ptStart = (0, int(b1))
ptEnd = (1917, int(1079*a1+b1))
point_color = (255, 0, 0) # BGR
thickness = 1
lineType = 4
cv.line(res_f, ptStart, ptEnd, point_color, thickness, lineType)
cv.circle(res_f, center, 5, (0, 0, 255), -1, 0)

ptStart = (int(d1), 0)
ptEnd = (int(1079*c1+d1), 1079)
point_color = (0, 255, 0) # BGR
thickness = 1
lineType = 8
cv.line(res_f, ptStart, ptEnd, point_color, thickness, lineType)

# cv.imwrite("data/res_f.png", res_f)
# res_f = cv.cvtColor(frame, cv.COLOR_GRAY2RGB);
cv.namedWindow("res", cv.WINDOW_FREERATIO)
cv.imshow("res", res_f)

# # cv.imshow("grad_canny",canny)# canny = cv.Canny(image,15,128)
# # cv.namedWindow("grad_canny",cv.WINDOW_FREERATIO)
# # cv.imshow("grad_canny",canny)
# # lines = cv.HoughLinesP(canny, 1, np.pi/180, 100)
# # # cv.namedWindow("grad_canny",cv.WINDOW_FREERATIO)
# # cv.imshow("lines", lines)
# src = image[0:480, 1:640]
#
# a = src[0]
# b = src[:, 0]
# v = np.argmin(a)
# v0 = np.argmin(b)
#
# print(v)
# print(v0)
# print(image.size)
# print(image.shape)
# print(image.shape[0])   # hang
# print(image.shape[1])   # lie
# # cv.imwrite("data/guss_mean_noise.png", final3)
cv.waitKey()
cv.destroyAllWindows()
# array = [0,1,2,3,4,5,6,7,8,9,10]
# print(array[1:4])
