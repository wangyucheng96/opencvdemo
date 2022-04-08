import cv2.cv2 as cv
import numpy as np
from collections import deque
from fit import *

num1 = deque()
num2 = deque()
num3 = deque()
num4 = deque()
h1 = deque()
h2 = deque()
sum1 = 0
sum2 = 0

cx0 = 800.00
cy0 = 600.00

cx1 = 799.80
cy1 = 599.80

cx2 = 799.60
cy2 = 599.60

cx3 = 799.40
cy3 = 599.40

cx4 = 799.20
cy4 = 599.20

thre = []


image = cv.imread("data/final20.png", 0)
rec, image = cv.threshold(image, 40, 0, cv.THRESH_TOZERO)

h = image.shape[0]  # 1200
w = image.shape[1]  # 1600
# print(h, w)
# lie
for i in range(0, w):
    for j in range(0, h):
        sum1 = sum1 + image[j, i]
    h1.append(sum1)
    sum1 = 0
# hang
for i in range(0, h):
    for j in range(0, w):
        sum2 = sum2 + image[i, j]
    h2.append(sum2)
    sum2 = 0
# for i in range(0, 1917):
#         h2 = h2 + frame[:, i]
v = np.argmax(h1)  # v = 311
v0 = np.argmax(h2)  # v0 = 2
print(v)  # 800
print(v0)  # 600

# test_img = image[0:v0-50, v-20:v+20]
# test_img2 = image[v0+50:h, v-20:v+20]
#
# h_test = test_img.shape[0]  # 1200
# w_test = test_img.shape[1]  # 1600
# h_test2 = test_img.shape[0]  # 1200
# w_test2 = test_img.shape[1]  # 1600
# test_dst = cv.resize(test_img, dsize=(w_test, h_test*10), interpolation=cv.INTER_CUBIC)
# test_dst2 = cv.resize(test_img2, dsize=(w_test2, h_test2*10), interpolation=cv.INTER_CUBIC)


frameT = np.transpose(image)

# t1 = []
# t2 = []
# for i in range(0, h_test*10):
#     dst0 = gray_weight_new(test_dst, i)
#     if np.isnan(dst0):
#         continue
#     t1.append(dst0)

for i in range(2, v0 - 550):
    dst1 = gray_weight(image, i, v)
    # dst2 = gray_weight2(frameT, i, v0)
    if np.isnan(dst1):
        continue
    num1.append(dst1)
for i in range(2, v - 750):
    # dst1 = gray_weight(src, i, v)
    dst2 = gray_weight2(frameT, i, v0)
    if np.isnan(dst2):
        continue
    num2.append(dst2)
door3 = v0 + 50
door4 = v + 50
if v0 + 390 >= h-1:
    door3 = h-1
for i in range(v0 + 10, door3):
    dst3 = gray_weight(image, i, v)
    # dst4 = gray_weight2(frameT, i, v0)
    if np.isnan(dst3):
        continue
    num3.append(dst3)
    # num4.append(dst4)
if v + 410 >= w-1:
    door4 = w-1
for i in range(v + 20, door4):
    # dst3 = gray_weight(src, i, v)
    dst4 = gray_weight2(frameT, i, v0)
    if np.isnan(dst4):
        continue
    # num3.append(dst3)
    num4.append(dst4)
num2 = num2+num4
num1 = num1+num3
pointer2 = len(num2)
pointer3 = len(num1)
space3 = np.linspace(0, pointer3-1, pointer3)
space4 = np.linspace(0, pointer2-1, pointer2)
a, b, popt1 = fit_line(num2, space4)
c, d, popt2 = fit_line(num1, space3)
# c = 1/c
# d = -d/c
vi = []
vi_1 = []
num2_1 = []
vs_sum = 0
vs_sum_1 = 0
for i in range(0, len(num2)):
    # vi.append(a*i+b - num2[i])
    vs = line_fit(i, *popt1) - num2[i]
    vs_sum = vs_sum + vs**2
    vi.append(vs)
xi = np.linspace(0, len(num2)-1, len(num2))
theta = (vs_sum/(len(num2)-1))**0.5
print(theta)
print(a, b)
print(c, d)
for i in range(0, len(num2)):
    # vi.append(a*i+b - num2[i])
    vs_1 = line_fit(i, *popt1) - num2[i]
    num2_1.append(num2[i])
    if vs_1 > 3*theta or vs_1 < -3*theta:
        num2_1.pop()
        continue
    vs_sum_1 = vs_sum_1 + vs_1 ** 2
    vi_1.append(vs_1)
theta_v = (vs_sum_1/(len(num2_1)-1))**0.5
print(theta_v)
pointer_2 = len(num2_1)
pointer_3 = len(num1)
space3 = np.linspace(0, pointer_3 - 1, pointer_3)
space4 = np.linspace(0, pointer_2 - 1, pointer_2)
a1, b1, popt1 = fit_line(num2_1, space4)
c1, d1, popt2 = fit_line(num1, space3)
xi = np.linspace(0, len(num2_1)-1, len(num2_1))
# x = (d-b)/(a-c)
# y =

x = (c1 * b1 + d1) / (1 - a1 * c1)
y = (a1 * d1 + b1) / (1 - a1 * c1)
v1 = abs(cx0-x)
v2 = abs(cy0-y)
delta_v = (v1**2+v2**2)**0.5
res = round(delta_v, 3)
print(res)
print(str(x)+ ', '+ str(y))
# print(v1, v2)
num1.clear()
num2.clear()
num3.clear()
num4.clear()
sum1 = 0
sum2 = 0


# for i in range(2, v0 - 25):
#     dst1 = gray_weight(image, i, v)
#     # dst2 = gray_weight2(frameT, i, v0)
#     if np.isnan(dst1):
#         continue
#     num1.append(dst1)
# for i in range(2, v - 25):
#     # dst1 = gray_weight(src, i, v)
#     dst2 = gray_weight2(frameT, i, v0)
#     if np.isnan(dst2):
#         continue
#     num2.append(dst2)
# door3 = v0 + 390
# door4 = v + 410
# if v0 + 390 >= h-1:
#     door3 = h-1
# for i in range(v0 + 10, door3):
#     dst3 = gray_weight(image, i, v)
#     # dst4 = gray_weight2(frameT, i, v0)
#     if np.isnan(dst3):
#         continue
#     num3.append(dst3)
#     # num4.append(dst4)
# if v + 410 >= w-1:
#     door4 = w-1
# for i in range(v + 20, door4):
#     # dst3 = gray_weight(src, i, v)
#     dst4 = gray_weight2(frameT, i, v0)
#     if np.isnan(dst4):
#         continue
#     # num3.append(dst3)
#     num4.append(dst4)
# num2 = num2+num4
# num1 = num1+num3
# pointer2 = len(num2)
# pointer3 = len(num1)
# space3 = np.linspace(0, pointer3-1, pointer3)
# space4 = np.linspace(0, pointer2-1, pointer2)
# a, b, popt1 = fit_line(num2, space4)
# c, d, popt2 = fit_line(num1, space3)
# # c = 1/c
# # d = -d/c
# vi = []
# vi_1 = []
# num2_1 = []
# vs_sum = 0
# vs_sum_1 = 0
# for i in range(0, len(num2)):
#     # vi.append(a*i+b - num2[i])
#     vs = line_fit(i, *popt1) - num2[i]
#     vs_sum = vs_sum + vs**2
#     vi.append(vs)
# xi = np.linspace(0, len(num2)-1, len(num2))
# theta = (vs_sum/(len(num2)-1))**0.5
# print(theta)
# print(a, b)
# print(c, d)
# for i in range(0, len(num2)):
#     # vi.append(a*i+b - num2[i])
#     vs_1 = line_fit(i, *popt1) - num2[i]
#     num2_1.append(num2[i])
#     if vs_1 > 3*theta or vs_1 < -3*theta:
#         num2_1.pop()
#         continue
#     vs_sum_1 = vs_sum_1 + vs_1 ** 2
#     vi_1.append(vs_1)
# theta_v = (vs_sum_1/(len(num2_1)-1))**0.5
# print(theta_v)
# pointer_2 = len(num2_1)
# pointer_3 = len(num1)
# space3 = np.linspace(0, pointer_3 - 1, pointer_3)
# space4 = np.linspace(0, pointer_2 - 1, pointer_2)
# a1, b1, popt1 = fit_line(num2_1, space4)
# c1, d1, popt2 = fit_line(num1, space3)
# xi = np.linspace(0, len(num2_1)-1, len(num2_1))
# # x = (d-b)/(a-c)
# # y =
#
# x = (c1 * b1 + d1) / (1 - a1 * c1)
# y = (a1 * d1 + b1) / (1 - a1 * c1)
# v1 = cx0-x
# v2 = cy0-y
# print(str(x)+ ', '+ str(y))
# print(v1, v2)
# v1 = abs(cx0-x)
# v2 = abs(cy0-y)
# delta_v = (v1**2+v2**2)**0.5
# res = round(delta_v, 2)
# print(res)

