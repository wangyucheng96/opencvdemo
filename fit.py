import cv2.cv2 as cv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.modeling import models, fitting
import time
from imageprocess import *
from collections import deque

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
POINT = 25
# img0 = cv.imread('data/final_new_5m5_1.png', 0)
img0 = cv.imread('data/opencv_frame_5.png', 0)
img0 = img0[0:480, 1:640]
# cv.imshow("OORR", img0)

a = img0[0]
b = img0[:, 0]
v = np.argmin(a)  # v = 311
v0 = np.argmin(b)  # v0 = 279
imgT = np.transpose(img0)
# cv.imshow("transpose", imgT)
num1 = deque()
num2 = deque()

T1 = 40
T2 = 40


def fit_gaosi(img, i, pos):
    x0 = np.linspace(0, 49, POINT * 2)
    y0 = img[i][pos - 25:pos + 25]
    popt0, pcov0 = curve_fit(guss_fit, x0, y0, maxfev=500000)
    # print("------result______of______guss___fit_______------")
    # print("No." + str(i))
    # print(popt0)
    # print(pcov0)
    s0 = pos - 25 + popt0[2]
    return s0


def fit_gaosi2(img, i, pos):
    x0 = np.linspace(0, 49, POINT * 2)
    y0 = img[i][pos - 25:pos + 25]
    popt0, pcov0 = curve_fit(guss_fit, x0, y0, maxfev=500000)
    # print("------result______of______guss___fit_______------")
    # print("No." + str(i))
    # print(popt0)
    # print(pcov0)
    s0 = pos - 25 + popt0[2]
    # print(s0)
    return s0


def fit_arctan(img, i, pos):
    x0 = np.linspace(0, 49, POINT * 2)
    y0 = img[i][pos - 25:pos + 25]
    # plt.figure(1)
    x1 = x0[0:24]
    y1 = 440 - y0[0:24]
    # plt.plot(x1, y1)
    # plt.text(5, 150, "模拟边缘曲线_1d_1")
    # plt.figure(2)
    x2 = x0[25:49]
    y2 = y0[25:49]
    # plt.plot(x2, y2)
    # plt.text(25, 150, "模拟边缘曲线_1d_2")
    # plt.show()
    popt1, pcov1 = curve_fit(arctanx_fit, x1, y1, maxfev=500000)
    popt2, pcov2 = curve_fit(arctanx_fit, x2, y2, maxfev=500000)
    # print("------result______of______arctan(x)___fit_______------")
    # print(popt1)
    # print(pcov1)
    s1 = pos - 25 - (popt1[2] / popt1[1])
    # print(s1)
    # print("------result______of______arctan(x)___fit------")
    # plt.plot(x1, y1)
    # plt.plot(x1, arctanx_fit(x1, *popt1))
    # plt.title("反正切函数拟合")
    # plt.text(0, 160, '梯度最大的位置: ' + str(s1))

    # print("------result______of______arctan(x)___fit_______------")
    # print(popt2)
    # print(pcov2)
    s2 = pos - 25 - (popt2[2] / popt2[1])
    # print(s2)
    # print("------result______of______arctan(x)___fit------")
    # plt.plot(x2, y2)
    # plt.plot(x2, arctanx_fit(x2, *popt2))
    # plt.title("反正切函数拟合")
    # plt.text(35, 160, '梯度最大的位置: ' + str(s2))
    # plt.show()
    res = (s1 + s2) / 2
    print("反正切拟合，第" + str(i) + "次： " + str(res))
    return res


def fit_arctan2(img, i, pos):
    x0 = np.linspace(0, 49, POINT * 2)
    y0 = img[i][pos - 25:pos + 25]
    # plt.figure(1)
    x1 = x0[0:24]
    y1 = 440 - y0[0:24]
    # plt.plot(x1, y1)
    # plt.text(5, 150, "模拟边缘曲线_1d_1")
    # plt.figure(2)
    x2 = x0[25:49]
    y2 = y0[25:49]
    # plt.plot(x2, y2)
    # plt.text(25, 150, "模拟边缘曲线_1d_2")
    # plt.show()
    popt1, pcov1 = curve_fit(arctanx_fit, x1, y1, maxfev=10000000)
    popt2, pcov2 = curve_fit(arctanx_fit, x2, y2, maxfev=10000000)
    # print("------result______of______arctan(x)___fit_______------")
    # print(popt1)
    # print(pcov1)
    s1 = pos - 25 - (popt1[2] / popt1[1])
    # print(s1)
    # print("------result______of______arctan(x)___fit------")
    # plt.plot(x1, y1)
    # plt.plot(x1, arctanx_fit(x1, *popt1))
    # plt.title("反正切函数拟合")
    # plt.text(0, 160, '梯度最大的位置: ' + str(s1))

    # print("------result______of______arctan(x)___fit_______------")
    # print(popt2)
    # print(pcov2)
    s2 = pos - 25 - (popt2[2] / popt2[1])
    # print(s2)
    # print("------result______of______arctan(x)___fit------")
    # plt.plot(x2, y2)
    # plt.plot(x2, arctanx_fit(x2, *popt2))
    # plt.title("反正切函数拟合")
    # plt.text(35, 160, '梯度最大的位置: ' + str(s2))
    # plt.show()
    res = (s1 + s2) / 2
    print("反正切拟合，第" + str(i) + "次： " + str(res))
    return res


def fit_poly(img, i):
    x0 = np.linspace(0, 49, POINT * 2)
    y0 = img[i][775:825]
    # plt.figure(1)
    x1 = x0[0:24]
    y1 = y0[0:24]
    # plt.plot(x1, y1)
    # plt.text(5, 150, "模拟边缘曲线_1d_1")
    # plt.figure(2)
    x2 = x0[25:49]
    y2 = 220 - y0[25:49]
    # plt.plot(x2, y2)
    # plt.text(25, 150, "模拟边缘曲线_1d_2")
    # plt.show()
    popt3, pcov3 = curve_fit(poly_fit, x1, y1)
    popt4, pcov4 = curve_fit(poly_fit, x2, y2)
    # print("------result______of______poly___fit_______------")
    # print(popt3)
    # print(pcov3)
    s3 = 775 - (popt3[1] / (3 * popt3[0]))
    # print(s3)
    # print("------result______of______poly___fit_______------")
    # print("------result______of______poly___fit_______------")
    # print(popt4)
    # print(pcov4)
    s4 = 775 - (popt4[1] / (3 * popt4[0]))
    # print(s4)
    # print("------result______of______poly___fit_______------")
    #
    # plt.plot(x1, y1)
    # plt.plot(x1, poly_fit(x1, *popt3))
    # plt.text(0, 160, '梯度最大的位置: ' + str(s3))
    # plt.title("三次多项式拟合")
    # plt.plot(x2, y2)
    # plt.plot(x2, poly_fit(x2, *popt4))
    # plt.text(35, 160, '梯度最大的位置: ' + str(s4))
    # plt.title("三次多项式拟合")
    # plt.show()
    res = (s3 + s4) / 2
    print("三次多项式拟合，第" + str(i) + "次： " + str(res))
    return res


def fit_poly2(img, i):
    x0 = np.linspace(0, 49, POINT * 2)
    y0 = img[i][575:625]
    # plt.figure(1)
    x1 = x0[0:24]
    y1 = y0[0:24]
    # plt.plot(x1, y1)
    # plt.text(5, 150, "模拟边缘曲线_1d_1")
    # plt.figure(2)
    x2 = x0[25:49]
    y2 = 220 - y0[25:49]
    # plt.plot(x2, y2)
    # plt.text(25, 150, "模拟边缘曲线_1d_2")
    # plt.show()
    popt3, pcov3 = curve_fit(poly_fit, x1, y1)
    popt4, pcov4 = curve_fit(poly_fit, x2, y2)
    # print("------result______of______poly___fit_______------")
    # print(popt3)
    # print(pcov3)
    s3 = 575 - (popt3[1] / (3 * popt3[0]))
    # print(s3)
    # print("------result______of______poly___fit_______------")
    # print("------result______of______poly___fit_______------")
    # print(popt4)
    # print(pcov4)
    s4 = 575 - (popt4[1] / (3 * popt4[0]))
    # print(s4)
    # print("------result______of______poly___fit_______------")
    #
    # plt.plot(x1, y1)
    # plt.plot(x1, poly_fit(x1, *popt3))
    # plt.text(0, 160, '梯度最大的位置: ' + str(s3))
    # plt.title("三次多项式拟合")
    #
    # plt.plot(x2, y2)
    # plt.plot(x2, poly_fit(x2, *popt4))
    # plt.text(35, 160, '梯度最大的位置: ' + str(s4))
    # plt.title("三次多项式拟合")
    # plt.show()
    res = (s3 + s4) / 2
    print("三次多项式拟合，第" + str(i) + "次： " + str(res))
    return res


def fit_line(src, space):
    # xl = np.linspace(0, 99, point)
    xl = space
    yl = src
    poptl, pcovl = curve_fit(line_fit, xl, yl)
    return poptl[0], poptl[1], poptl


def gray_weight_new(img, i):
    # 边界判断
    src = img[i]
    weight = 0
    t = 0
    for k in range(0, len(src)):
        # if src[k] >= 15:
        # src[k] = 0
        weight = weight + k * src[k]**2
        t = t + src[k]**2
        if t == 0:
            # print("t==0")
            continue
    # print(weight / t)
    res = weight / t
    # print("灰度重心法，第" + str(i) + "次： " + str(res))
    return res


def gray_weight_wide(img, i, pos):
    # 边界判断
    T = T2
    if pos < T:
        T = pos
    src = img[i][pos - T:pos]
    weight = 0
    t = 0
    for k in range(0, len(src)):
        # if src[k] >= 15:
        # src[k] = 0
        weight = weight + k * src[k]**2
        t = t + src[k]**2
        if t == 0:
            # print("t==0")
            continue
    # print(weight / t)
    res = pos - T + weight / t
    # print("灰度重心法，第" + str(i) + "次： " + str(res))
    return res


def gray_weight_wide2(img, i, pos):
    T = T2
    if pos + T > img.shape[1]:
        T = img.shape[1] - pos
    src = img[i][pos:pos + T]
    weight = 0
    t = 0
    for k in range(0, len(src)):
        # if src[k] >= 15:
        # src[k] = 1
        weight = weight + k * src[k]**2
        t = t + src[k]**2
        if t == 0:
            # print("t==0")
            continue
    # print(weight / t)
    res = pos + weight / t
    # print("灰度重心法，第" + str(i) + "次： " + str(res))
    return res


def gray_weight(img, i, pos):
    T = T1
    if pos-T/2<0 :
        T = pos
    if pos + T/2 > img.shape[1]:
        T = img.shape[1] - pos
    src = img[i][pos - int(T/2):pos + int(T/2)]
    # cv.imshow("before", src)
    # dst = cv.resize(src, dsize=(1, 60), interpolation=cv.INTER_LANCZOS4)
    # dst3 = np.transpose(dst)[0]
    # cv.imshow("after", dst3)
    weight = 0
    t = 0
    for k in range(0, len(src)):
        weight = weight + k * src[k]**2
        t = t + src[k]**2
        if t == 0:
            # print("t==0")
            continue
    res = pos - T/2 + weight / t
    # print("灰度重心法，第" + str(i) + "次： " + str(res))
    return res


def gray_weight2(img, i, pos):
    T = T1
    if pos - T/2 < 0:
        T = pos
    if pos + T/2 > img.shape[1]:
        T = img.shape[1] - pos
    src = img[i][pos - int(T/2):pos + int(T/2)]
    weight = 0
    t = 0
    for k in range(0, len(src)):
        weight = weight + k * src[k]**2
        t = t + src[k]**2
        if t == 0:
            # print("t==0")
            continue
    res = pos - T/2 + weight / t
    # print("灰度重心法，第" + str(i) + "次： " + str(res))
    return res


def gray_weight_wide_inter(img, i, pos):
    # 边界判断
    T = T1
    if pos < T:
        T = pos
    src = img[i][pos - T:pos]
    dst = cv.resize(src, dsize=(1, T+20), interpolation=cv.INTER_LANCZOS4)
    rec, dst = cv.threshold(dst, 40, 0, cv.THRESH_TOZERO)

    dst3 = np.transpose(dst)[0]
    weight = 0
    t = 0
    for k in range(0, len(dst3)):
        # if src[k] >= 15:
        # src[k] = 0
        weight = weight + k * dst3[k]**2
        t = t + dst3[k]**2
        if t == 0:
            # print("t==0")
            continue
    # print(weight / t)
    res = pos - T+20 + weight / t
    # print("灰度重心法，第" + str(i) + "次： " + str(res))
    return res


def gray_weight_wide2_inter(img, i, pos):
    T = T1
    if pos + T > img.shape[1]:
        T = img.shape[1] - pos
    src = img[i][pos:pos + T]
    dst = cv.resize(src, dsize=(1, T+20), interpolation=cv.INTER_LANCZOS4)
    rec, dst = cv.threshold(dst, 40, 0, cv.THRESH_TOZERO)

    dst3 = np.transpose(dst)[0]

    weight = 0
    t = 0
    for k in range(0, len(dst3)):
        # if src[k] >= 15:
        # src[k] = 1
        weight = weight + k * dst3[k]**2
        t = t + dst3[k]**2
        if t == 0:
            # print("t==0")
            continue
    # print(weight / t)
    res = pos + weight / t
    # print("灰度重心法，第" + str(i) + "次： " + str(res))
    return res


def gray_weight_inter(img, i, pos):
    T=T1
    if pos-T/2<0 :
        T = pos
    if pos + T/2 > img.shape[1]:
        T = img.shape[1] - pos
    src = img[i][pos - int(T/2):pos + int(T/2)]
    # cv.imshow("before", src)
    dst = cv.resize(src, dsize=(1, T+20), interpolation=cv.INTER_LANCZOS4)
    rec, dst = cv.threshold(dst, 40, 0, cv.THRESH_TOZERO)

    dst3 = np.transpose(dst)[0]
    # cv.imshow("after", dst3)
    weight = 0
    t = 0
    for k in range(0, len(dst3)):
        weight = weight + k * dst3[k]**2
        t = t + dst3[k]**2
        if t == 0:
            # print("t==0")
            continue
    res = pos - (T+20)/2 + weight / t
    # print("灰度重心法，第" + str(i) + "次： " + str(res))
    return res


def gray_weight2_inter(img, i, pos):
    T = T1
    if pos - T/2 < 0:
        T = pos
    if pos + T/2 > img.shape[1]:
        T = img.shape[1] - pos
    src = img[i][pos - int(T/2):pos + int(T/2)]
    dst = cv.resize(src, dsize=(1, (T+20)), interpolation=cv.INTER_LANCZOS4)
    rec, dst = cv.threshold(dst, 40, 0, cv.THRESH_TOZERO)

    dst3 = np.transpose(dst)[0]

    weight = 0
    t = 0
    for k in range(0, len(dst3)):
        weight = weight + k * dst3[k]**2
        t = t + dst3[k]**2
        if t == 0:
            # print("t==0")
            continue
    res = pos - (T+20)/2 + weight / t
    # print("灰度重心法，第" + str(i) + "次： " + str(res))
    return res


def deMerror(ls):
    de_sum = 0
    length = len(ls)-1
    mean = np.mean(ls)
    n = np.array(ls)
    theta = np.std(n, ddof=1)
    for i in range(0, length):
        error = abs(ls[i] - mean)
        if error > 3*theta:
            ls.pop(i)
    return ls


sum1 = 0
sum2 = 0
sum3 = 0
# start1 = time.perf_counter()
# for i in range(0, 100):
#     dst1 = fit_arctan(img0, i)
#     # dst2 = fit_arctan2(imgT, i)
#     # num1.append(dst1)
#     # num2.append(dst2)
#     sum1 = abs(dst1-799.90) + sum1
# end1 = time.perf_counter()
# # print("反正切拟合法运行时间： " + str(end1 - start1))
# # print("反正切拟合法误差： ", sum1)
# start2 = time.perf_counter()
# for i in range(0, 100):
#     dst_1 = fit_poly(img0, i)
#     # dst_2 = fit_poly2(imgT, i)
#     # num1.append(dst_1)
#     # num2.append(dst_2)
#     sum2 = abs(dst_1 - 799.90) + sum2
# end2 = time.perf_counter()
# # print("多项式拟合法运行时间： " + str(end2 - start2))
# # print("多项式拟合法误差： ", sum2)
# start3 = time.perf_counter()
# for i in range(0, 500):
#     dst_3 = gray_weight(img0, i)
#     dst_4 = gray_weight2(imgT, i)
#     num1.append(dst_3)
#     num2.append(dst_4)
#     sum3 = abs(dst_3 - 799.90) + sum3
# end3 = time.perf_counter()
# print("反正切拟合法运行时间： " + str(end1 - start1))
# print("反正切拟合法误差： ", sum1)
# print("多项式拟合法运行时间： " + str(end2 - start2))
# print("多项式拟合法误差： ", sum2)
# print("灰度重心法法运行时间： " + str(end3 - start3))
# print("灰度重心法误差： ", sum3)
# print("-------------------------------------------")
# num.append(dst)
# for i in range(2, 102):
#     dst1 = fit_arctan(img0, i)
#     dst2 = fit_arctan2(imgT, i)
#     num1.append(dst1)
#     num2.append(dst2)

# for i in range(2, 102):
#     dst1 = fit_gaosi(img0, i)
#     dst2 = fit_gaosi2(imgT, i)
#     num1.append(dst1)
#     num2.append(dst2)

# for i in range(2, 402):
#     dst1 = gray_weight(img0, i)
#     dst2 = gray_weight2(imgT, i)
#     num1.append(dst1)
#     num2.append(dst2)
#
# space1 = np.linspace(0, 399, 400)
# space2 = np.linspace(0, 399, 400)
#
# a, b = fit_line(num2, space1)
# c, d = fit_line(num1, space1)
# # c = 1/c
# # d = -d/c
#
# print(a, b)
# print(c, d)
#
# # x = (d-b)/(a-c)
# # y =
#
# x = (c*b + d)/(1-a*c)
# y = a*x + b
#
# print(x, y)
# print(sum1)
# print(sum2)
# print(sum3)

# print(num)
