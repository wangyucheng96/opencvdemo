import cv2.cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from imageprocess import guss_fit, poly_fit_2, cosh_fit
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

POINT = 25
img0 = cv.imread('data/final_new_5m5_1.png', 0)
img1 = cv.imread('data/image1.bmp', 0)
img2 = cv.imread('data/test3.png', 0)

# img2 = cv.imread('data/guss_mean_noise.png', 0)
# img3 = cv.imread('data/mean_noise.png', 0)

x0 = np.linspace(0, 49, POINT * 2)
y0 = img1[0][709:759]
plt.figure(1)
plt.plot(x0, y0)
plt.text(5, 150, "实际边缘曲线")
# plt.show()

# x0 = np.linspace(0, 50, POINT * 2)
# y0 = img2[0][175:225]
# plt.figure(2)
# plt.plot(x0, y0)
# plt.text(5, 150, "边缘曲线")
# plt.show()

# x0 = np.linspace(0, 50, POINT * 2)
# y0 = img3[0][175:225]
# plt.figure(3)
# plt.plot(x0, y0)
# plt.text(5, 150, "边缘曲线")
# plt.show()


x0 = np.linspace(0, 49, POINT * 2)
y0 = img0[0][775:825]
plt.figure(2)
plt.plot(x0, y0)
plt.text(5, 150, "模拟边缘曲线")

x1 = np.linspace(0, 249, POINT * 10)
y1 = img2[0][3875:4125]
plt.figure(3)
plt.plot(x1, y1)
plt.text(5, 150, "采样前模拟边缘曲线")
plt.show()

popt0, pcov0 = curve_fit(guss_fit, x0, y0)
print("------result______of______guss___fit_______------")
print(popt0)
print(pcov0)
s0 = 775 + popt0[2] + 0.2
print(s0)
print("------result______of______guss___fit_______------")
plt.plot(x0, y0)
plt.plot(x0, guss_fit(x0, *popt0))
plt.title("高斯曲线拟合")
plt.text(5, 160, 'mean_position: ' + str(s0))
plt.show()

popt3, pcov3 = curve_fit(cosh_fit, x0, y0, maxfev=500000)
print("------result______of______cosh__fit_______------")
print(popt3)
print(pcov3)
s3 = 775 - (popt3[2] / popt3[1]) + 0.2
print(s3)
print("------result______of______cosh___fit_______------")
plt.plot(x0, y0)
plt.plot(x0, cosh_fit(x0, *popt3))
plt.title("cosh曲线拟合")
plt.text(5, 160, 'mean_position: ' + str(s3))
plt.show()

popt2, pcov2 = curve_fit(poly_fit_2, x0, y0)
print("------result______of______poly2___fit_______------")
print(popt2)
print(pcov2)
s2 = 775 - (popt2[1] / (2 * popt2[0])) + 0.2
print(s2)
print("------result______of______guss___fit_______------")
plt.plot(x0, y0)
plt.plot(x0, poly_fit_2(x0, *popt2))
plt.title("二次曲线拟合")
plt.text(5, 160, 'mean_position: ' + str(s0))
plt.show()

popt1, pcov1 = curve_fit(guss_fit, x1, y1)
print("------result______of______guss___fit_______------")
print(popt1)
print(pcov1)
s1 = 3875 + popt1[2]
print(s1)
print("------result______of______guss___fit_______------")
plt.plot(x1, y1)
plt.plot(x1, guss_fit(x1, *popt1))
plt.title("采样前高斯曲线拟合")
plt.text(5, 160, 'mean_position: ' + str(s1))
plt.show()
