import cv2.cv2 as cv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.modeling import models, fitting
from imageprocess import *
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
POINT = 25

img0 = cv.imread('data/final_new_5m5_1.png', 0)

# cv.imshow("original_image", img0)
#
# ret, dst = cv.threshold(img0, 50, 255, cv.THRESH_TOZERO)
# print(ret)
# cv.imshow("the", dst)

x0 = np.linspace(0, 49, POINT * 2)
y0 = img0[0][775:825]
plt.figure(0)
plt.plot(x0, y0)
plt.text(5, 150, "模拟边缘曲线")
plt.figure(1)
x1 = x0[0:24]
y1 = y0[0:24]
plt.plot(x1, y1)
plt.text(5, 150, "模拟边缘曲线_1d_1")
plt.figure(2)
x2 = x0[25:49]
y2 = 220 - y0[25:49]
plt.plot(x2, y2)
plt.text(25, 150, "模拟边缘曲线_1d_2")
plt.show()

popt1, pcov1 = curve_fit(arctanx_fit, x1, y1)
popt2, pcov2 = curve_fit(arctanx_fit, x2, y2)
popt3, pcov3 = curve_fit(poly_fit, x1, y1)
popt4, pcov4 = curve_fit(poly_fit, x2, y2)

print("------result______of______arctan(x)___fit_______------")
print(popt1)
print(pcov1)
s1 = 775 - (popt1[2] / popt1[1]) + 0.2
print(s1)
print("------result______of______arctan(x)___fit------")
plt.plot(x1, y1)
plt.plot(x1, arctanx_fit(x1, *popt1))
plt.title("反正切函数拟合")
plt.text(0, 160, '梯度最大的位置: ' + str(s1))

print("------result______of______arctan(x)___fit_______------")
print(popt2)
print(pcov2)
s2 = 775 - (popt2[2] / popt2[1]) + 0.2
print(s2)
print("------result______of______arctan(x)___fit------")
plt.plot(x2, y2)
plt.plot(x2, arctanx_fit(x2, *popt2))
plt.title("反正切函数拟合")
plt.text(35, 160, '梯度最大的位置: ' + str(s2))
plt.show()
print((s1+s2)/2)

print("------result______of______poly___fit_______------")
print(popt3)
print(pcov3)
s3 = 775 - (popt3[1] / (3 * popt3[0]))
print(s3)
print("------result______of______poly___fit_______------")
print("------result______of______poly___fit_______------")
print(popt4)
print(pcov4)
s4 = 775 - (popt4[1] / (3 * popt4[0]))
print(s4)
print("------result______of______poly___fit_______------")

plt.plot(x1, y1)
plt.plot(x1, poly_fit(x1, *popt3))
plt.text(0, 160, '梯度最大的位置: ' + str(s3))
plt.title("三次多项式拟合")

plt.plot(x2, y2)
plt.plot(x2, poly_fit(x2, *popt4))
plt.text(35, 160, '梯度最大的位置: ' + str(s4))
plt.title("三次多项式拟合")
plt.show()
print((s3+s4)/2)

cv.waitKey()
cv.destroyAllWindows()

