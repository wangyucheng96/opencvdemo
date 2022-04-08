import cv2.cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from imageprocess import guss_fit
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
POINT = 25
img0 = cv.imread('data/final_new_5m5_1.png', 0)

x0 = np.linspace(0, 49, POINT * 2)
for i in range(0, 1199):
    y0 = img0[i][775:825]
    popt0, pcov0 = curve_fit(guss_fit, x0, y0)
    print("------result______of______guss___fit_______------")
    print("No."+str(i))
    print(popt0)
    print(pcov0)
    s0 = 775 + popt0[2] + 0.2
    print(s0)