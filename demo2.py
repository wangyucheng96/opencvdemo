import cv2.cv2 as cv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.modeling import models, fitting

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
POINT = 25

image2 = cv.imread("data/guss_mean_noise.png", 0)
cv.namedWindow("original_window", cv.WINDOW_FREERATIO)
cv.imshow("original_window", image2)


def image_cut(src, max_position):
    res = np.random.randint(10, 99, [500, POINT * 2], np.uint8)
    for k in range(0, 500):
        for j in range(0, POINT * 2):
            res.itemset((k, j), src.item(k + max_position - POINT, j))
    return res


def max_pos(array):
    return np.argmax(array)


def cal_diff(array, point):
    n = np.zeros(point - 1)
    for i in range(0, point - 1):
        n[i] = int(array[i + 1] - array[i])
    return n


max_val_pos = max_pos(image2[0])  # the biggest pixel value of first line

for i in range(max_val_pos - POINT, max_val_pos + POINT):
    print(image2[0][i])

x0 = np.linspace(0, 50, POINT * 2)
y0 = image2[0][175:225]
plt.plot(x0, y0)
plt.text(5, 150, "边缘曲线")
plt.show()


# guss fit
def guss_fit(x, a, sigma, miu):
    return a * np.exp(-((x - miu) ** 2) / (2 * sigma ** 2))


# arctanx fit ｙ＝Ａａｒｃｔａｎ（ωｘ＋φ）＋ｍ
def arctanx_fit(x, k, owmega, fai, m):
    '''''
    ｘ＝－φ/ω
    '''''
    return k * np.arctan(owmega * x + fai) + m


# poly fit
# f(x) = ax^3+bx^2+cx+d
def poly_fit(x, a, b, c, d):
    """''
    x = -b/3a
    ''"""
    return a * x ** 3 + b * x ** 2 + c * x + d


# tanh(x) fit
# f(x) = A*tanh(w*x+f)+l
def tanh(x, k, w, f, l):
    """''
      ｘ＝－φ/ω
      ''"""
    return k * np.tanh(w * x + f) + l


# cosh(x) fit
#  f(x) = A*cosh(w*x+f) + m
def cosh(x, a, w, f, m):
    # x = -w/f
    return a*np.cosh(w * x + f) + m


# sigmoid fit
# f(x) = a/(1+exp(-(x-b)/c)) + d
def sigmoid(x, a, b, c, d):
    # y = a/2 + d
    # x = b
    return a / (1 + np.exp(-(x - b) / c)) + d


# 2 poly fit
# f(x) = a*x**2+b*x+c
def poly_fit_2(x, a, b, c):
    # x = - (2*a)/b
    return a * x**2 + b*x + c


x1 = np.linspace(0, 33, POINT + 8)
y1 = image2[0][709:742]
plt.plot(x1, y1)
plt.show()

x2 = np.linspace(0, 33, POINT + 8)
y2 = image2[0][709:742]
plt.plot(x2, y2)
plt.show()

res = cal_diff(y1, POINT + 8)
print("--------------------diff_res____________________________")
print(res)
print("--------------------diff_res____________________________")

popt0, pcov0 = curve_fit(guss_fit, x0, y0)
popt1, pcov1 = curve_fit(arctanx_fit, x1, y1)
popt2, pcov2 = curve_fit(poly_fit, x2, y2)
popt3, pcov3 = curve_fit(tanh, x2, y2)
popt4, pcov4 = curve_fit(sigmoid, x2, y2)

s = 456
print("------result______of______guss___fit_______------")
print(popt0)
print(pcov0)
s0 = 709 + popt0[2]
print(s0)
print("------result______of______guss___fit_______------")
print("------result______of______arctan(x)___fit_______------")
print(popt1)
print(pcov1)
s1 = 709 - (popt1[2] / popt1[1])
print(s1)
print("------result______of______arctan(x)___fit------")
print("------result______of______poly___fit_______------")
print(popt2)
print(pcov2)
s2 = 709 - (popt2[1] / (3 * popt2[0]))
print(s2)
print("------result______of______poly___fit_______------")
print("------result______of______tanh(x)___fit_______------")
print(popt3)
print(pcov3)
s3 = 709 - (popt3[2] / popt3[1])
print(s3)
print("------result______of______tanh(x)___fit------")
print("------result______of______sigmoid___fit_______------")
print(popt4)
print(pcov4)
s4 = 709 + popt4[1]
print(s4)
print("------result______of______sigmoid___fit------")

plt.plot(x0, y0)
plt.plot(x0, guss_fit(x0, *popt0))
plt.title("高斯曲线拟合")
plt.text(5, 160, 'mean_position: ' + str(s0))
plt.show()

plt.plot(x1, y1)
plt.plot(x1, arctanx_fit(x1, *popt1))
plt.title("反正切函数拟合")
plt.text(5, 160, '梯度最大的位置: ' + str(s1))

plt.show()

plt.plot(x1, y1)
plt.plot(x1, poly_fit(x1, *popt2))
plt.text(5, 160, '梯度最大的位置: ' + str(s2))
plt.title("三次多项式拟合")

plt.show()
plt.plot(x1, y1)
plt.plot(x1, tanh(x1, *popt3))
plt.title("双曲正切函数拟合")
plt.text(5, 160, '梯度最大的位置: ' + str(s3))
plt.show()
plt.plot(x1, y1)
plt.plot(x1, sigmoid(x1, *popt4))
plt.title("sigmoid函数拟合")
plt.text(5, 160, '梯度最大的位置: ' + str(s4))
plt.show()

print(image2.item(0, 709))
print(image2.item(0, 710))
print(image2.item(0, 711))
print(image2.item(0, 712))

a = image2[0]
print(a)
print("def :", max_pos(a))
print(image2.shape[0])  # hang
print(image2.shape[1])  # lie

cut = image_cut(image2, max_pos(a))
# cv.namedWindow("cut_res", cv.WINDOW_FREERATIO)
# cv.imshow("cut_res", cut)
for i in range(0, image2.shape[0]):
    print(max_pos(image2[i]))

cv.waitKey()
cv.destroyAllWindows()
