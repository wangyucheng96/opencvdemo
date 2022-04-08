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


# 2 poly fit
# f(x) = a*x**2+b*x+c
def poly_fit_2(x, a, b, c):
    # x = - b/(2*a)
    return a * x**2 + b*x + c


# tanh(x) fit
# f(x) = A*tanh(w*x+f)+l
def tanh(x, k, w, f, l):
    """''
      ｘ＝－φ/ω
      ''"""
    return k * np.tanh(w * x + f) + l


# cosh(x) fit
#  f(x) = A*cosh(w*x+f) + m
def cosh_fit(x, a, w, f, m):
    # x = -w/f
    return a * np.cosh(w * x + f) + m


# sigmoid fit
# f(x) = a/(1+exp(-(x-b)/c)) + d
def sigmoid(x, a, b, c, d):
    # y = a/2 + d
    # x = b
    return a / (1 + np.exp(-(x - b) / c)) + d


# line fit
# f(x) = a * x + b
def line_fit(x, a, b):

    return a * x + b


