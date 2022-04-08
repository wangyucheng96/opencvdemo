import matplotlib.pyplot as plt
import numpy as np
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

down = [118.39,	118.58,	118.64,	118.72,	118.86,	118.99,	119.18,	119.29,	119.43]
up = [118.60, 118.36, 118.24,	118.21,	118.16,	118.11,	117.98,	117.92,	117.77]

b1 = [0.08, 0.063, 0.055, 0.059, 0.060, 0.066, 0.064, 0.065]
b2 = [0.05, 0.025, 0.017, 0.063, 0.020, 0.067, 0.079, 0.056]

f1 = [0.120, 0.09, 0.065, 0.055, 0.049, 0.052, 0.046, 0.052]
f2 = [0.0, 0.025, 0.10, 0.013, 0.010, 0.092, 0.071, 0.044]

mean1 = 0.0640
mean2 = 0.0471

x_b_1 = [2, 4, 6, 8, 10, 12, 14, 16]
y_b_1 = [0.0640, 0.0640, 0.0640, 0.0640, 0.0640, 0.0640, 0.0640, 0.0640]
y_b_2 = [0.0471, 0.0471, 0.0471, 0.0471, 0.0471, 0.0471, 0.0471, 0.0471]

mean3 = 0.0661
mean4 = 0.0444
x_f_1 = [2, 4, 6, 8, 10, 12, 14, 16]
y_f_1 = [0.0661, 0.0661, 0.0661, 0.0661, 0.0661, 0.0661, 0.0661, 0.0661]
y_f_2 = [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444]
# anno2 = ['0', '+2', '+4', '+6', '+8', '+10', '+12', '+14', '+16']
# anno1 = ['0', '-2', '-4', '-6', '-8', '-10', '-12', '-14', '-16']
# x = range(0, 18, 2)
# x1 = range(0, 18, 2)
# plt.xlabel("range, 单位：(min)")#x轴上的名字
# plt.ylabel("delta_v, 单位：(s)")#y轴上的名字
# l1, = plt.plot(x, y1, label='linear line')
# l2, = plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--', label='square line')

# 简单的设置legend(设置位置)
# 位置在右上角
# for i in range(0, 9):
#     plt.annotate(anno1[i], xy=(x[i], down[i]), xytext=(x[i], down[i]))
#     plt.annotate(anno2[i], xy=(x[i], up[i]), xytext=(x[i], up[i]))
#
# # plt.text(35, 160, '天宝')
# plt.title("天宝")
# plt.plot(x, down, '-o', color = 'blue', label='顺时针, backward')
# plt.plot(x, up, '-o', color = 'red', label='逆时针, forward')
# plt.legend(loc='upper left')
# plt.xticks(x1)
# plt.show()

plt.plot(x_b_1, b1, '-o', color = 'blue', label='本文算法')
plt.plot(x_b_1, b2, '-o', color = 'red', label='人眼法')
plt.plot(x_b_1, y_b_1, '--', color = 'blue', label='本文算法均值')
plt.plot(x_b_1, y_b_2, '--', color = 'red', label='人眼法均值')

plt.legend(loc='upper right')
plt.ylim((0.00, 0.15))
plt.title("补偿误差测量（后倾）")
plt.xlabel("在x = i 时测量，单位（min）")#x轴上的名字
plt.ylabel("补偿误差：单位(s/1 min)")
plt.show()


