import math

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from scipy.stats import kstest

from cal_mean_std import cal_mean_std as cmd
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from scipy.fft import fft, fftfreq, fftshift


def weighted_avg_and_std(values, weights):
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return average, math.sqrt(variance)


def read_data_from_file(filename):
    x = []
    y = []
    delta_v = []
    f = open(filename, "r")
    lines = f.readlines()
    for line in lines:
        sum = []
        line = line.strip('\n')  # 删除\n
        line = line.strip(',')
        # print(line)
        ls_i = eval(line)
        for element in ls_i:
            sum.append(element)
        x.append(sum[0])
        y.append(sum[1])
        delta_v.append(sum[2])
    return x, y, delta_v


# print(x)
filename_0 = "data/record_data0.txt"
filename_2 = "data/record_data2.txt"
filename_4 = "data/record_data4.txt"
filename_6 = "data/record_data6.txt"
filename_8 = "data/record_data8.txt"
filename_n2 = "data/record_data-2.txt"
filename_n4 = "data/record_data-4.txt"
filename_n6 = "data/record_data-6.txt"

filename_t_s = "data/record3.txt"

file_stream_n8 = "stream_n8.txt"
file_stream_n6 = "stream_n6.txt"
file_stream_n4 = "stream_n4.txt"
file_stream_n2 = "stream_n2.txt"
file_stream_0 = "stream_0.txt"
file_stream_2 = "stream_2.txt"
file_stream_4 = "stream_4.txt"
file_stream_6 = "stream_6.txt"
file_stream_8 = "stream_8.txt"


# x_0, y_0, delta_v_0 = read_data_from_file(filename_0)
# x_2, y_2, delta_v_2 = read_data_from_file(filename_2)
# x_4, y_4, delta_v_4 = read_data_from_file(filename_4)
# x_6, y_6, delta_v_6 = read_data_from_file(filename_6)
# x_8, y_8, delta_v_8 = read_data_from_file(filename_8)
# x_n2, y_n2, delta_v_n2 = read_data_from_file(filename_n2)
# x_n4, y_n4, delta_v_n4 = read_data_from_file(filename_n4)
# x_n6, y_n6, delta_v_n6 = read_data_from_file(filename_n6)
#
# x_ts, y_ts, delta_v_ts = read_data_from_file(filename_t_s)

x_stream_n8, y_stream_n8, delta_stream_n8 = read_data_from_file(file_stream_n8)
x_stream_n6, y_stream_n6, delta_stream_n6 = read_data_from_file(file_stream_n6)
x_stream_n4, y_stream_n4, delta_stream_n4 = read_data_from_file(file_stream_n4)
x_stream_n2, y_stream_n2, delta_stream_n2 = read_data_from_file(file_stream_n2)
x_stream_0, y_stream_0, delta_stream_0 = read_data_from_file(file_stream_0)
x_stream_2, y_stream_2, delta_stream_2 = read_data_from_file(file_stream_2)
x_stream_4, y_stream_4, delta_stream_4 = read_data_from_file(file_stream_4)
x_stream_6, y_stream_6, delta_stream_6 = read_data_from_file(file_stream_6)
x_stream_8, y_stream_8, delta_stream_8 = read_data_from_file(file_stream_8)

x_i_stream = range(0, len(delta_stream_n8))
plt.plot(x_i_stream, delta_stream_n8)
plt.show()

plt.plot(x_i_stream, delta_stream_n6)
plt.show()

plt.plot(x_i_stream, delta_stream_n4)
plt.show()

plt.plot(x_i_stream, delta_stream_n2)
plt.show()

plt.plot(x_i_stream, delta_stream_0)
plt.show()

plt.plot(x_i_stream, delta_stream_2)
plt.show()

plt.plot(x_i_stream, delta_stream_4)
plt.show()

plt.plot(x_i_stream, delta_stream_6)
plt.show()

plt.plot(x_i_stream, delta_stream_8)

plt.show()

# L = len(delta_stream_8)  # 信号长度
# N = np.power(2, np.ceil(np.log2(L)))  # 下一个最近二次幂
# print(N)
# yf = fft(delta_stream_8, int(N))
# xf = fftfreq(int(N), 1 / 60)
#
# plt.plot(xf, np.abs(yf), '.')
# plt.show()

# print(x_0)
# print(y_0)
# print(delta_v_0)
# print(len(delta_v_ts))


# mean_delta_v = np.mean(delta_v_0)
# v_delta_v = cmd(delta_v_0)
# v_y = cmd(y_0)
# print(len(y_0))

# acf_x, interval = acf(x=delta_v_ts, nlags=531, alpha=0.05)
# print('ACF:\n', acf_x)
# print('ACF95%置信区间下界:\n', interval[:, 0] - acf_x)
# print('ACF95%置信区间上界:\n', interval[:, 1] - acf_x)
#
# plot_acf(x=delta_v_ts, lags=531, alpha=0.05)
# plt.show()
#
# plt.psd(x=delta_v_ts, Fs=0.1, NFFT=128, sides='twosided')
# plt.show()


def to_dist(list_of_delta_v):
    f = Counter(list_of_delta_v)
    # print(len(f))
    dist_v = {}
    for ele_v, fre in f.items():
        dist_v[ele_v] = fre
    # print(dist_v)
    return dist_v


# dist_v_0 = to_dist(delta_v_0)
# dist_v_2 = to_dist(delta_v_2)
# dist_v_4 = to_dist(delta_v_4)
# dist_v_6 = to_dist(delta_v_6)
# dist_v_8 = to_dist(delta_v_8)
# dist_v_n2 = to_dist(delta_v_n2)
# dist_v_n4 = to_dist(delta_v_n4)
# dist_v_n6 = to_dist(delta_v_n6)

dist_stream_8 = to_dist(delta_stream_n8)


def cal_per_mean_std(dist, t=1):
    ls_delta_new = []
    ls_delta_weight = []
    weight_sum = 0
    v_sum = 0
    for k, v in dist.items():
        v_sum = v_sum + v
        if v > t:
            ls_delta_new.append(k)
            ls_delta_weight.append(v)
            weight_sum = weight_sum + v
            # print(k, v)
    # print(ls_delta_new)
    # print(ls_delta_weight)
    # print(weight_sum)
    percent = weight_sum / v_sum
    mean, std = weighted_avg_and_std(ls_delta_new, ls_delta_weight)
    mean = round(mean, 2)
    std = round(std, 2)
    percent = round(percent * 100, 2)
    return mean, std, percent


# mean_value = []
# mean_n6, std_n6, percent_n6 = cal_per_mean_std(dist_v_n6, 2)
# mean_value.append(mean_n6)
# mean_n4, std_n4, percent_n4 = cal_per_mean_std(dist_v_n4, 2)
# mean_value.append(mean_n4)
# mean_n2, std_n2, percent_n2 = cal_per_mean_std(dist_v_n2, 2)
# mean_value.append(mean_n2)
# mean0, std0, percent0 = cal_per_mean_std(dist_v_0, 2)
# mean_value.append(mean0)
# mean2, std2, percent2 = cal_per_mean_std(dist_v_2, 2)
# mean_value.append(mean2)
# mean4, std4, percent4 = cal_per_mean_std(dist_v_4, 2)
# mean_value.append(mean4)
# mean6, std6, percent6 = cal_per_mean_std(dist_v_6, 2)
# mean_value.append(mean6)
# mean8, std8, percent8 = cal_per_mean_std(dist_v_8, 2)
# mean_value.append(mean8)

# mean_stream_8, std_stream_8, percent_stream_8 = cal_per_mean_std(dist_stream_8, 2)
# 计算均值
def cal_mean_std(ls_0):
    df = pd.DataFrame(ls_0, columns=['value'])
    u = df['value'].mean()
    print(u)
    # 计算标准差
    std = df['value'].std()
    print(std)
    return u, std
# res = kstest(df, 't', (u, std))
# print(res)


def three_sigma(ls_0):
    u = cal_mean_std(ls_0)[0]
    std = cal_mean_std(ls_0)[1]
    df = pd.DataFrame(ls_0, columns=['value'])
    error = df[np.abs(df['value'] - u) > 2 * std]
    # 剔除异常值，保留正常的数据
    data_c = df[np.abs(df['value'] - u) <= 2 * std]
    return data_c


mean_stream_n8, std_stream_n8 = cal_mean_std(delta_stream_n8)
mean_stream_n6, std_stream_n6 = cal_mean_std(delta_stream_n6)
mean_stream_n4, std_stream_n4 = cal_mean_std(delta_stream_n4)
mean_stream_n2, std_stream_n2 = cal_mean_std(delta_stream_n2)
mean_stream_0, std_stream_0 = cal_mean_std(delta_stream_0)
mean_stream_2, std_stream_2 = cal_mean_std(delta_stream_2)
mean_stream_4, std_stream_4 = cal_mean_std(delta_stream_4)
mean_stream_6, std_stream_6 = cal_mean_std(delta_stream_6)
mean_stream_8, std_stream_8 = cal_mean_std(delta_stream_8)

print(mean_stream_n8, std_stream_n8)
print(mean_stream_n6, std_stream_n6)
print(mean_stream_n4, std_stream_n4)
print(mean_stream_n2, std_stream_n2)
print(mean_stream_0, std_stream_0)
print(mean_stream_2, std_stream_2)
print(mean_stream_4, std_stream_4)
print(mean_stream_6, std_stream_6)

u_n8 = round(mean_stream_n8, 2)

fixed_n8 = three_sigma(delta_stream_n8)
fixed_n6 = three_sigma(delta_stream_n6)
fixed_n4 = three_sigma(delta_stream_n4)
fixed_n2 = three_sigma(delta_stream_n2)
fixed_0 = three_sigma(delta_stream_0)
fixed_2 = three_sigma(delta_stream_2)
fixed_4 = three_sigma(delta_stream_4)
fixed_6 = three_sigma(delta_stream_6)
fixed_8 = three_sigma(delta_stream_8)

fixed_mean_n8, fixed_std_n8 = cal_mean_std(fixed_n8)
fixed_mean_n6, fixed_std_n6 = cal_mean_std(fixed_n6)
fixed_mean_n4, fixed_std_n4 = cal_mean_std(fixed_n4)
fixed_mean_n2, fixed_std_n2 = cal_mean_std(fixed_n2)
fixed_mean_0, fixed_std_0 = cal_mean_std(fixed_0)
fixed_mean_2, fixed_std_2 = cal_mean_std(fixed_2)
fixed_mean_4, fixed_std_4 = cal_mean_std(fixed_4)
fixed_mean_6, fixed_std_6 = cal_mean_std(fixed_6)
fixed_mean_8, fixed_std_8 = cal_mean_std(fixed_8)

print(fixed_mean_n8, fixed_std_n8)
print(fixed_mean_n6, fixed_std_n6)
print(fixed_mean_n6, fixed_std_n6)
print(fixed_mean_n4, fixed_std_n4)
print(fixed_mean_n2, fixed_std_n2)
print(fixed_mean_0, fixed_std_0)
print(fixed_mean_2, fixed_std_2)
print(fixed_mean_4, fixed_std_4)
print(fixed_mean_6, fixed_std_6)
print(fixed_mean_8, fixed_std_8)


# # 输出异常数据
# print(error)
# datac_np = np.array(data_c)
# mean_str_8_fix = round(np.mean(datac_np), 2)
# std_str_8_fix = round(np.std(datac_np),2)
# print("fixed", mean_str_8_fix, std_str_8_fix)
# print(mean_value)
# print(mean_n6, std_n6, percent_n6)
# print(mean_n4, std_n4, percent_n4)
# print(mean_n2, std_n2, percent_n2)
# print(mean0, std0, percent0)
# print(mean2, std2, percent2)
# print(mean4, std4, percent4)
# print(mean6, std6, percent6)
# print(mean8, std8, percent8)
#
# print(mean_stream_8, std_stream_8, percent_stream_8)


# x_mean_value_space = range(-6, 10, 2)
# plt.plot(x_mean_value_space, mean_value, '-o')
# plt.show()


def plot_data(hist_data):
    x1 = range(0, len(hist_data))
    plt.figure(0)
    # plt.plot(x1, delta_v_0)
    # plt.figure(1)
    # plt.plot(x1, x_0)
    # plt.figure(2)
    # plt.plot(x1, y_0)
    # plt.figure(3)
    plt.hist(hist_data, bins=200)
    # plt.figure(4)
    # plt.hist(y_0)
    # plt.hist(d8, bins=120, color='black');plt.hist(d6, bins=120, color='k');plt.hist(d4, bins=120,
    # color='dimgray');plt.hist(d2, bins=120, color='gray');plt.hist(d0, bins=120, color='darkgray');plt.hist(dn2,
    # bins=120, color='silver');plt.hist(dn4, bins=120, color='lightgray');plt.hist(dn6, bins=120, color='gainsboro');

    # plt.plot(x,y)
    plt.show()


# plot_data(delta_stream_n8)
