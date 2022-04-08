import math

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from cal_mean_std import cal_mean_std as cmd
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf


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

x_0, y_0, delta_v_0 = read_data_from_file(filename_0)
x_2, y_2, delta_v_2 = read_data_from_file(filename_2)
x_4, y_4, delta_v_4 = read_data_from_file(filename_4)
x_6, y_6, delta_v_6 = read_data_from_file(filename_6)
x_8, y_8, delta_v_8 = read_data_from_file(filename_8)
x_n2, y_n2, delta_v_n2 = read_data_from_file(filename_n2)
x_n4, y_n4, delta_v_n4 = read_data_from_file(filename_n4)
x_n6, y_n6, delta_v_n6 = read_data_from_file(filename_n6)

x_ts, y_ts, delta_v_ts = read_data_from_file(filename_t_s)

# print(x_0)
# print(y_0)
# print(delta_v_0)
print(len(delta_v_ts))
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


dist_v_0 = to_dist(delta_v_0)
dist_v_2 = to_dist(delta_v_2)
dist_v_4 = to_dist(delta_v_4)
dist_v_6 = to_dist(delta_v_6)
dist_v_8 = to_dist(delta_v_8)
dist_v_n2 = to_dist(delta_v_n2)
dist_v_n4 = to_dist(delta_v_n4)
dist_v_n6 = to_dist(delta_v_n6)


def cal_per_mean_std(dist, t=1):
    ls_delta_new = []
    ls_delta_weight = []
    weight_sum = 0
    for k, v in dist.items():
        if v > t:
            ls_delta_new.append(k)
            ls_delta_weight.append(v)
            weight_sum = weight_sum + v
            # print(k, v)
    # print(ls_delta_new)
    # print(ls_delta_weight)
    # print(weight_sum)
    percent = weight_sum / len(delta_v_0)
    mean, std = weighted_avg_and_std(ls_delta_new, ls_delta_weight)
    mean = round(mean, 2)
    std = round(std, 2)
    percent = round(percent*100, 2)
    return mean, std, percent


mean_value = []
mean_n6, std_n6, percent_n6 = cal_per_mean_std(dist_v_n6, 2)
mean_value.append(mean_n6)
mean_n4, std_n4, percent_n4 = cal_per_mean_std(dist_v_n4, 2)
mean_value.append(mean_n4)
mean_n2, std_n2, percent_n2 = cal_per_mean_std(dist_v_n2, 2)
mean_value.append(mean_n2)
mean0, std0, percent0 = cal_per_mean_std(dist_v_0, 2)
mean_value.append(mean0)
mean2, std2, percent2 = cal_per_mean_std(dist_v_2, 2)
mean_value.append(mean2)
mean4, std4, percent4 = cal_per_mean_std(dist_v_4, 2)
mean_value.append(mean4)
mean6, std6, percent6 = cal_per_mean_std(dist_v_6, 2)
mean_value.append(mean6)
mean8, std8, percent8 = cal_per_mean_std(dist_v_8, 2)
mean_value.append(mean8)

# print(mean_value)
print(mean_n6, std_n6, percent_n6)
print(mean_n4, std_n4, percent_n4)
print(mean_n2, std_n2, percent_n2)
print(mean0, std0, percent0)
print(mean2, std2, percent2)
print(mean4, std4, percent4)
print(mean6, std6, percent6)
print(mean8, std8, percent8)

x_mean_value_space = range(-6, 10, 2)
plt.plot(x_mean_value_space, mean_value, '-o')
plt.show()


def plot_data(delta_v):
    x1 = range(0, len(delta_v))
    plt.figure(0)
    # plt.plot(x1, delta_v_0)
    # plt.figure(1)
    # plt.plot(x1, x_0)
    # plt.figure(2)
    # plt.plot(x1, y_0)
    # plt.figure(3)
    plt.hist(delta_v, bins=len(delta_v))
    # plt.figure(4)
    # plt.hist(y_0)
    # plt.hist(d8, bins=120, color='black');plt.hist(d6, bins=120, color='k');plt.hist(d4, bins=120,
    # color='dimgray');plt.hist(d2, bins=120, color='gray');plt.hist(d0, bins=120, color='darkgray');plt.hist(dn2,
    # bins=120, color='silver');plt.hist(dn4, bins=120, color='lightgray');plt.hist(dn6, bins=120, color='gainsboro');

    # plt.plot(x,y)
    plt.show()


# plot_data(delta_v_0)

