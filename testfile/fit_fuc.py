import math

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from sklearn.metrics import mean_squared_error


def cal_mean_std(ls_0):
    df = pd.DataFrame(ls_0, columns=['value'])
    u = df['value'].mean()
    # print(u)
    # 计算标准差
    std = df['value'].std(ddof=1)
    # print(std)
    return u, std


def three_sigma_s(predict, x, ls_0, t=3):
    xi = []
    ls_1 = []
    predict_1 = []
    mse = mean_squared_error(ls_0, predict)
    rmse = np.sqrt(mse)
    for i in range(0, len(ls_0)):
        # vi.append(a*i+b - num2[i])
        vs_1 = np.abs(predict[i] - ls_0[i])
        ls_1.append(ls_0[i])
        xi.append(x[i])
        predict_1.append(predict[i])
        if vs_1 > t*rmse:
            ls_1.pop()
            predict_1.pop()
            xi.pop()
            continue
        # vs_sum_1 = vs_sum_1 + vs_1 ** 2
        # vi_1.append(vs_1)
    return xi, ls_1
    # u = cal_mean_std(ls_0)[0]
    # std = cal_mean_std(ls_0)[1]
    # df = pd.DataFrame(ls_0, columns=['value'])
    # error = df[np.abs(df['value'] - u) > 2 * std]
    # # 剔除异常值，保留正常的数据
    # data_c = df[np.abs(df['value'] - u) < 3 * std]
    # for e in data_c:
    #     x.append(data_c.index(e))
    # return x, data_c

