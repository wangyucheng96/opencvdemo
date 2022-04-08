import numpy as np

b1 = [0.08, 0.063, 0.055, 0.059, 0.060, 0.066, 0.064, 0.065]
b2 = [0.05, 0.025, 0.017, 0.063, 0.020, 0.067, 0.079, 0.056]

f1 = [0.120, 0.09, 0.065, 0.055, 0.049, 0.052, 0.046, 0.052]
f2 = [0.0, 0.025, 0.10, 0.013, 0.010, 0.092, 0.071, 0.044]


def cal_mean_std(ls):
    mean = np.mean(ls)
    ls_n = np.array(ls)
    std = np.std(ls_n, ddof=1)
    return [mean, std]


a = cal_mean_std(b1)
b = cal_mean_std(b2)
c = cal_mean_std(f1)
d = cal_mean_std(f2)
m1 = round(a[0], 2)
print(a,b,c,d)
print(m1)
