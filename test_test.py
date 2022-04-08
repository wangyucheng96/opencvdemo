import numpy as np

kx_o = np.load('k.npy')
kx_fix = np.load('theta_fix.npy')

theta_cal = kx_o[0][2]
theta_cal_fix = kx_fix[0][0]

print(theta_cal, theta_cal_fix)
