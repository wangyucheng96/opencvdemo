import numpy as np
import math

a = np.load('data/theo_angs.npy')
b = np.load('data/coording.npy')
a = a.tolist()
b = b.tolist()
print(a)
# print(b)
for i in range(0, len(a)):
    if a[i] > math.pi:
        a[i] = a[i] - 2 * math.pi
    elif a[i] < math.pi:
        a[i] = a[i] + 2 * math.pi
print(a)
for i in range(0, len(a)):
    a[i] = np.rad2deg(a[i]) * 3600
print(a)
h_v = np.array(a)
print(h_v.shape)
h_v_1 = h_v.reshape(-1, 1)
print(h_v_1.shape)
print(h_v_1)
x_y = np.array(b)
x_y_i = x_y.reshape(-1, 2)
print(x_y_i)
size = x_y_i.shape[0]
print(size)
G = np.zeros((2 * size, 6))
G1 = np.zeros((2 * size, 6))
for i in range(0, int(G1.shape[0] / 2)):
    G1[2 * i][0] = x_y_i[i][0]
    G1[2 * i][1] = x_y_i[i][1]
    G1[2 * i][2] = 0
    G1[2 * i][3] = 0
    G1[2 * i][4] = 1
    G1[2 * i][5] = 0
    G1[2 * i + 1][0] = 0
    G1[2 * i + 1][1] = 0
    G1[2 * i + 1][2] = x_y_i[i][0]
    G1[2 * i + 1][3] = x_y_i[i][1]
    G1[2 * i + 1][4] = 0
    G1[2 * i + 1][5] = 1
print(G1)
g_t = np.transpose(G1)
g_s = np.dot(g_t, G1)
g_inv = np.matrix(g_s)
g_inv_1 = g_inv.I
res = g_inv_1 * g_t * h_v_1
res1 = np.dot(g_inv_1, g_t)
res2 = np.dot(res1, h_v_1)
print(res)
print(res2)
np.save('data/res2.npy', res2)


