"""
1D FFT
https://www.delftstack.com/zh-tw/howto/python/fft-example-in-python/
"""


import numpy as np

"""
P_r = np.array([0.192, 0.249, 0.207, 0.16, 0.08, 0.059, 0.029, 0.019])

len_P_r = P_r.shape[0]
print("len_P_r : {}".format(len_P_r))

S_k = np.zeros((1, len_P_r))
for i in range(len_P_r):
    if(i == 0):
        S_k[0][i] = P_r[i]
    
    else :
        S_k[0][i] = P_r[i] + S_k[0][i-1]


S_k = (len_P_r-1) * S_k
print("S_k : {}".format(S_k))


Z_q = np.array([0, 0, 0, 0.15, 0.2, 0.299, 0.2, 0.15])

len_Z_q = Z_q.shape[0]
print("len_Z_q : {}".format(len_Z_q))

G_z = np.zeros((1, len_Z_q))
for i in range(len_Z_q):
    if(i == 0):
        G_z[0][i] = Z_q[i]
    
    else :
        G_z[0][i] = Z_q[i] + G_z[0][i-1]


G_z = (len_Z_q-1) * G_z
print("G_z : {}".format(G_z))
"""

f_x = np.array([1, 2, 4, 3])
F_u = np.fft.fft(f_x)
print("F_u : {}".format(F_u))

f_x = np.array([1, 2, 4, 3, 1, 2, 4, 3])
F_u = np.fft.fft(f_x)
print("F_u : {}".format(F_u))

f_x = np.array([1, -2, 4, -3, 1, -2, 4, -3])
F_u = np.fft.fft(f_x)
print("F_u : {}".format(F_u))




