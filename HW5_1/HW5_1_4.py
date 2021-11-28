"""
Reference :
Inverse filter
https://github.com/pratscy3/Inverse-Filtering/blob/master/full_inverse.py
"""

"""
"Fig5.25.jpg"
用Inverse filtering去還原
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from HW5_1_function import show_img
from HW5_1_function import Butterworth_Lowpass
from HW5_1_function import range_limited_255
from HW5_1_function import wiener_filtering
from HW5_1_function import normalize_255

#讀取圖片
img = plt.imread("Fig5.25.jpg")
print("img.shape : {}".format(img.shape))
show_img(np.abs(img), "Origin", "gray")
#讀取圖片


img_new = np.zeros(img.shape)


#預估函數
H = np.zeros((img.shape[0], img.shape[1]))
#M和N是中心點座標
M = img.shape[0]
N = img.shape[1]
#M和N是中心點座標
k = 0.0025
for u in range(img.shape[0]):
    for v in range(img.shape[1]):

        temp_M = math.pow(u - (M/2), 2)
        temp_N = math.pow(v - (N/2), 2)
        temp_MN = temp_M + temp_N
        temp = -k*math.pow(temp_MN, (5/6))

        H[u, v] = math.exp(temp)


plt.figure()
plt.imshow(H, cmap = "gray")
plt.title("H")
#預估函數

#butternworth
P = img.shape[0]
Q = img.shape[1]
m = img.shape[0]/2
n = img.shape[1]/2
D0 = 70
N = 10
Bw_lowpass = Butterworth_Lowpass(P, Q, D0, N, m, n)
plt.figure()
plt.imshow(np.abs(Bw_lowpass), cmap = "gray")
plt.title("Bw_lowpass")
#butternworth

# print("type((img.shape[2]) : {}".format(type(img.shape[2])))
for i in range(img.shape[2]):
    g = img[:, :, i]  #將brg取出

    #做FT
    g_ft = np.fft.fft2(g)
    g_ft = np.fft.fftshift(g_ft)

    # plt.figure()
    # plt.imshow(np.log(np.abs(g_ft)), cmap = "gray")
    # plt.title("g_ft")

    # print("max(g_ft) : {}".format(np.max(np.log(np.abs(g_ft)))))
    #做FT

    K = 0.025
    F_hat = wiener_filtering(g_ft, H, K)
    # plt.figure()
    # plt.imshow(np.log(np.abs(F_hat)), cmap = "gray")
    # plt.title("F_hat(after H)")


    #butternworth
    #F_hat = F_hat * Bw_lowpass

    # plt.figure()
    # plt.imshow(np.log(np.abs(F_hat)), cmap = "gray")
    # plt.title("F_hat(after butternworth)")
    #butternworth


    F_hat = np.fft.fftshift(F_hat)  #將其移回正確的位置
    f_hat = np.fft.ifft2(F_hat)

    #f_hat = range_limited_255(np.abs(f_hat))
    img_new[:,:,i] = normalize_255(np.abs(f_hat))
    # plt.figure()
    # plt.imshow(np.abs(f_hat), cmap = "gray")
    # plt.title("F_hat")


    # show_img(np.abs(f_hat), "F_hat(show_img)", "gray")
img_new = img_new.astype('uint8')
show_img(np.abs(img_new), "F_hat(show_img)", "gray")

plt.figure()
plt.imshow(np.abs(img_new), cmap = "gray")
plt.title("img_new")
plt.show()