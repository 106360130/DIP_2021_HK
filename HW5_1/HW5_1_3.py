"""
Reference :
複數運算
https://www.runoob.com/python/python-func-complex.html
https://stackoverflow.com/questions/8498310/can-i-calculate-exp12j-in-python/8498322


在陣列中存複數
https://stackoverflow.com/questions/22016847/assigning-complex-values-to-numpy-arrays
"""


"""
book-cover-blurred.tif
用Wiener filtering去還原
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from math import e
from HW5_1_function import show_img
from HW5_1_function import normalize_255
from HW5_1_function import wiener_filtering

#讀取圖片
img = plt.imread("book-cover-blurred.tif")
print("img.shape : {}".format(img.shape))
show_img(np.abs(img), "Origin", "gray")
# plt.get_current_fig_manager().full_screen_toggle()
#讀取圖片

#原圖做FT
g_ft = np.fft.fft2(img)
g_ft = np.fft.fftshift(g_ft)
# plt.figure()
# plt.imshow(np.log(np.abs(g_ft)), cmap = "gray")
# plt.title("g_ft")
# print("type(g_ft) : {}".format(type(g_ft)))
#原圖做FT


#預估函數
H = np.zeros((img.shape[0], img.shape[1]),dtype=complex)
T = 1  #曝光時間
a = b = 0.1  #影像移動的距離
j = complex(0, 1)
# print("j : {}".format(j))
for u in range(img.shape[0]) :   
    for v in range(img.shape[1]) :
        temp_uv = ((u - (img.shape[0]/2))*a + (u - (img.shape[1]/2))*b)
        pi = math.pi

        if(temp_uv == 0):
            H[u][v] = 1  #隨意給的值，不然會出錯

        else :
            temp_T = T / (pi * temp_uv)
            temp_sin = math.sin(pi * temp_uv)
            temp_e = e ** (-j * pi * temp_uv)
            # print("temp_T : {}".format(temp_T))
            # print("temp_sin : {}".format(temp_sin))
            # print("temp_e : {}".format(temp_e))
            # print("temp_T * temp_sin * temp_e : {}".format(temp_T * temp_sin * temp_e))
            H[u][v] = temp_T * temp_sin * temp_e

# plt.figure()
# plt.imshow(np.log(np.abs(H)), cmap = "gray")
# plt.title("H")
# H = np.fft.fftshift(H)  #錯的
#預估函數



K = 0.025
# temp_H = 1 / H
# temp_H2 = (np.abs(H) ** 2) / ((np.abs(H) ** 2) + K)
# F_hat = temp_H * temp_H2 * g_ft
# temp = np.conj(H) / (np.abs(H) ** 2 + K)
# F_hat = temp * g_ft
F_hat = wiener_filtering(g_ft, H, K)


F_hat[0][0] = 0
# print("F_hat : {}".format(F_hat))
# plt.figure()
# plt.imshow(np.log(np.abs(F_hat)), cmap = "gray")
# plt.title("F_hat")


F_hat = np.fft.fftshift(F_hat)  #將其移回正確的位置
# plt.figure()
# plt.imshow(np.log(np.abs(F_hat)), cmap = "gray")
# plt.title("F_hat(after shift)")

img_new = np.fft.ifft2(F_hat)
# print("img_new : {}".format(img_new))


# plt.figure()
# plt.imshow(np.log(np.abs(img_new)), cmap = "gray")
# plt.title("img_new")

# show_img(np.abs(img_new), "img_new", "gray")
# print("img_new : {}".format(img_new))

img_new_normalize = normalize_255(np.abs(img_new))
show_img(img_new_normalize, "img_new : {}".format(a), "gray")
# plt.get_current_fig_manager().full_screen_toggle()

plt.show()