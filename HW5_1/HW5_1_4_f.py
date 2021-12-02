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
#讀取圖片

#原圖做FT
g_ft = np.fft.fft2(img)
g_ft = np.fft.fftshift(g_ft)
#原圖做FT


#預估函數
H = np.zeros((img.shape[0], img.shape[1]),dtype=complex)
T = 1  #曝光時間
a = b = 0.1  #影像移動的距離
j = complex(0, 1)
# print("j : {}".format(j))
for u in range(img.shape[0]) :   
    for v in range(img.shape[1]) :
        temp_uv = ((u - (img.shape[0]/2))*a + (v - ((img.shape[1]/2))) *b)
        pi = math.pi

        if(temp_uv == 0):
            H[u][v] = 1  #隨意給的值，不然會出錯

        else :
            temp_T = T / (pi * temp_uv)
            temp_sin = math.sin(pi * temp_uv)
            temp_e = e ** (-j * pi * temp_uv)

            H[u][v] = temp_T * temp_sin * temp_e
#預估函數


K = 0.001
F_hat = wiener_filtering(g_ft, H, K)

F_hat[0][0] = 0

F_hat = np.fft.fftshift(F_hat)  #將其移回正確的位置
img_new = np.fft.ifft2(F_hat)

img_new_normalize = normalize_255(np.abs(img_new))
show_img(img_new_normalize, "img_new : {}".format(a), "gray")

plt.show()