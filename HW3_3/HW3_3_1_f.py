import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import math
from mpl_toolkits.mplot3d import axes3d
from HW3_3_function import convolve2D
from HW3_3_function import show_img_gray
from HW3_3_function import kernel_Guassian


#測試的圖 : 'checkerboard1024-shaded.tif' 和 'N1.bmp'
img = plt.imread("checkerboard1024-shaded.tif")  #會看有多少通道而讀進多少
show_img_gray(img, "ORIGIN")


#製造Guassian kernel
K = 1
sigma = 64  #1024/16(圖中的小框框size)
m = 257  #64*4(圖中的小框框size的4倍)
kernel_G = kernel_Guassian(K, sigma, m)
print("kernel_G : {}".format(kernel_G))
show_img_gray(kernel_G, "kernel_G")
#製造Guassian kernel



size_img = img.shape
print("size_img[0] : {}, size_img[1] : {}".format(size_img[0], size_img[1]))


#"img"做完convolution
img_c = convolve2D(img, kernel_G, padding=0, strides=1)
show_img_gray(img_c, "img_convolution")
#"img"做完convolution

#將"img_c"放大
img_c_2 = cv2.resize(img_c, (size_img[1], size_img[0]))
show_img_gray(img_c_2, "img_convolution_resize")
print("img_c_2 : {}".format(img_c_2))
print("np.max(img_c_2) : {}".format(np.max(img_c_2)))
print("np.min(img_c_2) : {}".format(np.min(img_c_2)))
#將"img_c"放大

#用相減的方式
"""
#將"img_c_2"亮暗置換
img_c_3 = (np.max(img_c_2) + np.min(img_c_2)) - img_c_2
show_img_gray(img_c_3, "img_convolution_resize_flip")
print("np.max(img_c_3) : {}".format(np.max(img_c_3)))
print("np.min(img_c_3) : {}".format(np.min(img_c_3)))
#將"img_c_2"亮暗置換


print("img.shape : {}".format(img.shape))
print("img_c_2.shape : {}".format(img_c_2.shape))
print("img_c_3.shape : {}".format(img_c_3.shape))
img_final = img - img_c_2 + img_c_3
show_img_gray(img_final, "img_final")
"""
#用相減的方式

img_c_2 = img_c_2/np.max(img_c_2)
img_final = img/img_c_2
show_img_gray(img_final, "img_final")

plt.show()  # 顯示圖形

