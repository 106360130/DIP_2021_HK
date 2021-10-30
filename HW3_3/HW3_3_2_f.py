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
img = plt.imread("N1.bmp")  #會看有多少通道而讀進多少
# plt.figure()
# plt.imshow(img)
# plt.title("ORIGIN")


#製造Guassian kernel
K = 1
sigma = 64  #1024/16(圖中的小框框size)
m = 257  #64*4(圖中的小框框size的4倍)
kernel_G = kernel_Guassian(K, sigma, m)
print("kernel_G : {}".format(kernel_G))
# show_img_gray(kernel_G, "kernel_G")
#製造Guassian kernel



size_img = img.shape
print("size_img[0] : {}, size_img[1] : {}".format(size_img[0], size_img[1]))

if(len(size_img) == 3) :
    b = np.zeros((img.shape[0], img.shape[1]))  #正確的
    g = np.zeros((img.shape[0], img.shape[1]))
    r = np.zeros((img.shape[0], img.shape[1]))

    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    """
    plt.figure()
    plt.imshow(r, cmap = 'Reds')
    plt.title("r")

    plt.figure()
    plt.imshow(g, cmap = 'Greens')
    plt.title("g")

    plt.figure()
    plt.imshow(b, cmap = 'Blues')
    plt.title("b")
    """

    #做convolution
    b_c = convolve2D(b, kernel_G, padding=0, strides=1)
    g_c = convolve2D(g, kernel_G, padding=0, strides=1)
    r_c = convolve2D(r, kernel_G, padding=0, strides=1)
    #做convolution

    #放大
    b_c_2 = cv2.resize(b_c, (size_img[1], size_img[0]))
    g_c_2 = cv2.resize(g_c, (size_img[1], size_img[0]))
    r_c_2 = cv2.resize(r_c, (size_img[1], size_img[0]))
    #放大


    b_c_2 = b_c_2/np.max(b_c_2)
    g_c_2 = g_c_2/np.max(g_c_2)
    r_c_2 = r_c_2/np.max(r_c_2)
 



    # print("np.max(b_c_2) : {}".format(np.max(b_c_2))) 
    # print("np.max(g_c_2) : {}".format(np.max(g_c_2))) 
    # print("np.max(r_c_2) : {}".format(np.max(r_c_2))) 
    # print("np.min(b_c_2) : {}".format(np.min(b_c_2))) 
    # print("np.min(g_c_2) : {}".format(np.min(g_c_2))) 
    # print("np.min(r_c_2) : {}".format(np.min(r_c_2))) 

    


    #"bgr"合併要用"int"型態
    b_final = np.around(b/b_c_2)
    g_final = np.around(g/g_c_2)
    r_final = np.around(r/r_c_2)
    #"bgr"合併要用"int"型態

    """
    plt.figure()
    plt.imshow(r_final, cmap = 'Reds')
    plt.title("r_final")

    plt.figure()
    plt.imshow(g_final, cmap = 'Greens')
    plt.title("g_final")

    plt.figure()
    plt.imshow(b_final, cmap = 'Blues')
    plt.title("b_final")
    """

img_final = cv2.merge([b_final, g_final, r_final])  #將bgr合併
# img_final = np.dstack([b_final, g_final, r_final])  #將bgr合併
img_final = img_final.astype('uint8')  

print("img_final.shape : {}".format(img_final.shape))
print("img_final : {}".format(img_final))

C = 150  #陰暗更明顯地顯示
b_c_2 = b_c_2*C
g_c_2 = g_c_2*C
r_c_2 = r_c_2*C

img_c = cv2.merge([b_c_2, g_c_2, r_c_2])  #將bgr合併
img_c = img_c.astype('uint8') 

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("ORIGIN")

plt.subplot(1, 3, 2)
plt.imshow(img_c)
plt.title("Lowpass filter")

plt.subplot(1, 3, 3)
plt.imshow(img_final)
plt.title("FINAL")




plt.show()  # 顯示圖形

