import matplotlib.pyplot as plt
from HW3_4_function import show_img
from HW3_4_function import kernel_Guassian
from HW3_4_function import convolve2D
from HW3_4_function import filter_Sobel
from HW3_4_function import filter_Laplacian
from HW3_4_function import normalize_255
import numpy as np
import cv2
import math


#測試的圖 : 'Bodybone.bmp' 和 'fish.jpg'
#讀取圖片
img = plt.imread("fish.jpg")

# show_img(img, "ORIGIN")
img_shape = img.shape
print("img.shape : {}".format(img.shape))
#讀取圖片


#將brg取出
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]
#將brg取出


#Sobel filter
b_S = filter_Sobel(b)
g_S = filter_Sobel(g)
r_S = filter_Sobel(r)

b_f_S = b + b_S
g_f_S = g + g_S
r_f_S = r + r_S
b_f_S = normalize_255(b_f_S)
g_f_S = normalize_255(g_f_S)
r_f_S = normalize_255(r_f_S)

img_f_S = np.dstack([b_f_S, g_f_S, r_f_S])
img_f_S = img_f_S.astype('uint8')  

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img);plt.title("ORIGIN")

plt.subplot(1, 2, 2)
plt.imshow(img_f_S);plt.title("Sobel filter")
#Sobel filter



#Laplacian filter
b_L = filter_Laplacian(b)
g_L = filter_Laplacian(g)
r_L = filter_Laplacian(r)


#Bodybone.bmp : c = 4.5
#fish.jpg : c = -10
c = -10  #常數
b_f_L = b + (c*b_L)
g_f_L = g + (c*g_L)
r_f_L = r + (c*r_L)
b_f_L = normalize_255(b_f_L)
g_f_L = normalize_255(g_f_L)
r_f_L = normalize_255(r_f_L)
print("np.max(g_f_L) : {}".format(np.max(g_f_L)))
print("np.min(g_f_L) : {}".format(np.min(g_f_L)))

img_f_L = np.dstack([b_f_L, g_f_L, r_f_L])
img_f_L = img_f_L.astype('uint8')  

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img);plt.title("ORIGIN")

plt.subplot(1, 2, 2)
plt.imshow(img_f_L);plt.title("Laplacian filter")
#Laplacian filter

plt.show()