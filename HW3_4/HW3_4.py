import matplotlib.pyplot as plt
from HW3_4_function import show_img
from HW3_4_function import kernel_Guassian
from HW3_4_function import convolve2D
import numpy as np
import cv2
import math


#測試的圖 : 'Bodybone.bmp' 和 'fish.jpg'
img = plt.imread("Bodybone.bmp")

show_img(img, "ORIGIN")
img_shape = img.shape
print("img.shape : {}".format(img.shape))

b = np.zeros((img.shape[0], img.shape[1]))  #正確的
b = img[:, :, 0]

"""
K = 1; sigma = 5; m = 31
kernel_G = kernel_Guassian(K, sigma, m)
b_blurred = convolve2D(b, kernel_G, padding=0, strides=1)
show_img(b, "ORIGIN", "gray")
show_img(b_blurred, "BlURRED", "gray")

print("b_blurred.shape : {}".format(b_blurred.shape))
b_blurred_r = cv2.resize(b_blurred, (img_shape[1], img_shape[0]))

b_UM = b - b_blurred_r  #"b"的unsharp mask
plt.figure()
plt.imshow(b_UM, cmap = "gray")
# show_img(b_UM, "UNSHARP MASK", "gray")

K_UM = 4.5  #unsharp mask的K參數, highboost method : K_UM > 1
b_final = b + b_UM
show_img(b_final, "UNSHARP MASK", "gray")
"""


"""
#Sobel operators
kernel_S_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  #g_x
kernel_S_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  #g_y
b_S_x = convolve2D(b, kernel_S_x, padding=0, strides=1)
b_S_y = convolve2D(b, kernel_S_y, padding=0, strides=1)
plt.figure()
plt.subplot(1,3,1)
plt.imshow(b_S_x, cmap = "gray")
plt.title("b_S_x")
plt.subplot(1,3,2)
plt.imshow(b_S_y, cmap = "gray")
plt.title("b_S_y")

M_xy = ((b_S_x ** 2) + (b_S_y ** 2)) ** 0.5
plt.subplot(1,3,3)
plt.imshow(M_xy*5, cmap = "gray")
plt.title("M_xy")
print("M_xy : {}".format(M_xy))
print("b_S_x : {}".format(b_S_x))
print("b_S_y : {}".format(b_S_y))
print("np.max(M_xy) : {}".format(np.max(M_xy)))
print("np.min(M_xy) : {}".format(np.min(M_xy)))
#Sobel operators

print("type(M_xy) : {}".format(type(M_xy)))

# for i in range(M_xy.shape[0]):
#     for j in range(M_xy.shape[1]):
#         print("M_xy[{}, {}] = {}".format(i, j, M_xy[i][j]))


# print("b_S_x[472, 907] : {}".format(b_S_x[472, 907]))
# print("b_S_y[472, 907] : {}".format(b_S_y[472, 907]))
M_xy_r = cv2.resize(M_xy, (img_shape[1], img_shape[0]))
b_final = b + M_xy_r
show_img(b_final, "Sobel", "gray")
"""

#Laplacian kernel
kernel_L = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
b_L = convolve2D(b, kernel_L, padding=0, strides=1)
b_L_r = cv2.resize(b_L, (img_shape[1], img_shape[0]))
print("b_L_r : {}".format(b_L_r))
plt.figure()
plt.imshow(b_L_r, cmap = "gray")

b_final = b - 4.5*b_L_r
show_img(b_final, "Laplacian", "gray")
#Laplacian kernel
print("np.max(b_L_r) : {}".format(np.max(b_L_r)))
print("np.min(b_L_r) : {}".format(np.min(b_L_r)))
b_L_r = b_L_r - np.min(b_L_r)  #先平移
b_L_r = b_L_r * (1/np.max(b_L_r)) * 255  #在標準化
plt.figure()
plt.imshow(b_L_r, cmap = "gray")

plt.show()