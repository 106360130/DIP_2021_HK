import matplotlib.pyplot as plt
from HW3_4_function import show_img
from HW3_4_function import kernel_Guassian
from HW3_4_function import convolve2D
import numpy as np
import cv2

#測試的圖 : 'Bodybone.bmp' 和 'fish.jpg'
img = plt.imread("Bodybone.bmp")

show_img(img, "ORIGIN")
img_shape = img.shape
print("img.shape : {}".format(img.shape))

b = np.zeros((img.shape[0], img.shape[1]))  #正確的
b = img[:, :, 0]

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

K_UM = 4.5  #unsharp mask的K參數
b_final = b + b_UM
show_img(b_final, "UNSHARP MASK", "gray")


plt.show()