import numpy as np
import matplotlib.pyplot as plt
import pylab
import math

from HW6_function import show_img
from HW6_function import filter_Sobel
from HW6_function import filter_Sobel_x
from HW6_function import filter_Sobel_y
from HW6_function import normalize_255
from HW6_function import hist_unnormalized
from HW6_function import color_edge_detection_2

#測試圖片 : "lenna-RGB.tif"、"Visual resolution.gif"
img = plt.imread("lenna-RGB.tif")
print("img.shape : {}".format(img.shape))
plt.figure()
plt.imshow(img, cmap = None, vmin = 0, vmax = 255)
plt.title("ORIGIN")


#計算影像的梯度
# img_r = img[:, :, 0]
# img_r_gx = filter_Sobel_x(img_r)
# img_r_gy = filter_Sobel_y(img_r)
# img_g = img[:, :, 1]
# img_g_gx = filter_Sobel_x(img_g)
# img_g_gy = filter_Sobel_y(img_g)
# img_b = img[:, :, 2]
# img_b_gx = filter_Sobel_x(img_b)
# img_b_gy = filter_Sobel_y(img_b)


# gxx = np.abs(img_r_gx) ** 2 + np.abs(img_g_gx) ** 2 + np.abs(img_b_gx) ** 2
# gyy = np.abs(img_r_gy) ** 2 + np.abs(img_g_gy) ** 2 + np.abs(img_b_gy) ** 2
# gxy = img_r_gx*img_r_gy + img_g_gx*img_g_gy + img_b_gx*img_b_gy

# theta_xy = (1/2)*np.arctan2(2*gxy, gxx-gyy)
# F_theta = ((1/2)*( (gxx + gyy) + (gxx - gyy)*np.cos(2*theta_xy) + 2*gxy*np.sin(2*theta_xy) )) ** (1/2)
# F_theta = normalize_255(F_theta)

# show_img(img_r_gx, "Reds", "gray")
# show_img(img_r_gy, "Reds", "gray")
# show_img(F_theta, "After gradient computed in RGB(1)", "gray")
#計算影像的梯度

img_gradient = color_edge_detection_2(img)
show_img(img_gradient, "After gradient computed in RGB", "gray")

plt.show()


