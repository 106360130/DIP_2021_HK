import numpy as np
import matplotlib.pyplot as plt
import pylab
import math

from HW6_function import show_img
from HW6_function import filter_Sobel
from HW6_function import normalize_255
from HW6_function import hist_unnormalized
from HW6_function import color_edge_detection_1

#測試圖片 : "lenna-RGB.tif"、"Visual resolution.gif"
img = plt.imread("lenna-RGB.tif")
print("img.shape : {}".format(img.shape))
plt.figure()
plt.imshow(img, cmap = None, vmin = 0, vmax = 255)
plt.title("ORIGIN")


# img_g = np.zeros((img.shape[0], img.shape[1]))  #RGB影像的梯度計算加上總和
# for i in range(img.shape[2]):
#     temp_img_g = np.zeros((img.shape[0], img.shape[1]))
#     temp_img = img[:, :, i]
#     temp_img_g = filter_Sobel(temp_img)  #計算影像的梯度
#     temp_img_g = normalize_255(temp_img_g)
#     show_img(temp_img_g, None, "gray")
#     print("max(temp_img_g) : {}".format(np.max(temp_img_g)))
#     print("min(temp_img_g) : {}".format(np.min(temp_img_g)))

#     if(i == 0) :
#         weight = 0.4

#     elif(i == 1) : 
#         weight = 0.3

#     else:
#         weight = 0.2

#     img_g = img_g + weight*temp_img_g

#img_g = img_g / 3
#img_g = normalize_255(img_g)
# 加強邊緣
# temp_img_g = filter_Sobel(img_g)
# temp_img_g = normalize_255(temp_img_g)
# show_img(temp_img_g, "Sobel filter", "gray")
# print("max(temp_img_g) : {}".format(np.max(temp_img_g)))
# print("min(temp_img_g) : {}".format(np.min(temp_img_g)))
# img_g = img_g + temp_img_g
# 加強邊緣



# show_img(img_g, "After gradient computed in RGB(1)", "gray")
# img_g = img_g ** 1.1
# img_g = normalize_255(img_g)
# show_img(img_g, "After gradient computed in RGB(2)", "gray")
# print("max(img_g) : {}".format(np.max(img_g)))
# print("min(img_g) : {}".format(np.min(img_g)))

# L = int(math.pow(2, 8))
# img_g_ravel = np.ravel(img_g)
# print("type(img_g_ravel) : {}".format(type(img_g_ravel)))
# hist_img = hist_unnormalized(img_g_ravel, L)

# plt.figure()
# plt.bar(range(len(hist_img)), hist_img)
# plt.xlabel("r$_{k}$")
# plt.ylabel("h(r$_{k}$)")
# plt.title("HISTOGRAM")

# for i in range(img.shape[0]):
#     for j in range(img.shape[0]):
#         if(img_g[i][j] < 40):
#             img_g[i][j] = 0
        
#         else:
#             img_g[i][j] = 255
# show_img(img_g, "After gradient computed in RGB(3)", "gray")


img_gradient = color_edge_detection_1(img)
show_img(img_gradient, "After gradient computed in RGB", "gray")


plt.show()


