"""
Reference :
格式化輸出(print)
https://www.runoob.com/python/att-string-format.html


"""

mport numpy as np
import matplotlib.pyplot as plt
import pylab
import math

from HW6_function import show_img
from HW6_function import color_edge_detection_1
from HW6_function import color_edge_detection_2

#測試圖片 : "lenna-RGB.tif"、"Visual resolution.gif"
img = plt.imread("lenna-RGB.tif")
print("img.shape : {}".format(img.shape))
plt.figure()
plt.imshow(img, cmap = None, vmin = 0, vmax = 255)
plt.title("ORIGIN")

img_gradient = color_edge_detection_1(img)
show_img(img_gradient, "Gradient Images 1", "gray")


img_gradient_2 = color_edge_detection_2(img)
show_img(img_gradient_2, "Gradient Images 2", "gray")

plt.show()

