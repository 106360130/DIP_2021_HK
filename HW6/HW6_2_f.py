import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import math

from HW6_function import show_img
from HW6_function import color_edge_detection_1
from HW6_function import color_edge_detection_2

frames=imageio.mimread('Visual resolution.gif') #frames是一個列表，存儲gif裏面的每一幀，長度就是幀個數
# print(len(frames)) #imageio的圖片類型爲imageio.core.util.Array

for f in frames: #把每一幀轉爲numpy
	img = np.array(f)
	
    
print("img.shape : {}".format(img.shape))
plt.figure()
plt.imshow(img, cmap = None, vmin = 0, vmax = 255)
plt.title("ORIGIN")


img_gradient = color_edge_detection_1(img)
show_img(img_gradient, "Gradient Images 1", "gray")

img_gradient_2 = color_edge_detection_2(img)
show_img(img_gradient_2, "Gradient Images 2", "gray")

plt.show()