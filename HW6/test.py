"""
import numpy as np

test = np.array([1, 2, 5, 6])
print("test : {}".format(test))

test = test - np.min(test)
test = (test / np.max(test)) * 100
print("test : {}".format(test))


test = np.array([1, 2, 5, 6])
test = test ** 2
test = test - np.min(test)
test = (test / np.max(test)) * 100
print("test : {}".format(test))
"""

import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt

frames=imageio.mimread('Visual resolution.gif') #frames是一個列表，存儲gif裏面的每一幀，長度就是幀個數
print(len(frames)) #imageio的圖片類型爲imageio.core.util.Array
cv_img=[]
for f in frames: #把每一幀轉爲numpy
	i = np.array(f)
	cv_img.append(i)

for c in cv_img:#顯示
    plt.figure()
    plt.imshow(c, cmap = None, vmin = 0, vmax = 255)
    plt.title("ORIGIN")
    print("c.shape : {}".format(c.shape))


plt.show()
