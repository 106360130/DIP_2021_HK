"""
Reference:
開灰階圖
https://blog.gtwang.org/programming/opencv-basic-image-read-and-write-tutorial/

開檔失敗:路徑錯誤
https://blog.csdn.net/JohinieLi/article/details/70855058

Numpy 
https://blog.techbridge.cc/2020/08/24/numpy-zen-intro-tutorial/

exp()
https://www.runoob.com/python/func-number-exp.html

指數
https://googoodesign.gitbooks.io/-ezpython/content/unit-1.html

一維轉二維
https://jennaweng0621.pixnet.net/blog/post/404217932-%5Bpython%5D-%E4%BD%BF%E7%94%A8numpy%E5%B0%87%E4%B8%80%E7%B6%AD%E8%BD%89%E4%BA%8C%E7%B6%AD%E6%88%96%E5%A4%9A%E7%B6%AD

格式化輸出小數
https://blog.jaycetyle.com/2018/01/python-format-string/

產生等差數列
http://softcans.blogspot.com/2019/01/python-range-func.html

多維變一維
https://blog.csdn.net/tymatlab/article/details/79009618


"""

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import math
from mpl_toolkits.mplot3d import axes3d
from HW3_3_function import convolve2D
from HW3_3_function import show_img_gray
from HW3_3_function import kernel_Guassian


#以gray的方式讀進圖片
# img = cv2.imread('checkerboard1024-shaded.tif', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('N1.bmp', cv2.IMREAD_GRAYSCALE)
#print("img : {}".format(img))
img2 = cv2.imread('N1.bmp', cv2.IMREAD_GRAYSCALE)
#print("img2.shape : {}".format(img2.shape))
show_img_gray(img2, "N1.bmp")
#print("type(img) : {}".format(type(img)))


# plt.figure()
# plt.imshow(img, cmap = plt.cm.gray)  #灰階形式顯示
show_img_gray(img, "ORIGIN")


#測試2D convolution
"""
kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
img_result = convolve2D(img, kernel, padding=0, strides=1)
print("img_result : {}".format(img_result))
print("img.shape : {}".format(img.shape))
print("img_result.shape : {}".format(img_result.shape))


plt.figure()
plt.imshow(img_result, cmap = plt.cm.gray)
plt.title("kernel_1")
"""
#測試2D convolution





#製造Guassian kernel
K = 1
sigma = 64  #1024/16(圖中的小框框size)
m = 257  #64*4(圖中的小框框size的4倍)
kernel_G = kernel_Guassian(K, sigma, m)
print("kernel_G : {}".format(kernel_G))
#製造Guassian kernel


# print("kernel_G[1][1] : {:f}".format(kernel_G[1][1]))
# print("kernel_G : {}".format(kernel_G))
# print("type(kernel_G) : {}".format(type(kernel_G)))

show_img_gray(kernel_G, "kernel_G")

#3D作圖
"""
plt.figure()
X = Y = list(range(-m_half, m_half+1))
Z = kernel_G
ax = plt.axes(projection="3d")
ax.plot_wireframe(X, Y, Z, color='green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
"""
#3D作圖

size_img = img.shape
print("size_img[0] : {}, size_img[1] : {}".format(size_img[0], size_img[1]))
"""
size_img_padding = (m-1) + size_img[0]
img_padding_1D = [0]*size_img_padding*size_img_padding
img_padding = np.array(img_padding_1D).reshape(size_img_padding, size_img_padding)

for i in range(size_img[0]):
    for j in range(size_img[0]):
        img_padding[i+m_half][j+m_half] = img[i][j]
"""

"""
plt.figure()
plt.imshow(img_padding, cmap = plt.cm.gray)
plt.title("img_padding")
"""



#"img"做完convolution
img_c = convolve2D(img, kernel_G, padding=0, strides=1)
# print("img_result : {}".format(img_result_2))
show_img_gray(img_c, "img_convolution")
#"img"做完convolution

#將"img_c"放大
img_c_2 = cv2.resize(img_c, (size_img[1], size_img[0]))
show_img_gray(img_c_2, "img_convolution_resize")
print("img_c_2 : {}".format(img_c_2))
print("np.max(img_c_2) : {}".format(np.max(img_c_2)))
print("np.min(img_c_2) : {}".format(np.min(img_c_2)))
#將"img_c"放大

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

img_final = img/img_c_2 *np.max(img_c_3)
show_img_gray(img_final, "img_final")


# 顯示圖形
plt.show()

