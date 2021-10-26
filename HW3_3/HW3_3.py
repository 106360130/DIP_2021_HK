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

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

#以gray的方式讀進圖片
img = cv2.imread('checkerboard1024-shaded.tif', cv2.IMREAD_GRAYSCALE)
print("img : {}".format(img))
img2 = cv2.imread('N1.bmp', cv2.IMREAD_GRAYSCALE)
print("img2.shape : {}".format(img2.shape))
plt.figure()
plt.imshow(img2, cmap = 'gray', vmin = 0, vmax = 255)
plt.title("N1.bmp")


print("type(img) : {}".format(type(img)))


plt.figure()
#plt.imshow(img, cmap = plt.cm.gray)  #灰階形式顯示
plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 255)
plt.title("ORIGIN")


kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
img_result = convolve2D(img, kernel, padding=0, strides=1)
print("img_result : {}".format(img_result))
print("img.shape : {}".format(img.shape))
print("img_result.shape : {}".format(img_result.shape))


"""
plt.figure()
plt.imshow(img_result, cmap = plt.cm.gray)
plt.title("kernel_1")
"""



K = 1
sigma = 64  #1024/16(圖中的小框框size)
m = 257  #64*4(圖中的小框框size的4倍)
m_half = math.floor(m/2)
kernel_G_1D = [0.0]*m*m  #要存小數宣告時也要存小數
kernel_G = np.array(kernel_G_1D).reshape(m, m)

i = j = 0
print("type(i) : {}".format(type(i)))
print("range : {} ~ {}".format(-m_half, m_half+1))
for s in range(-m_half, m_half+1):
    for t in range(-m_half, m_half+1):
        r_temp = math.pow(s, 2) + math.pow(t, 2)
        #print("r_temp : {}".format)
        r = math.pow(r_temp, 0.5)
        kernel_G[i][j] = K*math.exp(-r/(2*math.pow(sigma, 2)))
        
        #print("s : {}, t : {}".format(s, t))
        #print("kernel_G[{}][{}] : {}".format(i, j, kernel_G[i][j]))

        j = j+1


    i = i+1
    j = 0

kernel_G_sum = 0
for i in range(m):
    for j in range(m):
        kernel_G_sum = kernel_G_sum + kernel_G[i][j]

for i in range(m):
    for j in range(m):
        kernel_G[i][j] = kernel_G[i][j]/kernel_G_sum



    

print("kernel_G[1][1] : {:f}".format(kernel_G[1][1]))
print("kernel_G : {}".format(kernel_G))
print("type(kernel_G) : ".format(type(kernel_G)))

plt.figure()
plt.imshow(kernel_G, cmap = 'gray', vmin = 0, vmax = 255)
plt.title("kernel_G")

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





"""
img_result_2 = convolve2D(img, kernel_G, padding=0, strides=1)
print("img_result : {}".format(img_result_2))
plt.figure()
plt.imshow(img_result_2, cmap = 'gray', vmin = 0, vmax = 255)
plt.title("img_result_2")

img_new = cv2.resize(img_result_2, (size_img[0], size_img[0]))
plt.figure()
plt.imshow(img_new, cmap = 'gray', vmin = 0, vmax = 255)
plt.title("img_new")

print("img.shape : {}".format(img.shape))
print("img_result_2.shape : {}".format(img_result_2.shape))
print("img_new.shape : {}".format(img_new.shape))

img_new2_1D = [0.0]*size_img[0]*size_img[0]
img_new2 = np.array(img_new2_1D).reshape(size_img[0], size_img[0])
for i in range(size_img[0]):
    for j in range(size_img[0]):
        img_new2[size_img[0]-i-1][size_img[0]-j-1] = img_new[i][j]

plt.imshow(img_new2, cmap = 'gray', vmin = 0, vmax = 255)
plt.title("img_new2")   




img_final_1D = [0]*size_img[0]*size_img[0]
img_final = np.array(img_final_1D).reshape(size_img[0], size_img[0])
for i in range(size_img[0]):
    for j in range(size_img[0]):
        img_final[i][j] = img[i][j] - img_new[i][j] + img_new2[i][j]


plt.figure()
plt.imshow(img_final, cmap = 'gray', vmin = 0, vmax = 255)
plt.title("img_final")
"""

# 顯示圖形
plt.show()

