"""
矩陣宣告(設定初值)
https://blog.csdn.net/qq_26948675/article/details/54318917
https://vimsky.com/zh-tw/examples/usage/python-numpy.ones.html


plt.imshow()的camp值
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
https://www.cnblogs.com/denny402/p/5122594.html

"""

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2


#2D convolution
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
#2D convolution

def show_img(img, title_img, cmap_appinted = None):
    """
    'gray' : 灰階圖
    'Reds' : 紅色
    'Greens' : 綠色
    'Blues' : 藍色
    'viridis' : 彩色圖，"cmap"的預設值
    """
    # print("cmap_appinted : {}".format(cmap_appinted))
  
    plt.figure()
    plt.imshow(img, cmap = cmap_appinted, vmin = 0, vmax = 255)
    plt.title(title_img)


#製造Guassian kernel
def kernel_Guassian(K, sigma, m):
    # K = 1
    # sigma = 64  #1024/16(圖中的小框框size)
    # m = 257  #64*4(圖中的小框框size的4倍)
    m_half = math.floor(m/2)
    kernel_G = np.zeros((m,m))  #矩陣宣告預設值
    print("type(kernel_G) : {}".format(type(kernel_G)))

    i = j = 0

    # print("type(i) : {}".format(type(i)))
    # print("range : {} ~ {}".format(-m_half, m_half+1))

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

    kernel_G_sum = kernel_G.sum()
    kernel_G = kernel_G/kernel_G_sum
    # print("kernel_G : {}".format(kernel_G))

    return kernel_G
#製造Guassian kernel

#Sobel kernel所找到的邊緣
def filter_Sobel(img):
    kernel_S_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  #g_x
    kernel_S_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  #g_y

    img_S_x = convolve2D(img, kernel_S_x, padding=0, strides=1)
    img_S_y = convolve2D(img, kernel_S_y, padding=0, strides=1)

    img_M_xy = ((img_S_x ** 2) + (img_S_y ** 2)) ** 0.5
    img_M_xy_r = cv2.resize(img_M_xy, (img.shape[1], img.shape[0]))

    return img_M_xy_r
#Sobel kernel所找到的邊緣

#Laplacian kernel所找到的邊緣
def filter_Laplacian(img):
    kernel_L = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    img_L = convolve2D(img, kernel_L, padding=0, strides=1)
    img_L_r = cv2.resize(img_L, (img.shape[1], img.shape[0]))
    return img_L_r

#Laplacian kernel所找到的邊緣

#最大值不能大於255 且 最小值不能小於255
def normalize_255(img) :
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j] > 255):
                img[i][j] = 255
            
            if(img[i][j] < 0):
                img[i][j] = 0

    return img      
#最大值不能大於255 且 最小值不能小於255

