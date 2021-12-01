import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_img(img, title_img, cmap_appinted = None):
    # 'gray' : 灰階圖
    # 'Reds' : 紅色
    # 'Greens' : 綠色
    # 'Blues' : 藍色
    # 'viridis' : 彩色圖，"cmap"的預設值
    
    # print("cmap_appinted : {}".format(cmap_appinted))
  
    plt.figure()
    plt.imshow(img, cmap = cmap_appinted, vmin = 0, vmax = 255)
    plt.title(title_img)

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

#將數值normalize至0到255的範圍
def normalize_255(img) :
    img = img - np.min(img)
    img = (img / np.max(img))*255

    return img  
#將數值normalize至0到255的範圍

#統計每個強度有多少pixels
def hist_unnormalized(img_ravel, L):
    hist_unnormalized = [0]*L

    for i in range(len(img_ravel)):
        hist_unnormalized[int(img_ravel[i])] = hist_unnormalized[int(img_ravel[i])] + 1

    return hist_unnormalized
#統計每個強度有多少pixels
