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

#Sobel kernel所找到的邊緣
def filter_Sobel_x(img):
    kernel_S_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  #g_x
    img_S_x = convolve2D(img, kernel_S_x, padding=0, strides=1)
    img_S_x = cv2.resize(img_S_x, (img.shape[1], img.shape[0]))

    return img_S_x
#Sobel kernel所找到的邊緣

#Sobel kernel所找到的邊緣
def filter_Sobel_y(img):
    kernel_S_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  #g_y
    img_S_y = convolve2D(img, kernel_S_y, padding=0, strides=1)
    img_S_y = cv2.resize(img_S_y, (img.shape[1], img.shape[0]))

    return img_S_y
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


#Find gradient image, method 1
def color_edge_detection_1(img):

    img_g = np.zeros((img.shape[0], img.shape[1]))  #RGB影像的梯度計算加上總和
    for i in range(img.shape[2]):
        temp_img_g = np.zeros((img.shape[0], img.shape[1]))
        temp_img = img[:, :, i]
        temp_img_g = filter_Sobel(temp_img)  #計算影像的梯度
        # temp_img_g = normalize_255(temp_img_g)
        # show_img(temp_img_g, None, "gray")
        # print("max(temp_img_g) : {}".format(np.max(temp_img_g)))
        # print("min(temp_img_g) : {}".format(np.min(temp_img_g)))

        #直接相加，給權重
        if(i == 0) :
            weight = 0.4

        elif(i == 1) : 
            weight = 0.3

        else:
            weight = 0.2
        #直接相加，給權重

        img_g = img_g + weight*temp_img_g
        img_g = normalize_255(img_g)

    return img_g
#Find gradient image, method 1


#Find gradient image, method 2
def color_edge_detection_2(img):

    #計算影像的梯度
    img_r = img[:, :, 0]
    img_r_gx = filter_Sobel_x(img_r)
    img_r_gy = filter_Sobel_y(img_r)
    img_g = img[:, :, 1]
    img_g_gx = filter_Sobel_x(img_g)
    img_g_gy = filter_Sobel_y(img_g)
    img_b = img[:, :, 2]
    img_b_gx = filter_Sobel_x(img_b)
    img_b_gy = filter_Sobel_y(img_b)
    #計算影像的梯度

    #rgb的gradient不同的相加方式
    gxx = np.abs(img_r_gx) ** 2 + np.abs(img_g_gx) ** 2 + np.abs(img_b_gx) ** 2
    gyy = np.abs(img_r_gy) ** 2 + np.abs(img_g_gy) ** 2 + np.abs(img_b_gy) ** 2
    gxy = img_r_gx*img_r_gy + img_g_gx*img_g_gy + img_b_gx*img_b_gy

    theta_xy = (1/2)*np.arctan2(2*gxy, gxx-gyy)
    F_theta = ((1/2)*( (gxx + gyy) + (gxx - gyy)*np.cos(2*theta_xy) + 2*gxy*np.sin(2*theta_xy) )) ** (1/2)
    F_theta = normalize_255(F_theta)
    #不同的相加方式
    
    return F_theta
#Find gradient image, method 2