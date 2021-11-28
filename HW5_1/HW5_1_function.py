import matplotlib.pyplot as plt
import numpy as np
import math

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


def Butterworth_Lowpass(P, Q, D0, N, m, n):
    # "m" : filter要濾掉的區域的中心點位置
    # "n" : filter要濾掉的區域的中心點位置
    # "P" : filter的size
    # "Q" : filter的size
    # "D0" : 亮暗界線
    # "N" : 亮暗之間的平滑程度

    Bw_lowpass = np.zeros((P, Q))
    a = math.pow(D0, (2 * N)) 

    for u in range(P) :
        for v in range(Q) :
            temp = math.pow((u-(m+1.0)), 2) + math.pow((v-(n+1.0)), 2)
            Bw_lowpass[u][v] = 1 / (1 + math.pow(temp, N) / a)

    return Bw_lowpass

#最大值不能大於255 且 最小值不能小於0
def range_limited_255(img) :
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j] > 255):
                img[i][j] = 255
            
            if(img[i][j] < 0):
                img[i][j] = 0

    return img      
#最大值不能大於255 且 最小值不能小於0

#將數值normalize至0到255的範圍
def normalize_255(img) :
    img = img - np.min(img)
    img = (img / np.max(img))*255

    return img  

#將數值normalize至0到255的範圍


#Inverse filtering
def inverse_filtering(img_ft, H):
    F_hat = g_ft / H

    return F_hat  
#Inverse filtering


#Wiener filtering
def wiener_filtering(img_ft, H, K):
    #img_ft  #img經FT轉換
    #H  #評估函數
    #K = 10  #常數


    temp_H = 1 / H
    temp_H2 = (np.abs(H) ** 2) / (np.abs(H) ** 2 + K)
    F_hat = temp_H * temp_H2 * img_ft
    # temp = np.conj(H) / (np.abs(H) ** 2 + K)
    # F_hat = temp * img_ft

    return F_hat  
#Wiener filtering