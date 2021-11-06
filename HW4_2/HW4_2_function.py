"""
Reference :
Butterworth lowpass filter
https://www.twblogs.net/a/5b8d063d2b71771883397347

"""


import numpy as np
import math
import matplotlib.pyplot as plt
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

"""
m = n = 50
D0 = 10
N = 2
Bw_lowpass = Butterworth_Lowpass(m, n, D0, N)
plt.figure()
plt.imshow(Bw_lowpass, "gray")
plt.title("Butterworth Lowpass")
plt.show()
"""

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