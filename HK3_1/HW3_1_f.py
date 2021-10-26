import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


#統計每個強度有多少pixels
def hist_unnormalized(img_ravel, L):
    hist_unnormalized = [0]*L

    for i in range(len(img_ravel)):
        hist_unnormalized[img_ravel[i]] = hist_unnormalized[img_ravel[i]] + 1

    return hist_unnormalized
    

#統計每個強度佔圖片多少比例
def hist_normalized(img_ravel, L, img_size):

    hist_img = hist_unnormalized(img_ravel, L)

    hist_img_normalized = [0]*L

    for i in range(len(hist_img)):
        hist_img_normalized[i] = hist_img[i] / (img_size[0]*img_size[1])

    return hist_img_normalized

#找到新的圖案標準
def intensities_new(img_ravel, L, img_size):
    hist_img_normalized = hist_normalized(img_ravel, L, img_size)

    #將比例累加
    intensity_new = [0]*L
    for i in range(L):
        if i == 0:
            intensity_new[i] = hist_img_normalized[i]

        else:   
            intensity_new[i] = intensity_new[i-1] + hist_img_normalized[i]
    #將比例累加

    for i in range(L):
        intensity_new[i] = round(intensity_new[i]*(L-1))

    #print("intensity_new[L-1] : {}".format(intensity_new[L-1]))
    return intensity_new


def intensities_mean(img_ravel, L, img_size):
    hist_img_normalized = hist_normalized(img_ravel, L, img_size)
    
    intensity_mean = 0
    for i in range(len(hist_img_normalized)):
        intensity_mean = intensity_mean + (i*hist_img_normalized[i])

    return intensity_mean

def intensities_standard(img_ravel, L, img_size):
    hist_img_normalized = hist_normalized(img_ravel, L, img_size)
    intensity_mean = intensities_mean(img_ravel, L, img_size)

    intensity_standard = 0
    for i in range(len(hist_img_normalized)):
        diff_temp = i - intensity_mean
        intensity_standard = intensity_standard + math.pow(diff_temp, 2)*hist_img_normalized[i]
    
    return intensity_standard


##########################################################

#HW3_1_1
img = cv2.imread("aerial_view.tif", cv2.IMREAD_GRAYSCALE)  #以灰階的形式讀取
L = int(math.pow(2, 8))

hist_img = hist_unnormalized(img.ravel(), L)  #計算hist_unnormalized

#秀hist_unnormalized結果
plt.figure()
plt.subplot(1, 2, 1)
plt.bar(range(len(hist_img)), hist_img)
plt.xlabel("r$_{k}$")
plt.ylabel("h(r$_{k}$)")
plt.title("HISTOGRAM")

plt.subplot(1, 2, 2)
plt.imshow(img, cmap = 'gray')
plt.title("ORIGIN")
#秀hist_unnormalized結果
#HW3_1_1


#HW3_1_2
#Histogram equalization
img_size = img.shape
rk_new = intensities_new(img.ravel(), L, img_size)  #計算intensities_new

img_2_oneArray = [0]*len(img.ravel())
for i in range(len(img.ravel())):
    img_2_oneArray[i] = rk_new[ img.ravel()[i] ]

img_2 = np.array(img_2_oneArray).reshape(img_size[0], img_size[1])
img_2_size = img_2.shape
hist_img_2 = hist_unnormalized(img_2.ravel(), L)

plt.figure()
plt.subplot(1, 2, 1)
plt.bar(range(len(hist_img_2)), hist_img_2)
plt.xlabel("r$_{k}$")
plt.ylabel("h(r$_{k}$)")
plt.title("HISTOGRAM")

plt.subplot(1, 2, 2)
plt.imshow(img_2, cmap = 'gray')
plt.title("AFTER HISTOGRAM EQUALIZATION")
#Histogram equalization
#HW3_1_2


#HW3_1_3
#Histogram Matching (specificiation)
P_z = [0]*L
c = 0.0005968
for i in range(L):
    P_z[i] = c*math.pow(i, 0.4)

rk_specificiation = intensities_new(img.ravel(), L, img_size)

#藉由S和G找Z，當S和G相差最少時，Z值為多少
rk_final = [0]*L
for i in range(L):
    diff_small = 255
   
    for j in range(L):
        diff = abs(rk_new[i] - rk_specificiation[j])
        if(rk_new[i] == rk_specificiation[j]):
            rk_final[i] = j
            break
#藉由S和G找Z，當S和G相差最少時，Z值為多少


#Histogram equalization
img_3_oneArray = [0]*len(img.ravel())
for i in range(len(img.ravel())):
    img_3_oneArray[i] = rk_specificiation[ img.ravel()[i] ]

img_3 = np.array(img_3_oneArray).reshape(img_size[0], img_size[1])
img_3_size = img_3.shape
hist_img_3 = hist_unnormalized(img_3.ravel(), L)
#Histogram equalization

plt.figure()
plt.subplot(1, 2, 1)
plt.bar(range(len(hist_img_3)), hist_img_3)
plt.xlabel("r$_{k}$")
plt.ylabel("h(r$_{k}$)")
plt.title("HISTOGRAM")

plt.subplot(1, 2, 2)
plt.title("AFTER HISTOGRA MATCHING")
plt.imshow(img_3, cmap = 'gray')
#Histogram Matching (specificiation)
#HW3_1_3

plt.show()











