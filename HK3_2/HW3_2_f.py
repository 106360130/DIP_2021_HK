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

def part_of_img(img, size_range, i_now, j_now):
    size_part = (size_range*2) + 1
    temp_img_oneArray = [0]*size_part*size_part
    temp_img = np.array(temp_img_oneArray).reshape(size_part, size_part)  #宣告二維矩陣

    x = y = 0
    for i in range(i_now - size_range, i_now + size_range + 1):
        for j in range(j_now - size_range, j_now + size_range + 1):
            temp_img[x][y] = img[i][j]
            y = y+1

        x = x+1
        y = 0

    temp_img_ravel = temp_img.ravel()
    return temp_img_ravel


#HW3_2_1
img = cv2.imread("hidden object.jpg", cv2.IMREAD_GRAYSCALE)  #以灰階的形式讀取

L = int(math.pow(2, 8))
hist_img = hist_unnormalized(img.ravel(), L)

img_size = img.shape

#周圍補零
img_3_oneArray = [0]*( (img_size[0]+2) * (img_size[1]+2))
img_3 = np.array(img_3_oneArray).reshape(img_size[0]+2, img_size[1]+2)

for i in range(img_size[0]):
    for j in range(img_size[1]):
        img_3[i+1][j+1] = img[i][j]

img_3_size = img_3.shape
#周圍補零



#Local enhancement
size_range = 1
size_part = (size_range*2) + 1
size_temp_img = [size_part]*2
print("size_temp_img : {}".format(size_temp_img))
img_2_oneArray = [0]*img_size[0]*img_size[1]
img_2 = np.array(img_2_oneArray).reshape(img_size[0], img_size[1])  #Local enhancement的img
for i in range(1, img_3_size[0]-1):  # i = 1 ~ 654
    for j in range(1, img_3_size[1]-1):  # j = 1 ~ 651
        temp_img_oneArray = part_of_img(img_3, size_range, i, j)  #取圖像的局部
        intensity_new_temp = intensities_new(temp_img_oneArray, L, size_temp_img)  #找新的強度
            
        temp_img_2_oneArray = [0]*size_part*size_part
        for k in range(len(temp_img_oneArray)):
            temp_img_2_oneArray[k] = intensity_new_temp[ temp_img_oneArray[k] ]

        temp_img_2 = np.array(temp_img_2_oneArray).reshape(size_part, size_part)
    
        img_2[i-1][j-1] = temp_img_2[size_range][size_range]
#Local enhancement

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap = 'gray')
plt.title("ORIGINAL")
plt.subplot(1, 2, 2)
plt.title("LOCAL_ENHANCE")
plt.imshow(img_2, cmap = 'gray')




#Histogram Statistics
m_G = intensities_mean(img.ravel(), L, img_size)
sigma_G = math.pow(intensities_standard(img.ravel(), L, img_size), 0.5)
print("m_G : {}, sigma_G : {}".format(m_G, sigma_G))


img_4_oneArray = [0]*img_size[0]*img_size[1]
img_4 = np.array(img_4_oneArray).reshape(img_size[0], img_size[1])  #statistic的method
print("img_size : {}".format(img_size))
print("img_3_size : {}".format(img_3_size))
C = 22.8
k_0 = 0; k_1 = 0.25; k_2 = 0; k_3 = 0.1
for i in range(1, img_3_size[0]-1):  # i = 1 ~ 654
    for j in range(1, img_3_size[1]-1):  # j = 1 ~ 651
        temp_img = part_of_img(img_3, size_range, i, j)
        m_xy = intensities_mean(temp_img, L, size_temp_img)
        sigma_xy = math.pow(intensities_standard(temp_img, L, size_temp_img), 0.5)

        #print("i : {}, j : {}".format(i, j))
        if(k_0*m_G <= m_xy and m_xy <= k_1*m_G):
            if(k_2*sigma_G <= sigma_xy and sigma_xy <= k_3*sigma_G):
                img_4[i-1][j-1] = C*img_3[i][j]

            else:
                img_4[i-1][j-1] = img_3[i][j]
            
        else:
            img_4[i-1][j-1] = img_3[i][j]


plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap = 'gray')
plt.title("ORIGINAL")
plt.subplot(1, 2, 2)
plt.title("HISTOGRAM_STATISTICS")
plt.imshow(img_4, cmap = 'gray', vmin = 0, vmax = 255)
#Histogram Statistics


plt.show()