"""
Reference:
https://iter01.com/506332.html
https://blog.csdn.net/tymatlab/article/details/79009618
http://bmilab.ee.nsysu.edu.tw/lab/course/BII/Lecture01_Fundamentals.pdf
https://blog.gtwang.org/programming/opencv-basic-image-read-and-write-tutorial/
https://jennaweng0621.pixnet.net/blog/post/404217932-%5Bpython%5D-%E4%BD%BF%E7%94%A8numpy%E5%B0%87%E4%B8%80%E7%B6%AD%E8%BD%89%E4%BA%8C%E7%B6%AD%E6%88%96%E5%A4%9A%E7%B6%AD
https://blog.gtwang.org/programming/opencv-basic-image-read-and-write-tutorial/
https://blog.csdn.net/qq_23851075/article/details/51811177
https://www.dotblogs.com.tw/CYLcode/2020/03/12/113702
https://www.dotblogs.com.tw/YiruAtStudio/2021/03/13/153157
https://blog.csdn.net/GISuuser/article/details/98071333
https://easontseng.blogspot.com/2019/01/matplotlib-label.html
https://blog.csdn.net/sinat_35821976/article/details/84950697
https://sites.google.com/site/zsgititit/home/python-cheng-shi-she-ji/yi-wei-zhen-lie-yu-er-wei-zhen-lie-python
https://iter01.com/519791.html
https://codeinfo.space/imageprocessing/histogram-equalization/





"""


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
def hist_normalized(hist_img, L, img_size):
    hist_img_normalized = [0]*L

    for i in range(len(hist_img)):
        hist_img_normalized[i] = hist_img[i] / (img_size[0]*img_size[1])

    return hist_img_normalized

#找到新的圖案標準
def intensities_new(hist_img_normalized, L):
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

    print("intensity_new[L-1] : {}".format(intensity_new[L-1]))
    return intensity_new


def intensities_mean(hist_img_normalized):
    intensity_mean = 0
    for i in range(len(hist_img_normalized)):
        intensity_mean = intensity_mean + (i*hist_img_normalized[i])

    return intensity_mean

def intensities_standard(hist_img_normalized, intensity_mean):
    intensity_standard = 0
    for i in range(len(hist_img_normalized)):
        diff_temp = i - intensity_mean
        intensity_standard = intensity_standard + math.pow(diff_temp, 2)*hist_img_normalized[i]
    
    return intensity_standard


##########################################################

#HW3_1_1
img = cv2.imread("aerial_view.tif", cv2.IMREAD_GRAYSCALE)  #以灰階的形式讀取
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print("img : {}".format(img))
print("len(img.ravel()) : {}".format(len(img.ravel())))




#show圖片
"""
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#show圖片


L = int(math.pow(2, 8))
print("L : {}".format(L))

#秀hist_unnormalized結果
"""
plt.figure(1)
plt.hist(img.ravel(), 256, [0, 256])  #revel()是將多維轉到一維矩陣
plt.xlabel("r$_{k}$")
plt.ylabel("h(r$_{k}$)")
"""
#秀hist_unnormalized結果


print(img.ravel())

#計算hist_unnormalized
"""
hist_unnormalized = [0]*L
print("len(hist_unnormalized) : {}".format(len(hist_unnormalized)))
print("len(img.ravel()) : {}".format(len(img.ravel())))

for i in range(len(img.ravel())):
    hist_unnormalized[img.ravel()[i]] = hist_unnormalized[img.ravel()[i]] + 1
"""
print("img.ravel() : {}".format(img.ravel()))
hist_img = hist_unnormalized(img.ravel(), L)

#驗算
"""
total = 0
for i in range(L):
    total = total + hist_unnormalized[i]
    print("hist_unnormalized[{}] : {}".format(i, hist_unnormalized[i]))


print("total : {}".format(total))
"""
#驗算
#計算hist_unnormalized

#秀hist_unnormalized結果
"""
plt.figure()
plt.bar(range(len(hist_unnormalized)), hist_unnormalized)
plt.xlabel("r$_{k}$")
plt.ylabel("h(r$_{k}$)")
plt.show()
"""
plt.figure()
plt.bar(range(len(hist_img)), hist_img)
plt.xlabel("r$_{k}$")
plt.ylabel("h(r$_{k}$)")

#秀hist_unnormalized結果


#HW3_1_2
#計算hist_normalized
img_size = img.shape
print("img_size : {}".format(img_size))
"""
hist_normalized = [0]*L



for i in range(len(hist_normalized)):
    hist_normalized[i] = hist_unnormalized[i] / (img_size[0]*img_size[1])
"""

hist_normalized = hist_normalized(hist_img, L, img_size)

plt.figure()
plt.bar(range(len(hist_normalized)), hist_normalized)
plt.xlabel("r$_{k}$")
plt.ylabel("p(r$_{k}$)")
#計算hist_normalized


#計算intensities_new
"""
intensities_new = [0]*L
for i in range(L):
    if i == 0:
        intensities_new[i] = hist_normalized[i]

    else:   
        intensities_new[i] = intensities_new[i-1] + hist_normalized[i]

i = 125
print("intensities_new[{}] : {}".format(i, intensities_new[i]) )

for i in range(L):
    intensities_new[i] = round(intensities_new[i]*(L-1))
"""
rk_new = intensities_new(hist_normalized, L)
plt.figure()
plt.plot(range(len(rk_new)), range(L), range(len(rk_new)), rk_new)


"""
for i in range(100,125):
    print("intensities_new[{}] : {}".format(i, intensities_new[i]) )
"""
#計算intensities_new


#Histogram equalization
print(img.ravel()[1])
img_new = [0]*len(img.ravel())
for i in range(len(img.ravel())):
    img_new[i] = rk_new[ img.ravel()[i] ]

img_2 = np.array(img_new).reshape(img_size[0], img_size[1])
img_2_size = img_2.shape
print("img_new_2_size : {}".format(img_2_size))
#Histogram equalization

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap = 'gray')
plt.title("BEFORE")
plt.subplot(1, 2, 2)
plt.title("AFTER")
plt.imshow(img_2, cmap = 'gray')

hist_img_2 = hist_unnormalized(img_2.ravel(), L)

plt.figure()
plt.bar(range(len(hist_img_2)), hist_img_2)
plt.xlabel("r$_{k}$")
plt.ylabel("h(r$_{k}$)")


#HW3_1_3
P_z = [0]*L
c = 0.0005968
for i in range(L):
    P_z[i] = c*math.pow(i, 0.4)

plt.figure()
plt.bar(range(len(P_z)), P_z)
plt.xlabel("z$_{k}$")
plt.ylabel("P(r$_{k}$)")

rk_specificiation = intensities_new(hist_normalized, L)

#藉由S和G找Z，當S和G相差最少時，Z值為多少
rk_final = [0]*L
for i in range(L):
    diff_small = 255
    temp_rk = 0
    for j in range(L):
        diff = abs(rk_new[i] - rk_specificiation[j])
        if(diff < diff_small):
            diff_small = diff
            temp_rk = j

        if(diff_small == 0):  #"0"已經是最近，所以直接存
            break
        
        elif(j > temp_rk and diff >= diff_small):  #"diff"會呈現拋物線，所以如果"diff"開始有變大的趨勢就可以跳出
            break

    rk_final[i] = temp_rk
#藉由S和G找Z，當S和G相差最少時，Z值為多少




"""
plt.figure()
plt.bar(range(len(rk_specificiation)), rk_specificiation)
plt.xlabel("z$_{k}$")
plt.ylabel("P(r$_{k}$)")
"""

#Histogram equalization
print(img.ravel()[1])
img_new_2 = [0]*len(img.ravel())
for i in range(len(img.ravel())):
    img_new_2[i] = rk_final[ img.ravel()[i] ]

img_3 = np.array(img_new_2).reshape(img_size[0], img_size[1])
img_3_size = img_3.shape
print("img_new_2_size : {}".format(img_3_size))
#Histogram equalization

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap = 'gray')
plt.title("BEFORE")
plt.subplot(1, 2, 2)
plt.title("AFTER")
plt.imshow(img_2, cmap = 'gray')


hist_img_3 = hist_unnormalized(img_2.ravel(), L)

plt.figure()
plt.bar(range(len(hist_img_3)), hist_img_3)
plt.xlabel("r$_{k}$")
plt.ylabel("h(r$_{k}$)")

for i in range(L):
    print("hist_img_3[{}] : {}".format(i, hist_img_3[i]))





plt.show()











