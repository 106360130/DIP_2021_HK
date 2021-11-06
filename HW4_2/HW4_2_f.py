import numpy as np
import matplotlib.pyplot as plt
import pylab
from HW4_2_function import Butterworth_Lowpass
from HW4_2_function import show_img

#測試圖片 : "car-moire-pattern.tif"、"astronaut-interference.tif"
img = plt.imread("car-moire-pattern.tif")
print("img.shape : {}".format(img.shape))
show_img(np.abs(img), "Origin", "gray")

#FFT
img_ft = np.fft.fft2(img)
img_ft = np.fft.fftshift(img_ft)
#FFT

#noise location
array_NL = np.zeros((8, 2))  
array_NL[0][0] = 38; array_NL[0][1] = 110
array_NL[1][0] = 80; array_NL[1][1] = 113
array_NL[2][0] = 160; array_NL[2][1] = 113
array_NL[3][0] = 202; array_NL[3][1] = 113
array_NL[4][0] = 206; array_NL[4][1] = 56
array_NL[5][0] = 164; array_NL[5][1] = 57
array_NL[6][0] = 84; array_NL[6][1] = 54
array_NL[7][0] = 43; array_NL[7][1] = 54
#noise location

plt.figure()
plt.imshow(np.log(np.abs(img_ft)), cmap = "gray")
plt.title("FFT(shifted)")


#Butterworth
P = img.shape[0]
Q = img.shape[1]
Bw_lowpass = np.zeros((P, Q))
for i in range(8):

    m = int(array_NL[i][0])
    n = int(array_NL[i][1])
    D0 = 12
    N = 6
    temp_Bw_lowpass = Butterworth_Lowpass(P, Q, D0, N, m, n)
    Bw_lowpass = Bw_lowpass + temp_Bw_lowpass
#Butterworth

filter_Bw = 1 - Bw_lowpass
img_ft = img_ft * filter_Bw


plt.figure()
plt.imshow(np.log(np.abs(img_ft)), cmap = "gray")
plt.title("FFT(shifted) after filter")

img_result = np.fft.ifft2(img_ft)
show_img(np.abs(img_result), "RESULT", "gray")


#測試圖片 : "car-moire-pattern.tif"、"astronaut-interference.tif"
img_2 = plt.imread("astronaut-interference.tif")
print("img_2.shape : {}".format(img_2.shape))
show_img(np.abs(img_2), "Origin", "gray")

#FFT
img_2_ft = np.fft.fft2(img_2)
img_2_ft = np.fft.fftshift(img_2_ft)
#FFT

plt.figure()
plt.imshow(np.log(np.abs(img_2_ft)), cmap = "gray")
plt.title("FFT(shifted)")



# filter_reject = np.ones((img_2.shape[0], img_2.shape[1]))
# filter_reject[0 : 402, 499:501] = 0.00000001
# filter_reject[421 : img_2.shape[0], 499:501] = 0.00000001
# filter_reject[410 : 412, 0:490] = 0.00000001
# filter_reject[410 : 412, 510:img_2.shape[1]] = 0.00000001

# plt.figure()
# plt.imshow(filter_reject, "gray")
# plt.title("filter_reject")

#noise_location
array_NL_2 = np.zeros((2,2))
array_NL_2[0][0] = 387; array_NL_2[0][1] = 475
array_NL_2[1][0] = 437; array_NL_2[1][1] = 525
#noise_location

#Butterworth
P = img_2.shape[0]
Q = img_2.shape[1]
Bw_lowpass_2 = np.zeros((P, Q))
for i in range(2):

    m = int(array_NL_2[i][0])
    n = int(array_NL_2[i][1])
    D0 = 3
    N = 4
    temp_Bw_lowpass_2 = Butterworth_Lowpass(P, Q, D0, N, m, n)
    Bw_lowpass_2 = Bw_lowpass_2 + temp_Bw_lowpass_2

filter_Bw_2 = 1 - Bw_lowpass_2
# plt.figure()
# plt.imshow(filter_Bw_2, cmap = "gray")
# plt.title("filter_Bw_2")
#Butterworth

# P = img_2.shape[0]
# Q = img_2.shape[1]
# D0 = 12
# N = 1
# m = 411
# n = 500
# filter_test = Butterworth_Lowpass(P, Q, D0, N, m, n)
# plt.figure()
# plt.imshow(filter_test, cmap = "gray")
# plt.title("filter_test")


img_2_ft = img_2_ft*filter_Bw_2
#雜訊所在處
# img_2_ft[387][475] = 0
# img_2_ft[437][525] = 0
#雜訊所在處
plt.figure()
plt.imshow(np.log(np.abs(img_2_ft)), cmap = "gray")
plt.title("FFT(shifted) after filter")


img_2_result = np.fft.ifft2(img_2_ft)
show_img(np.abs(img_2_result), "RESULT", "gray")

plt.show()