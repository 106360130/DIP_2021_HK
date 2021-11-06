"""
Reference :
將plt.show()關閉
https://www.coder.work/article/4933595

"""
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

# plt.figure()
# plt.imshow(np.log(np.abs(img_ft)), cmap = "gray")
# plt.title("FFT")


img_ft = np.fft.fftshift(img_ft)
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




# for i in range(8):
#     temp_x = int(array_NL[i][0])
#     temp_y = int(array_NL[i][1])
#     print("img_ft[{}, {}] : {}".format(temp_x, temp_y, img_ft[temp_x, temp_y]))
#     print("np.abs(img_ft[{}, {}]) : {}".format(temp_x, temp_y, np.abs(img_ft[temp_x, temp_y])))
#     print("np.log(abs(img_ft[{}, {}])) : {}".format(temp_x, temp_y, np.log(np.abs(img_ft[temp_x, temp_y]))))


"""
temp_x = int(array_NL[0][0])
temp_y = int(array_NL[0][1])
print("img_ft[{}, {}] : {}".format(temp_x, temp_y, img_ft[temp_x, temp_y]))
print("np.abs(img_ft[{}, {}]) : {}".format(temp_x, temp_y, np.abs(img_ft[temp_x, temp_y])))
print("np.log(abs(img_ft[{}, {}])) : {}".format(temp_x, temp_y, np.log(np.abs(img_ft[temp_x, temp_y]))))
img_ft[temp_x, temp_y] = 0
print("img_ft[{}, {}] : {}".format(temp_x, temp_y, img_ft[temp_x, temp_y]))
print("np.abs(img_ft[{}, {}]) : {}".format(temp_x, temp_y, np.abs(img_ft[temp_x, temp_y])))
print("np.log(abs(img_ft[{}, {}])) : {}".format(temp_x, temp_y, np.log(np.abs(img_ft[temp_x, temp_y]))))

"""


# plt.figure()
# plt.imshow(np.abs(img_ft), cmap = "gray")
# plt.title("FFT(shifted)")

plt.figure()
plt.imshow(np.log(np.abs(img_ft)), cmap = "gray")
plt.title("FFT(shifted)")





# print("nnp.max(np.log(img_ft))) : {}".format(np.max(np.log(np.abs(img_ft)))))
# print("np.min(np.log(img_ft))) : {}".format(np.min(np.abs(img_ft))))
#FFT

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
# plt.figure()
# plt.imshow(Bw_lowpass, "gray")
# plt.title("Butterworth Lowpass")

Bw_highpass = 1 - Bw_lowpass
# plt.figure()
# plt.imshow(Bw_highpass, "gray")
# plt.title("Butterworth Highpass")

img_ft = img_ft * Bw_highpass


# plt.figure()
# plt.imshow(np.abs(img_ft), cmap = "gray")
# plt.title("FFT(shifted) after highpass")

plt.figure()
plt.imshow(np.log(np.abs(img_ft)), cmap = "gray")
plt.title("FFT(shifted) after highpass")




"""
test = np.zeros((P, Q))
for i in range(20, 40):
    for j in range(30, 50):
        test[i][j] = 1

test = 1 - test
plt.figure()
plt.imshow(test, "gray")
plt.title("test")

img_ft = img_ft * test
plt.figure()
plt.imshow(np.log(np.abs(img_ft)), cmap = "gray")
plt.title("FFT(shifted) after test")

img_back = np.fft.ifft2(img_ft)
plt.figure()
plt.imshow(np.abs(img_back), cmap = "gray")
plt.title("test")

for i in range(20, 40):
    for j in range(30, 50):
        print("img_ft[{}][{}] : {}".format(i, j, img_ft[i][j]))

"""

img_back = np.fft.ifft2(img_ft)
show_img(np.abs(img_back), "RESULT", "gray")


plt.show()

#圖片秀60秒後關閉
"""
plt.show(block=False)
plt.pause(60) # 60 seconds, I use 1 usually
plt.close("all")
"""
#圖片秀60秒後關閉