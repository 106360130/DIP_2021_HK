"""
fft
https://www.delftstack.com/zh-tw/howto/python/fft-example-in-python/
"""

import math
import matplotlib.pyplot as plt
import numpy as np

f_x = np.array([4, 2, 1, 5])
F_u = np.fft.fft(f_x)
print("F_u : {}".format(F_u))

f_x = np.array([4, 2, 1, 5, 4, 2, 1, 5])
F_u = np.fft.fft(f_x)
print("F_u : {}".format(F_u))

f_x = np.array([4, -2, 1, -5, 4, -2, 1, -5])
F_u = np.fft.fft(f_x)
print("F_u : {}".format(F_u))



"""
test_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
test_2 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
print("test_1 : {}".format(test_1))
print("test_2 : {}".format(test_2))
test_1 = test_1*test_2
print("test_1 : {}".format(test_1))
print("test_2 : {}".format(test_2))
test_1[:,1] = 1
print("test_1 : {}".format(test_1))

test_3 = np.zeros((1,10))
print("test_3 : {}".format(test_3))
test_3[:,2:5] = 1
print("test_3 : {}".format(test_3))
"""


"""
m = 20; n = 30  #要濾掉的區域的中心點
P = 50; Q = 50  #整個fiter的size


D0 = 10  #亮暗範圍
N = 2  #平滑程度

Bw_lowpass = np.zeros((P, Q))
a = math.pow(D0, (2 * N)) 

for u in range(P) :
    for v in range(Q) :
        temp = math.pow((u-(m+1.0)), 2) + math.pow((v-(n+1.0)), 2)
        Bw_lowpass[u][v] = 1 / (1 + math.pow(temp, N) / a)


plt.figure()
plt.imshow(Bw_lowpass, "gray")
plt.title("Butterworth Lowpass")
plt.show()

Bw_highpass = 1 - Bw_lowpass


plt.figure()
plt.imshow(Bw_highpass, "gray")
plt.title("Butterworth Highpass")
plt.show()
"""


"""
from __future__ import division             # forces floating point division 
import numpy as np                          # Numerical Python 
import matplotlib.pyplot as plt             # Python plotting
from PIL import Image                       # Python Imaging Library
from numpy.fft import fft2, fftshift, ifft2 # Python DFT
def butterLow(cutoff, critical, order):
    normal_cutoff = float(cutoff) / critical
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butterFilter(data, cutoff_freq, nyq_freq, order):
    b, a = butterLow(cutoff_freq, nyq_freq, order)
    y = signal.filtfilt(b, a, data)
    return y

#測試圖片 : "car-moire-pattern.tif"、"astronaut-interference.tif"
img = plt.imread("car-moire-pattern.tif")
print("img.shape : {}".format(img.shape))
plt.figure()
plt.imshow(img, cmap = "gray")
plt.title("Origin")



#FFT
img_ft = np.fft.fft2(img)
plt.figure()
plt.imshow(np.log(np.abs(img_ft)), cmap = "gray")
plt.title("FFT")

x=np.array(img_ft)
cutoff_frequency = some value
sample_rate = maximum value in your array *2 +1

y = butterFilter(x, cutoff_frequency, sample_rate/2)
"""

"""
# # Show plots in the notebook (don't use it in Python scripts)
# %matplotlib inline .


hW, hH = 600, 300
hFreq = 10.5

# Mesh on the square [0,1)x[0,1)
x = np.linspace( 0, 2*hW/(2*hW +1), 2*hW+1)     # columns (Width)
y = np.linspace( 0, 2*hH/(2*hH +1), 2*hH+1)     # rows (Height)

[X,Y] = np.meshgrid(x,y)
A = np.sin(hFreq*2*np.pi*X)

plt.imshow(A, cmap = 'gray');
H,W = np.shape(A)



F = fft2(A)/(W*H)                          
F = fftshift(F)
P = np.abs(F)                            
plt.imshow(P, extent = [-hW,hW,-hH,hH]);



plt.imshow(P[hH-25:hH+25,hW-25:hW+25], extent=[-25,25,-25,25]);



hFreq = 10.5
vFreq = 20.5

A1 = np.sin(hFreq*2*np.pi*X) + np.sin(vFreq*2*np.pi*Y)

plt.figure()
plt.imshow(A1, cmap = 'gray');

F1 = fft2(A1)/(W*H)                          
F1 = fftshift(F1)
P1 = np.abs(F1)

plt.figure()
plt.imshow(P1[hH-25:hH+25,hW-25:hW+25], extent=[-25,25,-25,25]);



hFreq = 10.5
vFreq = 20.5

A2 = np.sin(hFreq*2*np.pi*X + vFreq*2*np.pi*Y)

plt.figure()
plt.imshow(A2, cmap = 'gray');

F2 = fft2(A2)/(W*H)                          
F2 = fftshift(F2)
P2 = np.abs(F2)

plt.figure()
plt.imshow(P2[hH-25:hH+25,hW-25:hW+25], extent=[-25,25,-25,25]);




I = Image.open("car-moire-pattern.tif")
I = I.convert('L')                     # 'L' for gray scale mode
A3 = np.asarray(I, dtype = np.float32)  # Image class instance, I1, to float32 Numpy array, a

H,W = np.shape(A3)
hW = np.fix(0.5*W)
hH = np.fix(0.5*H)

plt.imshow(A3, cmap = 'gray');




F3 = fft2(A3)/(W*H)                          
F3 = fftshift(F3)
P3 = np.abs(F3)


plt.figure()
plt.imshow(P3)
plt.show()
"""

"""
r = 10
plt.figure()
plt.imshow(np.log(1+P3[hH-r:hH+r,hW-r:hW+r]), extent=[-r,r,-r,r]);




points = 100
Trange = np.linspace(0.05,1,points)
g = np.zeros(points, dtype = 'float32')
n = 0

for T in Trange:
    g[n] = np.count_nonzero(P3 >= T)
    n += 1   
    
plt.plot(g);    




T = 0.1
c = F3 * (P3 >= T)
fM = ifft2(c)*W*H
plt.imshow(np.abs(fM), cmap = 'gray');



out1 = np.count_nonzero(F3)
out2 = np.count_nonzero(c)
print(out1/out2)
"""

