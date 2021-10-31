from __future__ import division             # forces floating point division 
import numpy as np                          # Numerical Python 
import matplotlib.pyplot as plt             # Python plotting
from PIL import Image                       # Python Imaging Library
from numpy.fft import fft2, fftshift, ifft2 # Python DFT

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

