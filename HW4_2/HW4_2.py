import numpy as np
import matplotlib.pyplot as plt
import pylab


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


img_ft = np.fft.fftshift(img_ft)

plt.figure()
plt.imshow(np.abs(img_ft), cmap = "gray")
plt.title("FFT")

plt.figure()
plt.imshow(np.log(np.abs(img_ft)), cmap = "gray")
plt.title("FFT(shifted)")

print("nnp.max(np.log(img_ft))) : {}".format(np.max(np.log(np.abs(img_ft)))))
print("np.min(np.log(img_ft))) : {}".format(np.min(np.abs(img_ft))))
#FFT

plt.show()