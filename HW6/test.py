"""
import numpy as np

test = np.array([1, 2, 5, 6])
print("test : {}".format(test))

test = test - np.min(test)
test = (test / np.max(test)) * 100
print("test : {}".format(test))


test = np.array([1, 2, 5, 6])
test = test ** 2
test = test - np.min(test)
test = (test / np.max(test)) * 100
print("test : {}".format(test))
"""

"""
from PIL import Image
from PIL import ImageSequence
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("Visual resolution.gif")
print("type(img) : {}".format(type(img)))

i = 0
for frame in ImageSequence.Iterator(img):
    img_np = np.array(frame)
    i = i + 1


print("i : {}".format(i))

print("img.size : {}".format(img.size))
print("img : {}".format(img))

# img_np = np.array(img)
print("img_np : {}".format(img_np))
print("img_np.shape : {}".format(img_np.shape))
plt.figure()
plt.imshow(img_np, cmap = None, vmin = 0, vmax = 255)
plt.title("ORIGIN")

# for i in range(img.size[0]):
#     for j in range(img.size[1]):
#         print(img[][]])


plt.show()
"""
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt

frames=imageio.mimread('Visual resolution.gif') #frames是一個列表，存儲gif裏面的每一幀，長度就是幀個數
print(len(frames)) #imageio的圖片類型爲imageio.core.util.Array
cv_img=[]
for f in frames: #把每一幀轉爲numpy
	i = np.array(f)
	cv_img.append(i)

for c in cv_img:#顯示
    plt.figure()
    plt.imshow(c, cmap = None, vmin = 0, vmax = 255)
    plt.title("ORIGIN")
    print("c.shape : {}".format(c.shape))


plt.show()
