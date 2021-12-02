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