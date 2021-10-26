
import numpy as np
import matplotlib.pyplot as plt
import cv2
"""


img = cv2.imread("hidden object.jpg")  #讀取影像
print(img)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("read_img", img)  #秀影像

cv2.waitKey(0)  #秀出來的影像視窗不要不見
#cv2.destroyAllWindows()


hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.bar(range(1,257), hist)
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖檔
img = cv2.imread('hidden object.jpg')

# 轉為灰階圖片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 計算直方圖每個 bin 的數值
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# 畫出直方圖
plt.bar(range(1,257), hist)
plt.show()
"""

"""
squares = [1, 4, 5, 6, 8, 7, 9, 3, 5, 8]
plt.bar(range(len(squares)), squares)
plt.show()


def input_array(array):
    array_new = [0]*len(array)

    for i in range(len(array)):
        array_new[i] = array[i] + 1
    return array_new

squares_new = input_array(squares)
print("squares_new : {}".format(squares_new))
"""

for i in range(2, 3):
    print("i : {}".format(i))

for i in range(2, 3):
    print("i : {}".format(i))

for i in range(3):
    print("i : {}".format(i))

X = [10]*123
print("type(X) : {}".format(type(X)))

Y = np.array(X)
print("type(Y) : {}".format(type(Y)))

Z = Y/2
print("type(Z) : {}".format(type(Z)))
print("Z : {}".format(Z))

I = Z.sum()
print("type(I) : {}".format(type(I)))
print("I : {}".format(I))

J = np.zeros((3,4))
print("type(J) : {}".format(type(J)))
print("J : {}".format(J))