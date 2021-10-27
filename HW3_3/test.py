"""
matrix中找最大值及最小值
https://blog.csdn.net/qq_18293213/article/details/70175920

讀取圖片用plt.imread
https://vimsky.com/zh-tw/examples/detail/python-method-matplotlib.pyplot.imread.html


取出圖片三通道其中一通道
https://blog.csdn.net/mokeding/article/details/17618861

圖像bgr分開與合併
https://blog.csdn.net/qwowowowowo/article/details/83542013
https://blog.csdn.net/qq_28618765/article/details/78082438
https://blog.csdn.net/mokeding/article/details/17618861



"""



#方法一，利用關鍵字
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from HW3_3_function import show_img_gray
import cv2

"""
#定義座標軸
fig = plt.figure()
ax1 = plt.axes(projection='3d')
#ax = fig.add_subplot(111,projection='3d')  #這種方法也可以畫多個子圖


z = np.linspace(0,13,1000)
x = 5*np.sin(z)
y = 5*np.cos(z)
zd = 13*np.random.random(100)
xd = 5*np.sin(zd)
yd = 5*np.cos(zd)
ax1.scatter3D(xd,yd,zd, cmap='Blues')  #繪製散點圖
ax1.plot3D(x,y,z,'gray')    #繪製空間曲線
plt.show()
"""


"""
m_half = 2
X = list(range(-m_half, m_half+1))
print("X : {}".format(X))



for i in range(-2,2+1):
    print("i : {}".format(i))


X = np.arange(0, 0, 10)
print("X : {}".format(X))

matrix1 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
matrix1_np = np.array(matrix1)
print("matrix1_np.sum() : {}".format(matrix1_np.sum()))

list_random = [[8, 2, 5], [8, 6, 2], [5, 21, 0]]
matrix = np.array(list_random)
print("matrix : {}".format(matrix))
# print("np.min(matrix) : {}". format(np.min(matrix)))
# print("np.max(matrix) : {}". format(np.max(matrix)))

list_random2 = [[5, 8, 9], [4, 5, 3], [1, 3, 5]]
matrix_2 = np.array(list_random2)
print("matrix_2 : {}".format(matrix_2))

matrix_3 = matrix - matrix_2
print("matrix_3 : {}".format(matrix_3))

matrix_3 = matrix / matrix_2
print("matrix_3 : {}".format(matrix_3))
"""

#"checkerboard1024-shaded.tif"  二通道
#"N1.bmp"  三通道
img = plt.imread("N1.bmp")
print("type(img) : {}".format(type(img)))
print("img.shape : {}".format(img.shape))
print("type(img.shape) : {}".format(type(img.shape)))
print("len(img.shape) : {}".format(len(img.shape)))

plt.figure()
plt.imshow(img)
plt.title("N1.bmp")


print("type(img.shape[0]) : {}".format(type(img.shape[0])))  #
if(len(img.shape) == 3) :
    #b = np.zeros(img.shape[0], img.shape[1])  #錯誤的
    b = np.zeros((img.shape[0], img.shape[1]))  #正確的
    g = np.zeros((img.shape[0], img.shape[1]))
    r = np.zeros((img.shape[0], img.shape[1]))
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    print("r : {}".format(r))
    print("g : {}".format(g))
    print("b : {}".format(b))

    X = g - b
    print("X.sum() : {}".format(X.sum()))



    plt.figure()
    plt.imshow(r, cmap = 'Reds')
    plt.title("r")



    plt.figure()
    plt.imshow(g, cmap = 'Greens')
    plt.title("g")


    plt.figure()
    plt.imshow(b, cmap = 'Blues')
    plt.title("b")

    img_final = cv2.merge([b, g, r])

    plt.figure()
    plt.imshow(img_final, cmap = 'gray')
    plt.title("FINAL")
  

"""
img_2 = cv2.imread('checkerboard1024-shaded.tif')
print("type(img_2) : {}".format(type(img_2)))
print("img_2.shape : {}".format(img_2.shape))
"""
plt.show()
