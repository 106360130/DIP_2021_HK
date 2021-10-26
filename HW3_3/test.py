#方法一，利用關鍵字
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定義座標軸
fig = plt.figure()
ax1 = plt.axes(projection='3d')
#ax = fig.add_subplot(111,projection='3d')  #這種方法也可以畫多個子圖

import numpy as np
z = np.linspace(0,13,1000)
x = 5*np.sin(z)
y = 5*np.cos(z)
zd = 13*np.random.random(100)
xd = 5*np.sin(zd)
yd = 5*np.cos(zd)
ax1.scatter3D(xd,yd,zd, cmap='Blues')  #繪製散點圖
ax1.plot3D(x,y,z,'gray')    #繪製空間曲線
plt.show()


m_half = 2
X = list(range(-m_half, m_half+1))
print("X : {}".format(X))



for i in range(-2,2+1):
    print("i : {}".format(i))


X = np.arange(0, 0, 10)
print("X : {}".format(X))

matrix1 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
print("matrix1.sum() : ".format(matrix1.sum()))


