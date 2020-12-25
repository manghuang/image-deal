"""
    使用傅里叶描述子描述边界
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt


def dft(a):
    N = a.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * n * k / N) / N  # 变换矩阵
    result = np.dot(M, a)
    return result

def idft(a):
    N = a.shape[0]
    temp = 2
    k = np.arange(N)
    k = k.reshape((N, 1))
    n = np.arange(temp)
    M = np.exp(2j * np.pi * n * k / N)  # 反变换矩阵
    result = np.dot(M, a[:temp])
    return result

# 定义边界点的个数
N = 64
length = 17
img = np.ones([length, length], dtype=np.uint8)

s = []
for i in range(length):
    for j in range(length):
        if i == 0 or i == length-1 or j == 0 or j == length-1:
            img[i,j] = 0
            s.append(complex(i,j))
s = np.asarray(s)
# print(s.shape)
dft = dft(s)
# print(dft.shape)
a = idft(dft)
# print(a.shape)
img_back = np.ones([length, length], dtype=np.uint8)
for i in range(a.shape[0]):
    real = a[i].real
    imag = a[i].imag
    real = int(round(real))
    imag = int(round(imag))
    real = real if real<length else length-1
    imag = imag if imag<length else length-1
    img_back[real, imag] = 0

# print(img_back)
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Input Image")
plt.subplot(1,2,2)
plt.imshow(img_back)
plt.title("Image After Deal")
plt.savefig('./results/1.jpg')
plt.show()

