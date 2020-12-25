"""
    实现基于Prewitt微分算子的图像边缘提取
"""


def Prewitt(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    s_suanziX = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # X方向
    s_suanziY = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    for i in range(r - 2):
        print(i)
        for j in range(c - 2):
            new_imageX[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziX))
            new_imageY[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziY))
            new_image[i + 1, j + 1] = (new_imageX[i + 1, j + 1] * new_imageX[i + 1, j + 1] + new_imageY[i + 1, j + 1] *
                                       new_imageY[i + 1, j + 1]) ** 0.5
    return np.uint8(new_image)  # 无方向算子处理的图像


import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('./1.jpg')
lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Prewitt算子
# kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
# kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
# x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
# y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
# absX = cv2.convertScaleAbs(x)
# absY = cv2.convertScaleAbs(y)
# Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
Prewitt = Prewitt(grayImage)

plt.rcParams['font.sans-serif'] = ['SimHei']
titles = [u'原始图像', u'Prewitt算子']
images = [lenna_img, Prewitt]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.savefig("./results/3.jpg")
plt.show()
