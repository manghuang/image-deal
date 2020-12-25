"""
    实现基于Sobel微分算子的图像边缘提取
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Sobel(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    s_suanziX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # X方向
    s_suanziY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    for i in range(r - 2):
        print(i)
        for j in range(c - 2):
            new_imageX[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziX))
            new_imageY[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziY))
            new_image[i + 1, j + 1] = (new_imageX[i + 1, j + 1] * new_imageX[i + 1, j + 1] + new_imageY[i + 1, j + 1] *
                               new_imageY[i + 1, j + 1]) ** 0.5
    # return np.uint8(new_imageX)
    # return np.uint8(new_imageY)
    return np.uint8(new_image)  # 无方向算子处理的图像


img = cv2.imread('./1.jpg')
lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel算子
# Sobel_x = cv2.Sobel(grayImage, cv2.CV_16S, 1,0)
# Sobel_x = cv2.convertScaleAbs(Sobel_x)
# Sobel_y = cv2.Sobel(grayImage, cv2.CV_16S, 0,1)
# Sobel_y = cv2.convertScaleAbs(Sobel_y)
# result = cv2.addWeighted(Sobel_x, 0.5, Sobel_y, 0.5, 0)
result = Sobel(grayImage)

plt.rcParams['font.sans-serif'] = ['SimHei']
titles = [u'原始图像', u'Sobel算子']
images = [lenna_img, result]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.savefig("./results/2.jpg")
plt.show()