"""
    实现图像灰度的指数变换增强
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像
img = cv2.imread('./1.jpg')

# 图像灰度转换
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(grayImage.shape)
height = grayImage.shape[0]
width = grayImage.shape[1]

# 创建一幅图像
result = np.zeros((height, width), np.uint8)

# 指数变换
for i in range(height):
    for j in range(width):
        # gray = np.exp(grayImage[i, j] / 255) / np.e * 255
        gray = np.math.pow(grayImage[i, j], 0.5)
        result[i, j] = np.uint8(gray)

plt.subplot(121)
plt.title("Input Image")
plt.imshow(grayImage, cmap='gray')
plt.subplot(122)
plt.title("Image After Deal")
plt.imshow(result, cmap='gray')
plt.savefig("./results/3.jpg")
plt.show()

