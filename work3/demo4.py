"""
    实现图像的Laplace锐化处理
"""

# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


def laplacian_sharping(img, K_size=3):
    height, width = img.shape
    pad = K_size // 2
    out = np.zeros((height + pad * 2, width + pad * 2), dtype=np.float)
    out[pad:pad + height, pad:pad + width] = img.copy().astype(np.float)
    temp = out.copy()
    K = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    for i in range(height):
        for j in range(width):
            out[pad + i, pad + j] =abs(np.sum(K * (temp[i:i + K_size, j:j + K_size]))+ temp[pad + i, pad + j])
    out = out[pad: pad + height, pad:pad + width].astype(np.uint8)
    return out


img = cv2.imread('./1.jpg')
lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 拉普拉斯算法
# Laplacian = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)
# Laplacian = cv2.convertScaleAbs(Laplacian)
# Laplacian = Laplacian + grayImage

Laplacian = laplacian_sharping(grayImage)
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 显示图形
titles = [u'原始图像', u'Laplacian算子']
images = [lenna_img, Laplacian]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.savefig("./results/4.jpg")
plt.show()
