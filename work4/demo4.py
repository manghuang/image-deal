"""
    实现基于拉普拉斯算子微分算子的图像边缘提取
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def Laplacian(img, K_size=3):
    height, width = img.shape
    out = np.zeros(img.shape)
    K = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    # K = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    for i in range(height-2):
        print(i)
        for j in range(width-2):
            out[i, j] =abs(np.sum(K * (img[i:i + K_size, j:j + K_size])))
    return np.uint8(out)


img = cv2.imread('./1.jpg')
lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 拉普拉斯算法
# Laplacian = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)
# Laplacian = cv2.convertScaleAbs(Laplacian)
Laplacian = Laplacian(grayImage)

plt.rcParams['font.sans-serif'] = ['SimHei']
titles = [u'原始图像', u'Laplacian算子']
images = [lenna_img, Laplacian]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.savefig("./results/4.jpg")
plt.show()
