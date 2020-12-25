"""
    实现基于Roberts微分算子的图像边缘提取
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Roberts(img):
    r, c = img.shape
    R = [[-1, -1], [1, 1]]
    new_image = np.zeros(img.shape)
    for x in range(r-2):
        print(x)
        for y in range(c-2):
            imgChild = img[x:x + 2, y:y + 2]
            list_robert = R * imgChild
            new_image[x, y] = abs(list_robert.sum())  # 求和加绝对值
    return np.uint8(new_image)


img = cv2.imread("./1.jpg")
RGBImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
GRAYImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Roberts微分算子
# kernelx = np.array([[-1,0],[0,1]], dtype=int)
# kernely = np.array([[0,-1],[1,0]], dtype=int)
# x = cv2.filter2D(GRAYImage, cv2.CV_16S, kernelx)
# y = cv2.filter2D(GRAYImage, cv2.CV_16S, kernely)
# absX = cv2.convertScaleAbs(x)
# absY = cv2.convertScaleAbs(y)
# result = cv2.addWeighted(absX,0.5,absY,0.5,0)

result = Roberts(GRAYImage)

plt.rcParams['font.sans-serif'] = ['SimHei']
titles = [u'原始图像', u'Roberts算子']
images = [RGBImage, result]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.savefig("./results/1.jpg")
plt.show()
