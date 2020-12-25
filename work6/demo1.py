"""
    实现KNN算法，对图像进行分类处理
"""
from copy import deepcopy
import cv2
import numpy as np
from matplotlib import pyplot as plt


def k_means(x, k=3):
    index_list = np.arange(len(x))
    np.random.shuffle(index_list)
    centroids_index = index_list[:k]
    centroids = x[centroids_index]
    y = np.arange(len(x))
    iter_num = 0
    while True:
        isok = 0
        iter_num += 1
        print(iter_num)
        y_new = np.arange(len(x))
        for i, xi in enumerate(x):
            y_new[i]= np.argmin([sum(abs(xi - ci)) for ci in centroids])
            if y[i] != y_new[i]:
                isok += 1
        for j in range(k):
            centroids[j] = np.mean(x[np.where(y_new == j)], axis=0)
        y = y_new.copy()
        print(isok)
        if isok == 0:
            break
    return y


img = cv2.imread("./1.png")
img_int16 = np.asarray(img, dtype=np.int16)
h, w, _ = img_int16.shape
img_int16 = img_int16.reshape((h * w, 3))
result = k_means(img_int16)
result = result.reshape((h, w))
print(result)
img0 = deepcopy(img)
img1 = deepcopy(img)
img2 = deepcopy(img)

for i in range(h):
    for j in range(w):
        if result[i][j] == 0:
            img0[i][j] = [255, 255, 255]
        elif result[i][j] == 1:
            img1[i][j] = [255, 255, 255]
        else:
            img2[i][j] = [255, 255, 255]

plt.rcParams['font.sans-serif'] = ['SimHei']
titles = [u'Input Image', u'类别0', u'类别1', u'类别2']
images = [img, img0,img1,img2]
for i in range(4):
    plt.subplot(1, 4, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.savefig("./results/1.jpg")
plt.show()
