"""
    实现图像的中值滤波平滑处理——二维中值滤波
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_noise_img():
    img = cv2.imread("./1.jpg", cv2.IMREAD_UNCHANGED)
    rows, cols, chn = img.shape
    # 加噪声
    for i in range(50000):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img[x, y, :] = 0
    # 保存
    cv2.imwrite("./results/2_1.jpg", img)


def midFiltering(img):
    height, width = img.shape
    result = np.zeros((height, width), np.uint8)
    list = []
    x = [-1, -1, -1, 0, 1, 1, 1, 0]
    y = [-1, 0, 1, 1, 1, 0, -1, -1]
    for i in range(height):
        print(i)
        for j in range(width):
            list.clear()
            mid = 1
            list.append(img[i, j])
            for index in range(8):
                a = i + x[index]
                b = j + y[index]
                if(0<=a and a<height and 0<=b and b<width):
                    list.append(img[a,b])
                    mid += 1
            list.sort()
            result[i, j] = list[(int)(mid/2)]
    # result = cv2.medianBlur(img, 3)
    return result


get_noise_img()
img = cv2.imread("./results/2_1.jpg",cv2.IMREAD_GRAYSCALE)
img2 = midFiltering(img)
plt.subplot(121)
plt.title("Image With Noise")
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.title("Image After Deal")
plt.imshow(img2, cmap='gray')
plt.savefig("./results/2_2.jpg")
plt.show()
