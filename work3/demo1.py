"""
    自己实现直方图均衡化
"""
import cv2
from numpy import histogram, interp
from matplotlib import pyplot as plt


# 直方图均衡化
def histeq(img, nbr_bins=256):
    # 获取直方图p(r)
    imhist, bins = histogram(img.flatten(), nbr_bins)
    print(imhist)
    print(bins)
    # 获取T(r)
    cdf = imhist.cumsum()
    # print(cdf)
    cdf = 255 * cdf / cdf[-1]
    # print(cdf)
    # 获取s，并用s替换原始图像对应的灰度值
    result = interp(img.flatten(), bins[:-1], cdf)
    return result.reshape(img.shape)


# 测试
image1 = cv2.imread("./1.jpg", cv2.IMREAD_GRAYSCALE)
plt.subplot(121)
plt.title("Input Image")
plt.imshow(image1, cmap='gray')
image2 = histeq(image1)
plt.subplot(122)
plt.title("Image After Histeq")
plt.imshow(image2, cmap='gray')
plt.savefig("./results/1.jpg")
plt.show()
