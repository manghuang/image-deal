"""
完成图像的基本操作
图像的基本操作：
    1、图像的算数运算
        1、加法
        2、减法
        3、乘法
        4、除法
    2、图像逻辑运算
        1、求反
        2、异或
        3、或
        4、与
        5、形态学图像处理（基于图像的逻辑运算）
            1、膨胀
            2、腐蚀
            3、开操作
            4、闭操作
    3、图像几何变换
        1、基本变换
            1、平移变换
            2、旋转变换
            3、镜像变换
            4、放缩变换
            5、拉伸变换
            6、离散变换..
        2、灰度级插值..
            1、最邻近插值法
            2、一阶插值法
            3、高阶插值法
    4、图像非几何变换
        1、模板运算..
        2、灰度级变换..
            1、图像求反
            2、对比度拉伸
            3、动态范围压缩
            4、灰度级切片
        3、直方图变换
            1、直方图均衡化
            2、直方图匹配..
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


# 加载
def loadImage(path):
    img = cv2.imread(path)
    print(img.shape)
    return img


# 显示
def showImage(img):
    cv2.namedWindow("img", 0)
    cv2.resizeWindow("img", 768, 432)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 平移变换
def pallMoveImage(img):
    h, w, _ = img.shape
    M = np.float32([[1, 0, 500], [0, 1, 500]])
    img = cv2.warpAffine(img, M, (w, h))
    print(img.shape)
    return img


# 放缩变换
def resizeImage1(img):
    img = cv2.resize(img, (1920, 1080))
    print(img.shape)
    return img


def resizeImage2(img):
    h, w, _ = img.shape
    M = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
    img = cv2.warpAffine(img, M, (w, h))
    print(img.shape)
    return img


# 旋转变换
def rotationImage(img):
    h, w, _ = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 0.6)
    img = cv2.warpAffine(img, M, (w, h))
    print(img.shape)
    return img


# 镜像变换
def flipImage(img):
    img = cv2.flip(img, 0)
    print(img.shape)
    return img


# 拉伸变换
def pollImage(img):
    h, w, _ = img.shape
    mat_src = np.float32([[0, 0], [0, h - 1], [w - 1, 0]])
    mat_dst = np.float32([[50, 50], [300, h - 200], [w - 300, 100]])
    mat_Affine = cv2.getAffineTransform(mat_src, mat_dst)
    img = cv2.warpAffine(img, mat_Affine, (h, w))
    print(img.shape)
    return img


# 改变色彩空间
def cvtColorImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(img.shape)
    return img


# 求反
def bitwise_notImage(img):
    img = cv2.bitwise_not(img)
    print(img.shape)
    return img


# 与
def bitwise_andImage(img1, img2):
    img = cv2.bitwise_and(img1, img2)
    print(img.shape)
    return img


# 或
def bitwise_orImage(img1, img2):
    img = cv2.bitwise_or(img1, img2)
    print(img.shape)
    return img


# 异或
def bitwise_xorImage(img1, img2):
    img = cv2.bitwise_xor(img1, img2)
    print(img.shape)
    return img


# 膨胀
def dilateImage(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.dilate(img, kernel)
    print(img.shape)
    return img


# 腐蚀
def erodeImage(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.erode(img, kernel)
    print(img.shape)
    return img


# 开操作
def openImage(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    print(img.shape)
    return img


# 闭操作
def closeImage(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    print(img.shape)
    return img


# 加法
def addImage(img1, img2):
    img = cv2.add(img1, img2)
    print(img.shape)
    return img


# 混合
def addWeightedImage(img1, img2):
    img = cv2.addWeighted(img1, 0.7, img2, 0.2, 0)
    print(img.shape)
    return img


# 减法
def subImage(img1, img2):
    img = cv2.subtract(img1, img2)
    print(img.shape)
    return img


# 乘法
def multImage(img1, img2):
    img = cv2.multiply(img1, img2)
    print(img.shape)
    return img


# 除法
def divImage(img1, img2):
    img = cv2.divide(img1, img2)
    print(img.shape)
    return img


# 直方图和直方图均衡化
def equalizeHistImage(img):
    chans = cv2.split(img)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()
    img = cvtColorImage(img)
    img = cv2.equalizeHist(img)
    print(img.shape)
    return img


# 测试
image1 = loadImage("./images/1.jpg")
showImage(image1)
image1 = equalizeHistImage(image1)
showImage(image1)
