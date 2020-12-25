"""
    完成对jpg、RAW、BMP格式文件的相互转换
"""
import cv2

# JPG -> BMP
imgJPG = cv2.imread("./images/1.jpg", cv2.IMREAD_GRAYSCALE)
# imgJPG = cv2.imread("./images/1.jpg")  # 灰度图像才可以转换成功
print(imgJPG.shape)
cv2.imwrite("./results/1.bmp", imgJPG)


# BMP ->JPG
imgBMP = cv2.imread("./images/1.bmp")
print(imgBMP.shape)
cv2.imwrite("./results/1.jpg", imgBMP)