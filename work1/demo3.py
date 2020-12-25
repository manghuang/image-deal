"""
    完成对BMP格式文件的读取和显示
"""
from struct import unpack
import cv2

imgBMP = cv2.imread("./images/1.bmp")
print(imgBMP.shape)
cv2.namedWindow("imgBMP", 0)
cv2.resizeWindow("imgBMP", 768, 432)
cv2.imshow("imgBMP", imgBMP)
cv2.waitKey(0)
cv2.destroyAllWindows()
