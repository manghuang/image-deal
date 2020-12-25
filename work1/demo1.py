"""
    完成对jpg格式文件的读取和显示
"""
import cv2

imgJPG = cv2.imread("./images/1.jpg")
print(imgJPG.shape)
cv2.namedWindow("imgJPG", 0)
cv2.resizeWindow("imgJPG", 768, 432)
cv2.imshow('imgJPG', imgJPG)
cv2.waitKey(0)
cv2.destroyAllWindows()
