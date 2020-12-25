"""
    完成对CR2/RAW格式文件的读取和显示
"""
import cv2
import rawpy

with rawpy.imread("./images/1.CR2") as raw:
    rgb = raw.postprocess(output_color=rawpy.ColorSpace.Adobe)
    print(rgb.shape)
    cv2.namedWindow("imgRAW", 0)
    cv2.resizeWindow("imgRAW", 432, 668)
    cv2.imshow('imgRAW', rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
