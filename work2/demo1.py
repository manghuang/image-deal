"""
    numpy包中的二维fft变换和反变换
"""
import numpy as np
from matplotlib import pyplot as plt

img = np.zeros([128, 128],dtype=np.uint8)
# img[62: 66, 62: 66] = 1   # 4*4
# img[30: 34, 30: 34] = 1  # 4*4
# img[58: 70, 58: 70] = 1  # 12*12
img[63: 65, 63: 65] = 1  # 2*2
print(img)
plt.subplot(141)
plt.imshow(img, cmap='gray')
plt.title('Input Image')

fft = np.fft.fft2(img)  # fft变换
plt.subplot(142)
plt.imshow(abs(fft), cmap='gray')
plt.title('Image 1 After FFT')

f_shift = np.fft.fftshift(fft)  # 将直流分量移动到频谱的中央
plt.subplot(143)
plt.imshow(abs(f_shift), cmap='gray')
plt.title('Image 2 After FFT')

magnitude_spectrum  = 20 * np.log(1 + np.abs(f_shift))  # 动态增强
plt.subplot(144)
plt.imshow(magnitude_spectrum , cmap='gray')
plt.title('Image 3 After FFT')
plt.savefig("./results/4.png")
plt.show()


# f_ishift = np.fft.ifftshift(f_shift)
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.abs(img_back)
# plt.subplot(121)
# plt.imshow(img, cmap='gray')
# plt.title('Input Image')
#
# plt.subplot(122)
# plt.imshow(img_back, cmap='gray')
# plt.title('Image After IFFT')
# plt.show()
