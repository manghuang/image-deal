"""
    自己实现的二维fft变换和反变换
"""
import numpy as np
from matplotlib import pyplot as plt


def DFT(sig):
    N = sig.size
    V = np.array([[np.exp(-1j * 2 * np.pi * v * y / N) for v in range(N)] for y in range(N)])
    return sig.dot(V)


def FFT(x):
    N = x.shape[1]  # 只需考虑第二个维度，然后在第一个维度循环
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 8:  # this cutoff should be optimized
        return np.array([DFT(x[i, :]) for i in range(x.shape[0])])
    else:
        X_even = FFT(x[:, ::2])
        X_odd = FFT(x[:, 1::2])
        factor = np.array([np.exp(-2j * np.pi * np.arange(N) / N) for i in range(x.shape[0])])
        return np.hstack([X_even + np.multiply(factor[:, :int(N / 2)], X_odd),
                          X_even + np.multiply(factor[:, int(N / 2):], X_odd)])


def FFT2D(img):
    return FFT(FFT(img).T).T


def FFT_SHIFT(img):
    M, N = img.shape
    M = int(M / 2)
    N = int(N / 2)
    return np.vstack((np.hstack((img[M:, N:], img[M:, :N])), np.hstack((img[:M, N:], img[:M, :N]))))


# 测试
img = np.zeros([128, 128], dtype=np.uint8)
img[62: 66, 62: 66] = 1   # 4*4
# img[30: 34, 30: 34] = 1  # 4*4
# img[58: 70, 58: 70] = 1  # 12*12
# img[63: 65, 63: 65] = 1  # 2*2
print(img)
plt.subplot(141)
plt.imshow(img, cmap='gray')
plt.title('Input Image')

fft = FFT2D(img)  # fft变换
plt.subplot(142)
plt.imshow(abs(fft), cmap='gray')
plt.title('Image 1 After FFT')

f_shift = FFT_SHIFT(fft)  # 将直流分量移动到频谱的中央
plt.subplot(143)
plt.imshow(abs(f_shift), cmap='gray')
plt.title('Image 2 After FFT')

magnitude_spectrum = 20 * np.log(1 + np.abs(f_shift))  # 动态增强
plt.subplot(144)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Image 3 After FFT')
plt.show()
