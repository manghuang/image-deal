"""
    自己实现一维dft变换和反变换、二维dft变换和反变换
"""
import numpy as np
from math import *


# 一维dft变换
def dft(a):
    a = np.asarray(a, dtype=float)
    N = a.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * pi * k * n / N) / N  # 变换矩阵
    result = np.dot(M, a)
    return result


# 一维dft反变换
def idft(a):
    N = a.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * pi * k * n / N)  # 反变换矩阵
    result = np.dot(M, a)
    return result


# 二维dft变换
def shift_ft(img):
    M, N = img.shape
    shift = np.matrix([[pow(-1, i + j) for j in range(N)] for i in range(M)])
    U = np.matrix([[np.exp(-1j * 2 * 3.14159 * m * i / M) for m in range(M)] for i in range(M)])
    V = np.matrix([[np.exp(-1j * 2 * 3.14159 * n * j / N) for j in range(N)] for n in range(N)])
    return U.dot(np.multiply(img, shift)).dot(V)


# 二维dft反变换
def shift_ift(img):
    M, N = img.shape
    shift = np.matrix([[pow(-1, i + j) for j in range(N)] for i in range(M)])
    U = np.matrix([[np.exp(1j * 2 * 3.14159 * m * i / M) for m in range(M)] for i in range(M)])
    V = np.matrix([[np.exp(1j * 2 * 3.14159 * n * j / N) for j in range(N)] for n in range(N)])
    return np.multiply(shift, U.dot(img).dot(V)) / M / N


# 测试
x = [1, 2, 3, 4, 5]
print(x)
print(dft(x))
print(idft(dft(x)))
