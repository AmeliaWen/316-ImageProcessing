import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.sparse import csr_matrix
import scipy.sparse
import time
import statistics
import argparse

def slow_one_dimension(a):
    a = np.asarray(a, dtype=complex)
    N = a.shape[0]
    res = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            res[k] += a[n] * np.exp(-2j * np.pi * k * n / N)
    return res

def slow_one_dimension_inverse(a):
    a = np.asarray(a, dtype=complex)
    N = a.shape[0]
    res = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            res[n] += a[k] * np.exp(2j * np.pi * k * n / N)
        res[n] /= N
    return res


def fast_one_dimension_inverse(a):
    a = np.asarray(a, dtype=complex)
    N = a.shape[0]

    if N % 2 != 0:
        raise AssertionError("size of a must be a power of 2")
    elif N <= 16:
        return slow_one_dimension_inverse(a)
    else:
        even = fast_one_dimension_inverse(a[::2])
        odd = fast_one_dimension_inverse(a[1::2])
        res = np.zeros(N, dtype=complex)

        half_size = N // 2
        for n in range(N):
            res[n] = half_size * even[n % half_size] + np.exp(2j * np.pi * n / N) * half_size * odd[n % half_size]
            res[n] /= N

        return res

def fast_two_dimension_inverse(a):
    a = np.asarray(a, dtype=complex)
    N, M = a.shape
    res = np.zeros((N, M), dtype=complex)
    for row in range(N):
        res[row, :] = fast_one_dimension_inverse(a[row, :])
    for col in range(M):
        res[:, col] = fast_one_dimension_inverse(res[:, col])
    return res

def fast_one_dimension(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    if N % 2 != 0:
        raise AssertionError("size of a must be a power of 2")
    elif N <= 16:
        return slow_one_dimension(x)
    else:
        even = fast_one_dimension(x[::2])
        odd = fast_one_dimension(x[1::2])
        res = np.zeros(N, dtype=complex)
        for n in range (N):
            res[n] = even[n % (N//2)] + np.exp(-2j * np.pi * n / N) * odd[n % (N//2)]
        return res

def fast_two_dimension (img):
    a = np.asarray(img, dtype=complex)
    w, h = a.shape
    res = np.empty_like(img, dtype=complex)
    #res = np.zeros((w, h), dtype=complex)
    for i in range(h):
        res[:, i] = fast_one_dimension(a[:,i])
    for j in range(w):
        res[j, :] = fast_one_dimension(res[j, :])
    return res


def IFFT2_helper(x):
    temp = np.flip(x[1:])
    x = np.concatenate(([x[0]], temp), axis=0)
    return fast_one_dimension(x)


def IFFT2(x):
    result = np.empty_like(x, dtype=complex)
    w, h = x.shape

    for i in range(w):
        result[i, :] = IFFT2_helper(x[i, :])

    for i in range(h):
        result[:, i] = IFFT2_helper(result[:, i])

    return result / (w * h)



def denoise(img):
    fft_img = img.copy()

    h, w = fft_img.shape
    h_fraction = 0.1
    fft_img[int(h_fraction * h):int(h*(1- h_fraction)),:] = 0.0
    w_fraction = 0.1
    fft_img[:, int(w_fraction * w):int(w*(1- w_fraction))] = 0.0

    non_zero_count = np.count_nonzero(fft_img)
    print("amount of non-zeros: ", non_zero_count)
    print("fraction of non-zero coefficient: ", non_zero_count / fft_img.size)

    denoised = fast_two_dimension_inverse(fft_img)
    return denoised.real



def compress_f (img, filename, precentage):
    fft_img = img.copy()
    h = int (math.sqrt(1-precentage) * (fft_img.shape[0] / 2))
    w = int (math.sqrt(1-precentage) * (fft_img.shape[1] / 2))
    fft_img[h:-h, :] = 0 + 0.j
    fft_img[:, w:-w] = 0 + 0.j
    nonzero_pre_compression = np.count_nonzero(img)
    nonzero_post_compression = np.count_nonzero(fft_img)
    print("nonzero values: ", np.count_nonzero(fft_img))
    print("compression ratio: ", 1 - (nonzero_post_compression / nonzero_pre_compression))
    temp_parse = csr_matrix (fft_img)
    scipy.sparse.save_npz(filename+'_'+str(precentage) + ".npz", temp_parse)
    return temp_parse

def slow_two_dimension(a):
    a = np.asarray(a, dtype=complex)
    N, M = a.shape
    res = np.zeros((N, M), dtype=complex)
    for k in range(N):
        for l in range(M):
            for m in range(M):
                for n in range(N):
                    res[k, l] += a[n, m] * \
                         np.exp(-2j * np.pi * ((l * m / M) + (k * n / N)))
    return res

def mode_4():
    print("mode 4 is triggered")
    # print(np.allclose(a, a2))
    # print(str(endTime-startTime) + " " + str(endTime2-startTime2))
    size = [32, 64, 128, 256, 512]

    dft_mean = list()
    dft_std = list()
    fft_mean = list()
    fft_std = list()

    x = 32
    for j in range(5):
        dft_list = list()
        for i in range(15):
            y = np.random.rand(x, x)
            startTime = time.perf_counter()
            fast_two_dimension(y)
            endTime = time.perf_counter()
            # print(np.allclose(my, np.fft.fft2(y)))
            diffTime = endTime - startTime
            dft_list.append(diffTime)

        dft_mean.append(statistics.mean(dft_list))
        dft_std.append(statistics.stdev(dft_list))
        x *= 2

    x = 32
    for j in range(5):
        fft_list = list()
        for i in range(15):
            y = np.random.rand(x, x)
            startTime = time.perf_counter()
            slow_two_dimension(y)
            endTime = time.perf_counter()
            # print(np.allclose(my, np.fft.fft2(y)))
            diffTime = endTime - startTime
            fft_list.append(diffTime)

        fft_mean.append(statistics.mean(fft_list))
        fft_std.append(statistics.stdev(fft_list))
        x *= 2

    plt.figure("Mode_4")
    plt.subplot(121)
    plt.plot(size, dft_mean, label="DFT")
    plt.plot(size, fft_mean, label="FFT")
    plt.xlabel('Size')
    plt.ylabel('Runtime Mean')
    plt.title('Mean')
    plt.legend()

    plt.subplot(122)
    plt.plot(size, dft_std, label="DFT")
    plt.plot(size, fft_std, label="FFT")
    plt.xlabel('Size')
    plt.ylabel('Runtime Std. Dev.')
    plt.title('Standard Deviation')
    plt.legend()
    plt.show()


def mode_2 (iname):
    img = cv2.imread(iname, cv2.IMREAD_UNCHANGED)
    vertical = img.shape[0]
    horizontal = img.shape[1]
    new_shape = (changeSize(vertical), changeSize(horizontal))
    img = cv2.resize(img, new_shape)
    img_FFT = fast_two_dimension(img)
    denoised = denoise(img_FFT)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(np.abs(denoised), norm=colors.LogNorm())
    plt.show()

def mode_3 (iname):
    filename = iname.split('.')[0]
    img = cv2.imread(iname, cv2.IMREAD_UNCHANGED)
    vertical = img.shape[0]
    horizontal = img.shape[1]
    new_shape = (changeSize(vertical), changeSize(horizontal))
    img = cv2.resize(img, new_shape)
    img_FFT = fast_two_dimension(img)
    compress_1 = compress_f (img_FFT, filename, 0)
    compress_2 = compress_f(img_FFT, filename, 0.25)
    compress_3 = compress_f(img_FFT, filename, 0.4)
    compress_4 = compress_f(img_FFT, filename, 0.6)
    compress_5 = compress_f(img_FFT, filename, 0.8)
    compress_6 = compress_f(img_FFT, filename, 0.95)
    uncompressed_1 = fast_two_dimension_inverse(compress_1.toarray())
    uncompressed_2 = fast_two_dimension_inverse(compress_2.toarray())
    uncompressed_3 = fast_two_dimension_inverse(compress_3.toarray())
    uncompressed_4 = fast_two_dimension_inverse(compress_4.toarray())
    uncompressed_5 = fast_two_dimension_inverse(compress_5.toarray())
    uncompressed_6 = fast_two_dimension_inverse(compress_6.toarray())
    plt.subplot(321), plt.imshow(uncompressed_1.real, cmap='gray')
    plt.title("0% compression"), plt.xticks([]), plt.yticks([])

    plt.subplot(322), plt.imshow(uncompressed_2.real, cmap='gray')
    plt.title("25% compression"), plt.xticks([]), plt.yticks([])

    plt.subplot(323), plt.imshow(uncompressed_3.real, cmap='gray')
    plt.title("40% compression"), plt.xticks([]), plt.yticks([])

    plt.subplot(324), plt.imshow(uncompressed_4.real, cmap='gray')
    plt.title("60% compression"), plt.xticks([]), plt.yticks([])

    plt.subplot(325), plt.imshow(uncompressed_5.real, cmap='gray')
    plt.title("80% compression"), plt.xticks([]), plt.yticks([])

    plt.subplot(326), plt.imshow(uncompressed_6.real, cmap='gray')
    plt.title("95% compression"), plt.xticks([]), plt.yticks([])

    plt.show()


def changeSize (n):
    p = int(math.log(n, 2))
    return int(pow(2, p+1))

def mode_1 (iname) :
    img = cv2.imread(iname, cv2.IMREAD_GRAYSCALE)
    vertical = img.shape[0]
    horizontal = img.shape[1]
    new_shape = (changeSize(vertical), changeSize(horizontal))
    img = cv2.resize(img, new_shape)
    img_FFT = fast_two_dimension(img)
    plt.figure("Mode_1")
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(np.abs(img_FFT), norm=colors.LogNorm())
    plt.show()

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', action='store', dest='mode',
                        help='Mode of operation 1-> fast, 2-> denoise, 3-> compress&save 4-> plot', type=int, default=1)
    parser.add_argument('-i', action='store', dest='image',
                        help='image path to work on', type=str, default='moonlanding.png')
    return parser.parse_args()

if __name__ == '__main__':
    mode = 1
    image = "moonlanding.png"
    try :
        result = parseArgs()
    except BaseException as e:
        print("ERROR\tIncorrect input syntax: Please check arguments and try again")
        exit(1)
    mode = result.mode
    image = result.image
    if (mode ==1):
        mode_1(image)
    elif (mode == 2):
        mode_2(image)
    elif (mode == 3):
        mode_3(image)
    elif (mode == 4):
        mode_4()
    else:
        print("mode not recognized")
        exit(1)