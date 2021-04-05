import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import argparse
from tqdm import tqdm

# this is the fast fourier transform base case
def sfft_1d(a):
    a = np.asarray(a, dtype=complex)
    N = a.shape[0]
    res = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            res[k] += a[n] * np.exp(-2j * np.pi * k * n / N)
    return res

# this is the inverse fast fourier transform in 1 dimension (base case)
def ifft_1d(a):
    a = np.asarray(a, dtype=complex)
    N = a.shape[0]
    res = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            res[n] += a[k] * np.exp(2j * np.pi * k * n / N)
        res[n] /= N
    return res

# this is the inverse fast fourier transform called from ifft_2d
def ifft(a):
    a = np.asarray(a, dtype=complex)
    N = a.shape[0]
    # check size validation
    if N % 2 != 0:
        raise AssertionError("size of a must be a power of 2")
    # run base case
    elif N <= 16:
        return ifft_1d(a)
    # recursive call
    else:
        even = ifft(a[::2])
        odd = ifft(a[1::2])
        res = np.exp(2j * np.pi * np.arange(N) / N).astype(np.complex64)
        return np.concatenate((even + res[:N // 2] * odd,
                               even + res[N // 2:] * odd), axis=0)

# this is the inverse fast fourier transform in 2 dimension
def ifft_2d(a):
    a = np.asarray(a, dtype=complex)
    N, M = a.shape
    res = np.zeros((N, M), dtype=complex)
    for row in range(N):
        res[row, :] = ifft(a[row, :])
    for col in range(M):
        res[:, col] = ifft(res[:, col])
    return res

# this is the fast fourier transform in 1 dimension
def fft_1d(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    if N % 2 > 0:
        raise AssertionError("size of a must be a power of 2")
    elif N <= 16:
        return sfft_1d(x)
    else:
        even = fft_1d(x[::2])
        odd = fft_1d(x[1::2])
        res = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([even + res[:int(N / 2)] * odd, even + res[int(N / 2):] * odd])

# this is the fast fourier transform in 2 dimension
def fft_2d (img):
    a = np.asarray(img, dtype=complex)
    w, h = a.shape
    res = np.empty_like(img, dtype=complex)
    for i in range(h):
        res[:, i] = fft_1d(a[:,i])
    for j in range(w):
        res[j, :] = fft_1d(res[j, :])
    return res

# this is the helper method for mode 2
# we investigated three denoising methods
# 1. remove high frequency
# 2. width and height have different fraction
# 3. threshold everything, threshold is 0.9
# it prints in the command line the number of non-zeros
def denoise(img, type, precentage, test):
    fft_img = img.copy()
    h, w = fft_img.shape
    if type == 1:
        print("remove high frequency")
        for r in tqdm(range(h)):
            for c in range(w):
                if r > h * precentage and r < h*(1-precentage):
                    fft_img[r, c]= 0
                if c > w * precentage and c < w*(1-precentage):
                    fft_img[r, c] = 0
        non_zero_count = np.count_nonzero(fft_img)
        print("amount of non-zeros: ", non_zero_count)
        print("fraction of non-zero coefficient: ", non_zero_count / fft_img.size)
        denoised = ifft_2d(fft_img)
        #not in test mode
        if test == 0:
            plt.subplot(122)
        else:
            plt.subplot(131)
        plt.imshow(np.abs(denoised), norm=colors.LogNorm())

    elif type == 2:
        print("width and height have different fraction")
        h_fraction = 0.1
        fft_img[int(h_fraction * h):int(h * (1 - h_fraction)), :] = 0.0
        w_fraction = 0.15
        fft_img[:, int(w_fraction * w):int(w * (1 - w_fraction))] = 0.0
        non_zero_count = np.count_nonzero(fft_img)
        print("amount of non-zeros: ", non_zero_count)
        print("fraction of non-zero coefficient: ", non_zero_count / fft_img.size)
        denoised = ifft_2d(fft_img)
        if test == 0:
            plt.subplot(122)
        else:
            plt.subplot(132)
        plt.imshow(np.abs(denoised), norm=colors.LogNorm())

    elif type == 3:
        print("threshold everything, threshold is 0.9 ")
        threshold = fft_img.real.max() * 0.9
        for r in tqdm(range(h)):
            for c in range(w):
                if fft_img[r, c] < threshold and fft_img[r, c] > -threshold:
                    fft_img[r, c] = fft_img[r, c]
                elif fft_img[r, c] <= -threshold:
                    fft_img[r, c] = -threshold
                else :
                    fft_img[r, c] = threshold
        non_zero_count = np.count_nonzero(fft_img)
        print("amount of non-zeros: ", non_zero_count)
        print("fraction of non-zero coefficient: ", non_zero_count / fft_img.size)
        denoised = ifft_2d(fft_img)
        if test == 0:
            plt.subplot(122)
        else:
            plt.subplot(133)
        plt.imshow(np.abs(denoised), norm=colors.LogNorm())


# this is the helper method for mode 3
# it keeps the value for a certain percentage of image file and make others 0
def compress_f (img, filename, precentage):
    fft_img = img.copy()
    w, h = fft_img.shape
    h = int (math.sqrt(1-precentage) * (fft_img.shape[0] / 2))
    w = int (math.sqrt(1-precentage) * (fft_img.shape[1] / 2))
    fft_img[h:-h, :] = 0
    fft_img[:, w:-w] = 0
    print("compressing ", precentage, " percentage of the image")
    print("nonzero values: ", np.count_nonzero(fft_img))
    name = filename+"_"+str(precentage) + ".csv"
    np.savetxt(name, fft_img, delimiter=",")
    return ifft_2d(fft_img).real

# this method is the slow version of fft algorithm
def sfft (a):
    a = np.asarray(a, dtype=complex)
    N, M = a.shape
    res = np.zeros((N, M), dtype=complex)
    for r in range(N):
        for c in range(M):
            for m in range(M):
                for n in range(N):
                    res[r, c] += a[n, m] * np.exp(-2j * np.pi * ((float(r * n) / N) + (float (c * m) / M)))
    return res

# this method is called when using mode 4
# we produce plots that summarize the runtime complexity of your algorithms.
# It prints in the command line the means and variances of the runtime of your algorithms versus the problem size.
def mode_4():
    print("mode 4 is triggered")
    size = [32, 64, 128]
    slow_time = list()
    fast_time = list()
    dft_mean = list()
    dft_std = list()
    fft_mean = list()
    fft_std = list()

    x = 32
    for j in range(3):
        dft_list = list()
        fft_list = list()
        for i in range(10):
            y = np.random.rand(x, x).astype(np.float32)
            startTime = time.time()
            fft_2d(y)
            endTime = time.time()
            diffTime = endTime - startTime
            print("Fast time: {}".format(diffTime))
            dft_list.append(diffTime)
            slow_start = time.time()
            sfft(y)
            slow_end = time.time()
            diffTimeSlow = slow_end-slow_start
            print("Slow time: {}".format(diffTimeSlow))
            fft_list.append(diffTimeSlow)
        slow_time.append(fft_list)
        fast_time.append(dft_list)
        x *= 2
    slow_time = np.array(slow_time)
    fast_time = np.array(fast_time)
    slow_mean = slow_time.mean(axis=1)
    slow_std = slow_time.std(axis=1) * 2
    fast_mean = fast_time.mean(axis=1)
    fast_std = fast_time.std(axis=1) * 2
    plt.figure("Mode_4")
    power = np.arange(5, 8)
    plt.subplot(133)
    plt.errorbar(power, slow_mean, yerr=slow_std, label="slow")
    plt.errorbar(power, fast_mean, yerr=fast_std, label="fast")
    plt.xlabel("size of test data (power of 2)")
    plt.ylabel("runtime (second)")
    plt.xticks(power)
    plt.title("Runtime for slow FT against fast FT")
    plt.legend(loc='best')
    plt.show()

# after experiment, we found type2 denoise method produces the best result.
# this method output a one by two subplot.
# In this subplot we include the original image next to its denoised version.
def mode_2 (iname, type, precentage):
    img = cv2.imread(iname, cv2.IMREAD_UNCHANGED)
    vertical = img.shape[0]
    horizontal = img.shape[1]
    new_shape = (changeSize(vertical), changeSize(horizontal))
    img = cv2.resize(img, new_shape)
    img_FFT = fft_2d(img)
    plt.subplot(121)
    plt.imshow(img)
    denoise(img_FFT, type, precentage, 0)
    plt.show()

# this method is used for the test mode
# it produces 3 subplots using different denoise methods
def mode_2_test (iname, precentage):
    img = cv2.imread(iname, cv2.IMREAD_UNCHANGED)
    vertical = img.shape[0]
    horizontal = img.shape[1]
    new_shape = (changeSize(vertical), changeSize(horizontal))
    img = cv2.resize(img, new_shape)
    img_FFT = fft_2d(img)
    denoise(img_FFT, 1, precentage, 1)
    denoise(img_FFT, 2, precentage, 2)
    denoise(img_FFT, 3, precentage, 3)
    plt.show()

# Firstly, we take the FFT of the image to compress it.
# The compression comes from setting some Fourier coefficients to zero calling compress_f.
# we experiment on various parameters from compression
def mode_3 (iname):
    filename = iname.split('.')[0]
    img = cv2.imread(iname, cv2.IMREAD_UNCHANGED)
    vertical = img.shape[0]
    horizontal = img.shape[1]
    new_shape = (changeSize(vertical), changeSize(horizontal))
    img = cv2.resize(img, new_shape)
    img_FFT = fft_2d(img)

    compress_1 = compress_f(img_FFT, filename, 0)
    compress_2 = compress_f(img_FFT, filename, 0.25)
    compress_3 = compress_f(img_FFT, filename, 0.4)
    compress_4 = compress_f(img_FFT, filename, 0.6)
    compress_5 = compress_f(img_FFT, filename, 0.8)
    compress_6 = compress_f(img_FFT, filename, 0.95)

    plt.subplot(321), plt.imshow(compress_1.real, cmap='gray')
    plt.title("0% compression"), plt.xticks([]), plt.yticks([])

    plt.subplot(322), plt.imshow(compress_2.real, cmap='gray')
    plt.title("25% compression"), plt.xticks([]), plt.yticks([])

    plt.subplot(323), plt.imshow(compress_3.real, cmap='gray')
    plt.title("40% compression"), plt.xticks([]), plt.yticks([])

    plt.subplot(324), plt.imshow(compress_4.real, cmap='gray')
    plt.title("60% compression"), plt.xticks([]), plt.yticks([])

    plt.subplot(325), plt.imshow(compress_5.real, cmap='gray')
    plt.title("80% compression"), plt.xticks([]), plt.yticks([])

    plt.subplot(326), plt.imshow(compress_6.real, cmap='gray')
    plt.title("95% compression"), plt.xticks([]), plt.yticks([])
    plt.show()


# this method is used to resize the image
def changeSize (n):
    p = int(math.log(n, 2))
    return int(pow(2, p+1))

# simply perform the FFT and output a one by two subplot
# of the original image and next to it its Fourier transform.
def mode_1 (iname) :
    img = cv2.imread(iname, cv2.IMREAD_GRAYSCALE)
    vertical = img.shape[0]
    horizontal = img.shape[1]
    new_shape = (changeSize(vertical), changeSize(horizontal))
    img = cv2.resize(img, new_shape)
    img_FFT = fft_2d(img)
    plt.figure("Mode_1")
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(np.abs(img_FFT), norm=colors.LogNorm())
    plt.show()

# this produces the two subplots
# of the Fourier transform we implemented and next to it the built in fft2 function in numpy.
def mode_1_test (iname):
    img = cv2.imread(iname, cv2.IMREAD_GRAYSCALE)
    vertical = img.shape[0]
    horizontal = img.shape[1]
    new_shape = (changeSize(vertical), changeSize(horizontal))
    img = cv2.resize(img, new_shape)
    img_FFT = fft_2d(img)
    plt.figure("Mode_1_test")
    plt.subplot(121)
    plt.imshow(np.abs(img_FFT), norm=colors.LogNorm())
    img_FFT_2 = np.fft.fft2(img)
    plt.subplot(122)
    plt.imshow(np.abs(img_FFT_2), norm=colors.LogNorm())
    plt.show()

def parseArgs():
    parser = argparse.ArgumentParser()
    helper = {
        1: "[1] (Default) for fast mode where ther image is converted into its FFT form and displayed",
        2: "[2] for denoising where the image is denoised by applying an FFT, truncating high frequencies and then displyed",
        3: "[3] for compressing and saving the image",
        4: "[4] for plotting the runtime graphs for the report"
    }
    parser.add_argument('-m', action='store', dest='mode',
                        help=''.join(helper.values()), type=int, default=1)
    parser.add_argument('-i', action='store', dest='image',
                        help='image to process', type=str, default='moonlanding.png')
    parser.add_argument('-t', action='store', dest='test',
                        help='this mode is used to test the program', type=int, default=0)
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
    test = result.test
    if (test == 1):
        mode_1_test(image)
    elif (test == 2):
        mode_2_test(image, 0.1)
    elif (mode ==1):
        mode_1(image)
    elif (mode == 2):
        mode_2(image, 1, 0.1)
    elif (mode == 3):
        mode_3(image)
    elif (mode == 4):
        mode_4()
    else:
        print("mode not recognized")
        exit(1)
