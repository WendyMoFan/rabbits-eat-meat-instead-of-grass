# 小波维纳滤波
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pywt
from PIL import Image
import shutil


# print(pywt.families())
# ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']

# 引入高斯噪声
# def guass_noise(pic, SNR=1):
#     # SNR为信噪比
#     pic = np.array(pic, dtype=float)
#     SNR = 10 ** (SNR / 10)
#     row, col = np.shape(pic)
#     pic_power = np.sum(pic * pic) / (row * col)
#     noise_power = pic_power / SNR
#     noise = np.random.randn(row, col) * np.sqrt(noise_power)
#     pic = (noise + pic)
#     pic = np.where(pic <= 0, 0, pic)
#     pic = np.where(pic > 255, 255, pic)
#     return np.uint8(pic)


# 原图和加噪图像做差
# def SNF_calcu(pic, pic_):
#     # pic 原图  pic_被污染的图片
#     pic = np.array(pic, dtype=float)
#     pic_ = np.array(pic_, dtype=float)
#     noise = pic - pic_
#     return 10 * np.log10(np.sum(pic * pic) / np.sum(noise * noise))


# 维纳滤波器
def wiener_filter(pic, HH):
    # r padding半径
    row, col = np.shape(pic)
    noise_std = (np.median(np.abs(HH)) / 0.6745)
    noise_var = noise_std ** 2
    var = 1 / (row * col) * np.sum(pic * pic) - noise_var
    ans = pic * var / (var + noise_var)
    return ans


# 维纳滤波与重构
def wiener_dwt(pic, index=1):
    # index 为进行几层分解与重构
    pic = np.array(pic, dtype=float)
    coeffs = pywt.dwt2(pic, 'bior4.4')
    LL, (LH, HL, HH) = coeffs

    # LL为低频信号 LH为水平高频 HL为垂直高频  HH为对角线高频信号

    # 维纳滤波
    LH = wiener_filter(LH, HH)
    HL = wiener_filter(HL, HH)
    HH = wiener_filter(HH, HH)

    # 重构
    if index > 1:
        LL = wiener_dwt(LL, index - 1)
        # bior4.4小波重构可能会改变矩阵维数，现统一矩阵维数
        row, col = np.shape(LL)
        d1 = row - np.shape(HH)[0]
        d2 = col - np.shape(HH)[1]
        if d1 > 0 or d2 > 0:
            d1 = row - np.arange(d1) - 1
            d2 = col - np.arange(d2) - 1
            LL = np.delete(LL, d1, axis=0)
            LL = np.delete(LL, d2, axis=1)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(LL, cmap='gray')
    plt.title('The ' + str(time - index + 1) + 'th dwt----' + 'LL')
    plt.subplot(2, 2, 2)
    plt.imshow(LH, cmap='gray')
    plt.title('The ' + str(time - index + 1) + 'th dwt----' + 'LH')
    plt.subplot(2, 2, 3)
    plt.imshow(HL, cmap='gray')
    plt.title('The ' + str(time - index + 1) + 'th dwt----' + 'HL')
    plt.subplot(2, 2, 4)
    plt.imshow(HH, cmap='gray')
    plt.title('The ' + str(time - index + 1) + 'th dwt----' + 'HH')
    plt.show()
    pic_ans = pywt.idwt2((LL, (LH, HL, HH)), 'bior4.4')
    # pic_ans = np.where(pic_ans <= 0, 0, pic_ans)
    # pic_ans = np.where(pic_ans > 255, 255, pic_ans)
    return pic_ans


# 图像与去噪后图像做差
def subtract_img(img_1, img_2):
    width = img_1.shape[1]
    height = img_1.shape[0]

    img_2 = cv2.resize(img_2, (width, height))
    # img_1 = cv2.imread(img1_filename)
    # img_2 = cv2.imread(img2_filename)
    diffImg1 = cv2.subtract(img_1, img_2)
    diffImg2 = cv2.subtract(img_2, img_1)
    # diffImg3 = img_1 - img_2
    # diffImg4 = img_2 - img_1
    plt.subplot(1, 2, 1)
    plt.imshow(diffImg1, cmap='gray')
    plt.title('image after subtract')
    plt.subplot(1, 2, 2)
    plt.imshow(diffImg2, cmap='gray')
    plt.title('image after subtract')
    plt.show()
    # cv2.imshow('subtract(img1,img2)', diffImg1)
    # cv2.imshow('subtract(img2,img1)', diffImg2)
    # cv2.imshow('img1 - img2', diffImg3)
    # cv2.imshow('img2 - img1', diffImg4)
    # cv2.imwrite("noise.jpg", diffImg3)
    return diffImg1  # 暂时的哈


# 增强图像    看那个噪声图像在哪种增强方法下更容易分辨咱就用哪个
def enhance_noise(image):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image_clahe = cv2.merge([b, g, r])
    return image_clahe


# 分割噪声图像
def split_image(pic_path):
    data = 0
    i = 0
    j = 0
    # pic_path = 'noise.jpg'  # 分割的图片的位置
    pic_target = 'result'  # 分割后的图片保存的文件夹
    # 要分割后的尺寸
    cut_width = 60  # 512
    cut_length = 60  # 512
    # 读取要分割的图片，以及其尺寸等数据
    picture = cv2.imread(pic_path)
    (width, length, depth) = picture.shape
    # 预处理生成0矩阵
    pic = np.zeros((cut_width, cut_length, depth))
    # 计算可以划分的横纵的个数
    num_width = int(width / cut_width)
    num_length = int(length / cut_length)
    # for循环迭代生成
    for i in range(0, num_width):
        for j in range(0, num_length):
            pic = picture[i * cut_width: (i + 1) * cut_width, j * cut_length: (j + 1) * cut_length, :]
            result_path = pic_target + '{}_{}.jpg'.format(i + 1, j + 1)
            data = (i + 1)*(j + 1)
            cv2.imwrite(os.path.join('D:/code/bishe/result', result_path), pic)
    print("共分割成了", data, "张照片")
    # print(i, "行", j, "列")
    return data, i + 1, j + 1


# 比较分割后结果
# test_5.py
# 运行函数
def run(filename):
    # SNR = 22
    global time
    flag = []
    time = 5  # 分解次数
    f = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # f_noise = guass_noise(f, SNR)
    f_process = wiener_dwt(f, time)
    f_process = np.where(f_process <= 0, 0, f_process)
    f_process = np.where(f_process > 255, 255, f_process)
    f_process = np.uint8(f_process)

    # print(SNF_calcu(f, f_noise))
    # print(SNF_calcu(f, f_process))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(f, cmap='gray')
    plt.title('original image')
    # plt.subplot(1, 3, 2)
    # plt.imshow(f_noise, cmap='gray')
    # plt.title('polluted image ----SNR = ' + str(SNR))
    plt.subplot(1, 2, 2)
    plt.imshow(f_process, cmap='gray')
    plt.title('image after wiener_dwt')
    plt.show()
    # print(f.shape, f_process.shape)
    # subtract_img(f, f_process)
    noise_img = subtract_img(f, f_process)
    cv2.imwrite("noise.jpg", noise_img)
    noise_img = cv2.imread("noise.jpg")
    noise_img_enhance = enhance_noise(noise_img)
    cv2.imwrite("noise_enhance.jpg", noise_img_enhance)
    # split_image('noise_enhance.jpg')
    flag = split_image('noise_enhance.jpg')
    print(flag[0], flag[1], flag[2])


def pixel_equal(image1, image2, x, y):
    """
    判断两个像素是否相同
    :param image1: 图片1
    :param image2: 图片2
    :param x: 位置x
    :param y: 位置y
    :return: 像素是否相同
    """
    # 取两个图片像素点
    piex1 = image1.load()[x, y]
    piex2 = image2.load()[x, y]
    threshold = 10
    # 比较每个像素点的RGB值是否在阈值范围内，若两张图片的RGB值都在某一阈值内，则我们认为它的像素点是一样的
    if abs(piex1[0] - piex2[0]) < threshold and abs(piex1[1] - piex2[1]) < threshold and abs(
            piex1[2] - piex2[2]) < threshold:
        return True
    else:
        return False


def compare(image1, image2):
    """
    进行比较
    :param image1:图片1
    :param image2: 图片2
    :return:
    """
    left = 10  # 坐标起始位置
    right_num = 0  # 记录相同像素点个数
    false_num = 0  # 记录不同像素点个数
    all_num = 0  # 记录所有像素点个数
    for i in range(left, image1.size[0]):
        for j in range(image1.size[1]):
            if pixel_equal(image1, image2, i, j):
                right_num += 1
            else:
                false_num += 1
            all_num += 1
#    same_rate = right_num / all_num  # 相同像素点比例
    nosame_rate = false_num / all_num  # 不同像素点比例
    # print("same_rate: ", same_rate)
    # print("nosame_rate: ", nosame_rate)
    return nosame_rate


def compare_all(number, i, j):
    pic_target = 'D:/code/bishe/result'
    file_list = os.listdir(pic_target)
    flag_0 = 0
    flag = 0
    f = 0
    # rename()
    for count_r in range(0, len(file_list)):
        im_name = file_list[count_r]
        im_path = os.path.join(pic_target, im_name)
        image1 = Image.open(im_path)
        for count_l in range(count_r + 1, len(file_list)):
            im_name = file_list[count_l]
            im_path = os.path.join(pic_target, im_name)
            image2 = Image.open(im_path)
            cc = compare(image1, image2)
            if cc > 0.3:
                flag_0 = flag_0 + 1
            flag = flag + 1
    # print(flag_0)
    # print(flag)
    rate = flag_0/flag
    if rate > 0.4:
        f = 1
        print("通过检测该图像经过篡改")
    else:
        f = 0
        print("该图像未经过篡改")
    return f


# if __name__ == "__main__":
#     shutil.rmtree('result')
#     os.mkdir('result')
#     run('D:/code/bishe/test/sp_2/16.jpg')
#     compare_all(0, 0, 0)

if __name__ == "__main__":

    target = 'D:/code/bishe/test/sp_1'
    l_ist = os.listdir(target)
    f_lag = 0

    for count in range(0, len(l_ist)):
        imname = l_ist[count]
        impath = os.path.join(target, imname)
        shutil.rmtree('result')
        os.mkdir('result')
        print("第", count, "个")
        run(impath)
        f = compare_all(0, 0, 0)
        if f == 1:
            f_lag = f_lag + 1
    print("篡改图片个数", f_lag)
