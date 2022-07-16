import numpy as np
import cv2 as cv
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import math
import os
import sys
import matplotlib.pyplot as plt
import scipy.signal as signal
from skimage import feature


def load_image(path):
    img_gray = cv.imread(path, 0)
    return img_gray


def LBP_Extract(path):
    img_gray = load_image(path)

    # 使用LBP方法提取图像的纹理特征
    img = feature.local_binary_pattern(img_gray, 8, 1, 'default')
    # 转换数据类型便于后面处理
    img = img.astype('uint8')
    # cv.imshow("img",img)

    # 将編碼圖分块為4×2
    tile_cols = 4
    tile_rows = 2
    n = 1
    rows, cols = img.shape
    step_rows = round(rows / tile_rows) - 1
    step_cols = round(cols / tile_cols) - 1
    LBP_feature = []
    for i in range(tile_rows):
        for j in range(tile_cols):
            tile = img[i * step_rows:(i + 1) * step_rows - 1, j * step_cols:(j + 1) * step_cols - 1]
            # 统计直方图
            hist = cv.calcHist([tile], [0], None, [255], [0, 256])
            # 對直方圖幅值進行归一化
            hist = normalization(hist)
            # 將靜脈圖像的LBP直方圖特徵串接
            LBP_feature.append(hist)

            # plt.figure(n)
            # plt.subplot(4,2,n)
            # TODO:調整子圖閒的距離
            # plt.subplots_adjust(left=0.04, top=0.96, right=0.96, bottom=0.04, wspace=0.01, hspace=0.01)
            # plt.plot(hist)
            # plt.axis('off')
            # title_name = 'LBP'+str(n)
            # plt.title(title_name)
            # plt.show()
            # plt.clf()
            # 計算直方圖的數量
            n += 1
    # plt.show()

    LBP_feature = np.array(LBP_feature)  # 8*255
    LBP_feature = LBP_feature.ravel()  # 2040*1

    # 繪製串接后的LBP特徵圖
    # plt.figure(2)
    # plt.plot(LBP_feature)
    # plt.show()
    return LBP_feature


def LBP_match(path1, path2):
    hist_template = LBP_Extract(path1)
    hist_test = LBP_Extract(path2)
    template_sum = hist_template.sum()
    score = 0
    # score就是取兩直方圖對應點的最小值,計算所有點之和
    # 取最小值是因爲最小值相當於是兩直方圖重叠的部分
    for i in range(hist_template.size):
        if hist_template[i] <= hist_test[i]:
            score = score + hist_template[i]
        else:
            score = score + hist_test[i]
    score = score / template_sum
    return score


# 對直方圖的幅值進行歸一化處理
def normalization(array):
    result = []
    array = array.ravel()
    array_sum = array.sum()
    for i in array:
        result.append(i / array_sum)
    return np.array(result)


if __name__ == "__main__":

    dir_1 = 'roi_hand_1'
    dir_2 = 'roi_hand_2'
    imgList_1 = os.listdir(dir_1)
    imgList_2 = os.listdir(dir_2)
    # 读取图片路径
    im_path_1 = []
    im_path_2 = []
    for count_1 in range(0, len(imgList_1)):
        im_name_1 = imgList_1[count_1]
        im_path_1.append(os.path.join(dir_1, im_name_1))

    for count_2 in range(0, len(imgList_2)):
        im_name_2 = imgList_2[count_2]
        im_path_2.append(os.path.join(dir_2, im_name_2))
    # 储存20张图片的路径

    # 类内概率密度图
    score_1 = []
    score_2 = []
    i = 0
    j = 1
    while i < 9:
        while j < 10:
            score_1.append(LBP_match(im_path_1[i], im_path_1[j]))  # 類内
            score_2.append(LBP_match(im_path_1[i], im_path_2[j]))  # 類間
            j = j + 1
        j = i + 2
        score_2.append(LBP_match(im_path_1[i], im_path_2[i]))
        i = i + 1
    # print(score_1)
    # print(score_2)

    sns.kdeplot(score_1, label="inner-class")  # 類内
    sns.kdeplot(score_2, label="between-class")  # 類間
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('匹配分数')  # 设置X轴标签
    plt.ylabel('概率')  # 设置Y轴标签
    plt.legend()
    plt.show()
    cv.destroyAllWindows()
