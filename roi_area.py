import cv2 as cv
import numpy as np
import math
from PIL import Image
import os
import os.path
from matplotlib import pyplot as plt


# 边缘检测画出边缘
def get_edges(img):  # img : single channel img which has been processed by cv.Canny
    rows, cols = img.shape
    center = rows / 2
    index = []
    up_index = []
    for i in range(cols):
        temp_index = np.argwhere(img[:, i] == 255)
        temp_index = temp_index.tolist()
        for j in temp_index:
            if j[0] < center:
                up_index.append(center - j[0])
        up = len(up_index) - 1
        up_index.clear()
        down = up + 1
        try:
            index.append([temp_index[up][0], i])
        except IndexError:
            index.append([index[-2][0], i])
        try:
            index.append([temp_index[down][0], i])
        except IndexError:
            index.append([index[-2][0], i])
    result_img = np.zeros((rows, cols), dtype="uint8")
    for j in index:
        result_img[j[0], j[1]] = 255  # put the Corresponding coordinate into 255
    return result_img


# 得到中点坐标
def get_center_line(image):
    rows, cols = image.shape
    code = []
    for i in range(cols):
        temp_index = np.argwhere(image[:, i] == 255)
        temp = temp_index.tolist()
        center_point = round((temp[0][0] + temp[1][0]) / 2)
        temp.clear()
        code.append([center_point, i])
    points = np.array(code)
    return points


# 将中线虚拟呈直线并在边缘图上画出
c_test = cv.imread('05.jpg', 0)
sobel_x = cv.Sobel(c_test, cv.CV_64F, 1, 0, ksize=5)
laplacian = cv.Laplacian(c_test, cv.CV_64F)
canny = cv.Canny(c_test, 50, 100)
finger_edges = get_edges(canny)
points = get_center_line(finger_edges)
line_vir = cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01)
# print(line_vir)
line = cv.line(finger_edges, (int(line_vir[2] - 1000 * line_vir[1] / line_vir[0]), int(line_vir[2] - 1000)),
               (int(line_vir[2] + 1000 * line_vir[1] / line_vir[0]), int(line_vir[2] + 1000)), 255)

# 根据中线角度旋转图形
k = line_vir[1] / line_vir[0]
angle = math.atan(k)
im = Image.open("05.jpg")
# im.show()
im1 = im.rotate(angle)
im1.show()
# cv.imshow("line", line)
im1.save("tt.jpg")

# test_0 = cv.imread('tt.jpg', 0)
# gray_map = test_0[240]
# print(gray_map)
# plt.plot(gray_map)
# plt.show()
pic_test = cv.imread('tt.jpg', 0)
rows, cols = pic_test.shape
gray_map = pic_test[240]
sobel_x_pic = cv.Sobel(pic_test, cv.CV_64F, 1, 0, ksize=5)
laplacian_pic = cv.Laplacian(pic_test, cv.CV_64F)
canny_pic = cv.Canny(pic_test, 50, 100)
finger_edges_pic = get_edges(canny_pic)
points_pic = get_center_line(finger_edges_pic)
# print(points_pic)
line_vir_pic = cv.fitLine(points_pic, cv.DIST_L2, 0, 0.01, 0.01)


# cv.waitKey(0)
# cv.destroyAllWindows()


# 找到最亮的两个横坐标
def find_peak(arr):
    peaks = []
    step = 1
    pos = 40
    while pos < 400:  # 实际上两个peak主要分布在100-200和450-550
        if (arr[pos] >= arr[pos + step]) and (arr[pos] > arr[pos - step]):
            if (arr[pos] > arr[pos + 20]) and (arr[pos] > arr[pos - 20]):  # 这个if语句可以滤除很多的噪音peak
                peaks.append(pos)
        pos = pos + step
    peak1 = 0
    peak2 = 300  # 取的图片中间靠后的位置，但是有可能存在后面的关节腔灰度值比中间位置低的情况
    # print(peaks)
    for i in peaks:  # 这个循环从好多个peak中找到我们要的两个最大的peak
        if i < 200:
            if arr[i] >= arr[peak1]:
                peak1 = i
        else:
            if arr[i] >= arr[peak2]:
                peak2 = i
    return peak1, peak2


divide_point = find_peak(gray_map)
# print(divide_point)
points_pic_1 = line_vir_pic[3]
min_x = int(divide_point[0] - 30)
max_x = int(divide_point[1] + 20)
min_y = int(points_pic_1 - 60)
max_y = int(points_pic_1 + 60)
pic_image = pic_test[min_y:max_y, min_x:max_x]
cv.imshow("ROI", pic_image)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite("roi.jpg", pic_image)
