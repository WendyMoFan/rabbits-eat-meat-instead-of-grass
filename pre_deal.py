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
im.show()
im1 = im.rotate(angle)
im1.show()
cv.imshow("line", line)
cv.imshow("canny", canny)
im1.save("tt.jpg")
cv.waitKey(0)
cv.destroyAllWindows()
