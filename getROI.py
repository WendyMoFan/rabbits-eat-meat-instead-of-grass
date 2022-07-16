import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.signal as signal
import math


def getRoiImg(PalmveinImg):
    # **********说明************
    # 这个函数是总的截取ROI的函数，输入原始手掌图片，返回手掌ROI图片，这个函数里面也包含了其他的辅助函数
    # *Input : the original PalmImage
    # *Output : the Palm ROI Image

    # ********伪代码**********
    Original_img = PalmveinImg.copy()
    PalmveinImg_gray = cv2.cvtColor(PalmveinImg, cv2.COLOR_BGR2GRAY)
    BackImg = PalmveinImg_gray.copy()
    # otsu method
    threshold, imgOtsu = cv2.threshold(BackImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('otsu.jpg', imgOtsu)

    # for row in range(0, 100):
    #     for col in range(800, 900):
    #         imgOtsu[row][col] = 255

    # cv2.imshow('Maximum inscribed circle', imgOtsu)
    # cv2.waitKey(0)

    #################################计算手掌的大小#################################
    ############计算图片的大小############
    rows = len(imgOtsu)
    cols = len(imgOtsu[0])
    length = count_size(imgOtsu, rows, cols)

    ##############################找手掌下面(手腕)的中点############################
    Pref_row = 0
    Pref_col = int(cols / 2)
    for row in range(0, rows):
        if imgOtsu[rows - 1 - row][Pref_col] == 255:
            Pref_row = rows - 1 - row - int(length * 0.01)
            # Pref_row = rows - 1 - row
            break
    col_boundary = np.zeros(2)
    i = 0
    for col in range(1, cols):
        if imgOtsu[Pref_row][col] != imgOtsu[Pref_row][col - 1]:
            col_boundary[i] = col
            i = i + 1
            if (i == 2):
                break
    Pref_col = int((col_boundary[0] + col_boundary[1]) / 2)
    # print(Pref_x,Pref_y)

    ############################找最大轮廓，再根据计算极值点#########################
    # Point* maxarea_point = new Point[10000];
    # totalnum = 0
    # 获取最大轮廓点集
    max_contours = getMaxRegion(imgOtsu, PalmveinImg)
    # print(len(max_contours))
    # print(len(max_contours[0]))

    # 获取极值点
    point9 = np.zeros((9, 2))
    point9 = find9point(Pref_row, max_contours)
    point9 = sortpoint(point9)
    # 打印9个点坐标
    # print(point9)

    # cv2.circle(PalmveinImg, (int(max_contours[10][0]),int(max_contours[10][1])), 5, (0,255,0))
    # cv2.circle(PalmveinImg, (Pref_row,Pref_col), 5, (0,0,255))
    # # 将点在原图中画出来并且显示
    for i in range(0, 9):
        cv2.circle(PalmveinImg, (int(point9[i][0]), int(point9[i][1])), 5, (255, 0, 0))
    cv2.imwrite('point9.jpg', PalmveinImg)

    # 判断左右手，右手：第一个点(0)和第三个点(2)的距离大于第七个点(6)和第九个点(8)
    # 若为右手，连接第四个点(3)和第八个点(7)作为基准线
    # 若为左手，连接第二个点(1)和第六个点(5)作为基准线
    # （其实旋转角度不需要画出直线，计算斜率得到倾斜角就可以进行旋转矫正）
    base_point = np.zeros((2, 2))
    if (distance(point9[0], point9[2]) > distance(point9[6], point9[8])):
        base_point[0] = point9[3]
        base_point[1] = point9[7]
    else:
        base_point[0] = point9[1]
        base_point[1] = point9[5]
    # cv2.circle(PalmveinImg, (int(base_point[0][0]),int(base_point[0][1])), 5, (255,0,0))
    # cv2.circle(PalmveinImg, (int(base_point[1][0]),int(base_point[1][1])), 5, (255,0,0))
    # cv2.imshow('point9',PalmveinImg)
    # cv2.waitKey(0)
    k = (base_point[0][1] - base_point[1][1]) / (base_point[0][0] - base_point[1][0])
    # print(k)
    angle = math.atan(k)
    # print(angle)

    ###############################按算出的angle旋转图片############################
    # rotation_img = cv2.GetRotationMatrix2D((Pref_row, Pref_col), angle, 1.0, PalmveinImg)
    rot_mat = cv2.getRotationMatrix2D((base_point[0][0], base_point[0][1]), angle * 180 / 3.14, 1.0)
    rotation_img = cv2.warpAffine(PalmveinImg, rot_mat, PalmveinImg.shape[1::-1], flags=cv2.INTER_LINEAR)

    # # 保存矫正图片
    # plt.subplot(1,2,1), plt.imshow(Original_img), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1,2,2), plt.imshow(rotation_img), plt.title('rotation_img'), plt.xticks([]), plt.yticks([])
    # plt.savefig('rotation_2072.jpg')

    # cv2.imshow('rotation_img',rotation_img)
    # cv2.waitKey(0)

    ##################################提取矩形ROI区域##############################
    # Point* squarepoint = new Point[2];
    # cv.rectangle(img_rotation_rectangle, (left_line, up_line), (right_line, down_line), (255, 255, 255), 2)
    Square_length = distance(base_point[0], base_point[1])
    # img_rotation[up_line:down_line,left_line:right_line]
    # print(base_point)
    # print(Square_length)
    img_rotation_ROI = rotation_img[int(base_point[0][1] + 10):int(base_point[0][1] + Square_length + 10),
                       int(base_point[0][0] - Square_length):int(base_point[0][0])]


    return img_rotation_ROI


#################################################################################################################

def getMaxRegion(imgOtsu, PalmveinImg):
    # /**********说明************
    # 这个函数是根据二值化图片找出面积最大对应的轮廓点坐标
    # *Input : binary Image, the return Image, maxarea_point, the totalnum of the maxarea_point
    # *Output : None
    # **************************/

    # /********伪代码**********
    # 提取轮廓
    # cv2.imshow('Maximum inscribed circle', imgOtsu)
    # cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(imgOtsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_area = 0
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        # print(area)
        if (area > max_area):
            max_area = area
    # print(max_area)

    # 标记轮廓
    cv2.drawContours(imgOtsu, contours, -1, (255, 0, 255), 3)

    size = imgOtsu.shape
    tempImg = np.zeros(imgOtsu.shape, np.uint8)

    i = 0
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if (area == max_area):
            # print(contours[i])
            # totalnum = contours[i].size()
            # for j in range(0, totalnum-1):
            #     maxarea_point[j] = contours[i][j]
            # cv2.drawContours(PalmveinImg, contours[i], -1, 255, 1)
            # cv2.imwrite("contour.jpg",tempImg)
            break
    max_contours = np.zeros((len(contours[i]), 2))

    for j in range(len(contours[i])):
        max_contours[j][0] = int(contours[i][j][0][0])
        max_contours[j][1] = int(contours[i][j][0][1])
        # print(max_contours[j][0])
        # print(max_contours[j][1])
    return max_contours


#################################################################################################################

###########找出手掌最高点和最低点，返回差值，后续按比例移动相应的距离#########
def count_size(img, rows, cols):
    up = -1
    down = -1
    for row in range(0, rows):
        for col in range(0, cols):
            # if(row == 00 and col > 300 and col < 800):
            #     print(img[row][col])
            if (img[row][col] == 255):
                up = row
                break
        if (up != -1):
            break
    for row in range(0, rows):
        for col in range(0, cols):
            if (img[rows - 1 - row][col] == 255):
                down = rows - 1 - row
                break
        if (down != -1):
            break
    # print(up, down)
    return down - up


#################################################################################################################

###########################找出手掌九个凹凸点############################
# 传入参数：
#       level：手腕高度线
#       points：轮廓点集
# 输出：
#       result：9个凹凸点
def find9point(level, points):
    left = 1300
    left_index = 0
    right = 0
    right_index = 0
    # 找到手掌最左和最右的点及对应下标
    for i in range(len(points)):
        if (points[i][1] < level and points[i][0] < left):
            left = points[i][0]
            left_index = i
        if (points[i][1] < level and points[i][0] > right):
            right = points[i][0]
            right_index = i

    # print(left,right)
    # 定义结果集和结果索引
    result = np.zeros((9, 2))
    result[0] = points[0]
    result_index = 1
    # 轮廓是逆时针，第一个点是最高点，遍历的时候从最高点往左，到达最左，再进行一次从最右点到中间最高点的遍历
    # 记录当前趋势是上升还是下降，1为上升，0为下降
    # 思路1：计算纵坐标差
    # 思路2：计算索引差，要记录上一个索引值
    # 思路3：结合思路1和2
    raise_flag = 0
    last_index = 0
    dis = 30
    for i in range(0, left_index + 1):
        if (i > 10 and points[i][1] < (level - 100) and points[i][1] > points[i - 10][1]):
            if (raise_flag == 1):
                if (result_index == 0 or abs(i - last_index) > 100 and abs(
                        points[i][1] - result[result_index - 1][1]) > dis):
                    # if(result_index==0 or abs(i - last_index) >150):
                    last_index = i
                    result[result_index] = points[i - 10]
                    result_index += 1
            raise_flag = 0
        elif (i > 10 and points[i][1] < (level - 100) and points[i][1] < points[i - 10][1]):
            if (raise_flag == 0):
                # if(result_index==0 or abs(points[i][0] - result[result_index-1][0]) >50):
                if (result_index == 0 or abs(i - last_index) > 100 and abs(
                        points[i][1] - result[result_index - 1][1]) > dis):
                    # if(result_index==0 or abs(i - last_index) >150):
                    last_index = i
                    result[result_index] = points[i - 10]
                    result_index += 1
            raise_flag = 1
        if (result_index == 9):
            break

    raise_flag = 1
    for i in range(right_index, len(points)):
        if (i > 10 and points[i][1] < (level - 100) and points[i][1] > points[i - 10][1]):
            if (raise_flag == 1):
                # if(result_index==0 or abs(points[i][0] - result[result_index-1][0]) >50):
                if (result_index == 0 or abs(i - last_index) > 100 and abs(
                        points[i][1] - result[result_index - 1][1]) > dis):
                    # if(result_index==0 or abs(i - last_index) >150):
                    last_index = i
                    result[result_index] = points[i - 10]
                    result_index += 1
            raise_flag = 0
        elif (i > 10 and points[i][1] < (level - 100) and points[i][1] < points[i - 10][1]):
            if (raise_flag == 0):
                # if(result_index==0 or abs(points[i][0] - result[result_index-1][0]) >50):
                if (result_index == 0 or abs(i - last_index) > 100 and abs(
                        points[i][1] - result[result_index - 1][1]) > dis):
                    # if(result_index==0 or abs(i - last_index) >150):
                    last_index = i
                    result[result_index] = points[i - 10]
                    result_index += 1
            raise_flag = 1
        if (result_index == 9):
            break

    if (result_index < 9):
        print("错误：未找齐9个点！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")

    return result


#################################################################################################################

#######################将9个点按从右到左的顺序排序########################
# 传入参数：
#       points：9个凹凸点
# 输出：
#       result：从右到左排序的9个凹凸点
def sortpoint(points):
    result = np.zeros((9, 2))
    j = 0
    i = 1
    # 找到左右换边的点
    while i < 9:
        if (points[i][0] > points[i - 1][0]):
            break
        i += 1
    mid = i
    while i < 9:
        result[j] = points[i]
        i += 1
        j += 1
    for i in range(0, mid):
        result[j] = points[i]
        j += 1
    return result


#################################################################################################################

###########################计算两点之间的距离############################
# 传入参数：
#       p1：点1
#       p2：点2
# 输出：
#       距离
def distance(p1, p2):
    return int(math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])))


#################################################################################################################

#################################开始运行###############################
# imgFile = '208_1.bmp'
imgFile = '209_2.bmp'

# # load an original image
img = cv2.imread(imgFile)

# # 获取掌心ROI区域
roi = getRoiImg(img)
cv2.imwrite('roi.jpg', roi)
