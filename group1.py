import math
import cv2
import imutils
import imageio
from scipy import ndimage
import numpy as np
import os


def picture_correction(picture_path, storage_path):
    # 读取图片
    img = cv2.imread(picture_path)

    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 进行canny边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 霍夫变换
    # rho = x cos (theta) + y sin (theta)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
    rotate_angle = 0
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if x1 == x2 or y1 == y2:
            continue
        t = float(y2 - y1) / (x2 - x1)
        rotate_angle = math.degrees(math.atan(t))
        if rotate_angle > 45:
            rotate_angle = -90 + rotate_angle
        elif rotate_angle < -45:
            rotate_angle = 90 + rotate_angle
    # print("rotate_angle : " + str(rotate_angle))
    rotate_img = ndimage.rotate(img, rotate_angle, mode='nearest')

    # 输出保存
    imageio.imsave(os.path.join(storage_path, f'{0}.jpg'), rotate_img)
    # cv2.imshow("img", rotate_img)
    cv2.waitKey(0)


def picture_clipping(picture_path, storage_path):
    # 边缘检测算法的选择通常要根据具体的应用场景和数据特征来确定，不同算法有不同的适用性。
    # 对于检测图片中的表格，Canny算法和Sobel算法都是常用的算法，因为它们对噪声比较鲁棒，能够较好地提取表格的边缘信息。
    # Canny算法相对于Sobel算法更加复杂，需要进行更多的参数调整。
    # 在实际应用中，可以先尝试使用Sobel算法，如果效果不理想再考虑使用Canny算法。
    # 同时，表格检测还需要考虑到表格的形状和结构特征，可以结合轮廓检测、形态学变换等技术来进一步提高检测的准确性

    # Sobel算法
    # 基于Sobel算法来检测图片中的表格可以按照以下步骤进行：
    # 读取图像并将其转换为灰度图像；
    # 对灰度图像进行Sobel边缘检测，得到水平方向和垂直方向上的梯度；
    # 计算梯度幅值和方向，然后应用非极大值抑制，以减少边缘的数量；
    # 根据双阈值处理策略，将强边缘和弱边缘分别标记为255和128；
    # 对标记为128的边缘进行连接，生成完整的边缘；
    # 对边缘进行形态学操作，例如膨胀、腐蚀和闭运算等，以填充和连接不完整的表格边缘；
    # 对表格边缘进行轮廓检测，找到表格的外轮廓；
    # 判断外轮廓的形状和大小是否符合表格的特征，例如矩形或近似矩形的形状、一定的长宽比等；
    # 如果外轮廓符合表格的特征，则认为该区域是表格，否则排除该区域。

    # 读取图像
    img = cv2.imread(picture_path)

    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯滤波，减少噪声
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 计算水平和垂直方向上的梯度,卷积结果
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

    # 计算梯度幅值和方向
    mag, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)

    # 非极大值抑制
    mag_max = np.max(mag)
    mag[mag < 0.1 * mag_max] = 0

    # 双阈值处理
    high_threshold = np.max(mag) * 0.2
    low_threshold = high_threshold * 0.5
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            if mag[i, j] > high_threshold:
                # 强边缘
                mag[i, j] = 255
            elif mag[i, j] < low_threshold:
                mag[i, j] = 0
            else:
                # 弱边缘
                mag[i, j] = 128

    # 连接弱边缘，对边缘进行形态学操作，例如膨胀、腐蚀等，以填充和连接不完整的表格边缘
    kernel = np.ones((3, 3), np.uint8)
    edge_map = cv2.dilate(mag, kernel, iterations=2)
    edge_map = cv2.erode(edge_map, kernel, iterations=2)

    # 将图像像素值类型转换为CV_8UC1
    gray_CV_8UC1 = cv2.convertScaleAbs(edge_map)

    # 找到轮廓
    contours, hierarchy = cv2.findContours(gray_CV_8UC1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤掉不符合条件的轮廓
    table_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if area < 1000:  # 面积太小的轮廓过滤掉
            continue
        approx = cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True) * 0.05, True)
        if not cv2.isContourConvex(approx):  # 非凸形状的轮廓过滤掉
            continue
        if len(approx) == 4 and area > 500 and w > 50 and h > 50:  # 选择四边形并且面接以及高和宽有要求
            table_contours.append(approx)

    # 绘制表格边缘
    cv2.drawContours(gray_CV_8UC1, table_contours, -1, (0, 0, 255), 1)

    # 切割表格区域
    for i, contour in enumerate(table_contours):
        # 计算轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        # 裁剪出表格区域
        table_roi = img[y:y + h, x:x + w]
        # 保存裁剪出的表格区域
        cv2.imwrite(os.path.join(storage_path, f'{i + 1}.jpg'), table_roi)


def merge_cols(lst_v, ran=5):
    """
    合并竖线
    :param lst_v: 读取到的所有竖线列表
    :param ran: 合并范围，默认=5
    :return: 合并之后的竖线
    """
    dic = {}  # {267:[514,95],794:[72,501]}
    for line in lst_v:
        flag = True
        start, end = line[0], line[1]  # start[0]与end[0]绝对相等
        for i in range(-1 * ran, ran + 1):  # 合并区间 正负5
            if start[0] + i in dic:
                dic[start[0] + i][0] = max(dic[start[0] + i][0], start[1])
                dic[start[0] + i][1] = min(dic[start[0] + i][1], end[1])
                flag = False
        if flag:
            dic[start[0]] = [start[1], end[1]]

    # 转化为列表
    lst_col = []
    for i in dic:
        lst_col.append(((i, dic[i][0]), (i, dic[i][1])))
    return lst_col


def merge_rows(lst_r, ran=8):
    """
    合并横线
    :param lst_r: 读取到的所有横线列表
    :param ran: 合并范围，默认=8
    :return: 合并之后的横线
    """
    dic = {}
    for line in lst_r:  # [((722, 191), (1320, 191)), ((584, 192), (1226, 192))]
        flag = True
        start, end = line[0], line[1]  # start[0]与end[0]绝对相等
        for i in range(-1 * ran, ran + 1):  # 合并区间 正负5
            if start[1] + i in dic:
                dic[start[1] + i][0] = min(dic[start[1] + i][0], start[0])
                dic[start[1] + i][1] = max(dic[start[1] + i][1], end[0])
                flag = False
        if flag:
            dic[start[1]] = [start[0], end[0]]

    # 转化为列表
    lst_row = []
    for i in dic:
        lst_row.append(((dic[i][0], i), (dic[i][1], i)))
    return lst_row


def photo2lst(file_name):
    lst_all = []
    image = cv2.imread(file_name)
    # 灰度图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
    rows, cols = binary.shape
    scale = 40

    # 识别横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedcol = cv2.dilate(eroded, kernel, iterations=1)
    linesRow = cv2.HoughLinesP(dilatedcol, 1, math.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # 遍历每条直线并获取起点和终点坐标
    lst = []
    for line in linesRow:
        x1, y1, x2, y2 = line[0]
        start_point = (x1, y1)
        end_point = (x2, y2)

        # 将起点和终点坐标保存为元组
        line_coordinates = (start_point, end_point)
        lst.append(line_coordinates)

    lstRows = merge_rows(lst)
    lst_all.append(lstRows)

    # 识别竖线
    scale = 20
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)
    linesCol = cv2.HoughLinesP(dilatedrow, 1, math.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    lst2 = []
    for line in linesCol:
        x1, y1, x2, y2 = line[0]
        start_point = (x1, y1)
        end_point = (x2, y2)

        # 将起点和终点坐标保存为元组
        line_coordinates = (start_point, end_point)
        lst2.append(line_coordinates)
    # 合并竖线
    lst2 = merge_cols(lst2)
    lst_all.append(lst2)

    return lst_all


# 通过横线和竖线集合，获得交点集
def getDotSet(horizontal_lineSeg, vertical_lineSeg):
    '''
    假设 有边界，标准。
    集合内为元组，元组表示坐标，最长的线段起止位置。
    形如  横：{((0,0),(3,0)),((0,1),(3，1)),...} 竖：{...}
    '''
    dotSet = set()
    for i in horizontal_lineSeg:
        for j in vertical_lineSeg:
            dot = findIntersection(i, j)
            if (dot != None):
                dotSet.add(dot)
    return dotSet


# 计算交点坐标
def findIntersection(address1, address2):
    x1 = address1[0][0]
    x2 = address1[1][0]
    x3 = address2[0][0]
    x4 = address2[1][0]
    y1 = address1[0][1]
    y2 = address1[1][1]
    y3 = address2[0][1]
    y4 = address2[1][1]
    if (x1 > x2):
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    if (x3 > x4):
        x3, x4 = x4, x3
        y3, y4 = y4, y3
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

    def judge(px):
        if (px < x1 - 2 or px < x3 - 2 or px > x2 + 2 or px > x4 + 2):
            return False
        else:
            return True

    if (judge(px)):
        return (px, py)
    else:
        return None


# 从左上开始，每个交点，找到右下的、距离最近的其他三个交点（仅考虑右下方向），即构成一个矩形，创建一个新结点
def formRectangle(dotSet):
    nodeLst = []
    dotSet = sorted(list(dotSet))
    dotSet_copy = dotSet[:]
    # 排序后找纵坐标相同的结点比较容易，like[(0, 0), (0, 1), (1, 0), (1, 1.1)]
    # 相当于左下方的节点已经找到
    for i in range(len(dotSet)):
        # 第一版本假设纵坐标严格相等，第二版加入近似的判断
        # 排序后下方交点为i+1
        if (similar(dotSet[i][0], dotSet[i + 1][0])):  # 加入近似判断
            temp = []
            temp.append(dotSet[i])
            temp.append(dotSet[i + 1])
            if (i == 0):
                dotSet_copy.remove(dotSet[i])
                dotSet_copy.remove(dotSet[i + 1])
            else:
                dotSet_copy.remove(dotSet[i + 1])
            for j in dotSet_copy:
                if (similar(j[1], dotSet[i + 1][1])):
                    temp.append(j)
                    break
            if (len(temp) == 2):
                break
            temp.append((temp[2][0], temp[0][1]))  # 最后一个交点 计算得出
            # 新键结点
            newVertex = Vertex(temp[0], temp[1], temp[2], temp[3])
            newVertex.setName(str(len(nodeLst) + 1))
            nodeLst.append(newVertex)
        # 其他情况，不相等则为最后的结点
    return nodeLst

    # 每个矩形为一个结点，找到有公共顶点的其他结点（仅找右下一圈的，形成图结构


def formGraph(nodelst):
    for i in range(len(nodelst)):
        lst1 = nodelst[i].getNode()
        ll = lst1[1]
        lr = lst1[2]
        ur = lst1[3]
        for j in range(i + 1, len(nodelst)):
            # 左上交点
            l2 = nodelst[j].getNode()[0]
            if (l2[0] == lr[0] and ur[1] <= l2[1] <= lr[1]) or (
                    l2[1] == ll[1] and ll[0] <= l2[0] <= lr[0]):
                if (nodelst[j] not in nodelst[i].getConnections()):
                    nodelst[i].addNeighbor(nodelst[j])
    return nodelst


def similar(a, b):
    if b - 2 <= a <= b + 2:
        return True
    else:
        return False


# 异形则补充交点，变为标准表格，将补充的交点单独存放，重复标准化表格的操作
# 根据补充的交点合并一些结点，形成异形表格的图结构
# 有边界表格，无边界表格
# 标准，异形

class Vertex:
    # 存放右方，下方的结点
    connectedTo = []
    name = ''

    def __init__(self, up_left, low_left, low_right, up_right):  # 初始化，四个顶点坐标形如（2.3，4.5）
        self.ul = up_left
        self.ur = up_right
        self.lr = low_right
        self.ll = low_left
        self.visitied = False
        self.connectedTo = []

    def setVisited(self):
        self.visitied = True

    def setName(self, nm):
        self.name = nm

    def getName(self):
        return self.name

    # 加入右方、下方结点
    def addNeighbor(self, neighbor):
        self.connectedTo.append(neighbor)

    # 返回邻接表中的所有顶点
    def getConnections(self):
        return self.connectedTo

    # 返回顶点坐标
    def getNode(self):
        return self.ul, self.ll, self.lr, self.ur

    # 返回矩形两边的长度，返回顺序为垂直边，水平边
    def getLen(self):
        len_left = math.sqrt(math.pow((self.ul[0] - self.ll[0]), 2) + math.pow((self.ul[1] - self.ll[1]), 2))
        len_right = math.sqrt(math.pow((self.ur[0] - self.lr[0]), 2) + math.pow((self.ur[1] - self.lr[1]), 2))
        len1 = (len_left + len_right) / 2
        len_up = math.sqrt(math.pow((self.ur[0] - self.ul[0]), 2) + math.pow((self.ur[1] - self.ul[1]), 2))
        len_low = math.sqrt(math.pow((self.lr[0] - self.ll[0]), 2) + math.pow((self.lr[1] - self.ll[1]), 2))
        len2 = (len_up + len_low) / 2
        return len1, len2


class Graph:
    def __init__(self, nodelst):  # 属性包括一个表格里的结点list和结点数
        self.vertList = nodelst
        self.numVertices = len(nodelst)

    # 新加结点
    def addVertex(self, vertex):
        self.numVertices = self.numVertices + 1
        self.vertList.append(vertex)

    # 获取单元格列表
    def getVertList(self):
        return self.vertList

    # 获取单元格数量
    def getNum(self):
        return self.numVertices


if __name__ == "__main__":
    picture_correction('D:\pythonProject/region segmentation/output.png', 'D:\pythonProject/region segmentation/images')
    picture_clipping("D:\pythonProject/region segmentation/images/0.jpg", "D:\pythonProject/region segmentation/images")
    LSTLines = photo2lst("D:\pythonProject/region segmentation/images/1.jpg")
    horizontal_lineSegment = set(LSTLines[0])  # 9
    vertical_lineSegment = set(LSTLines[1])  # 6

    nodelst = formGraph(formRectangle(getDotSet(horizontal_lineSegment, vertical_lineSegment)))
    gragh = Graph(nodelst)
    print("矩形：" + (str)(gragh.getVertList()))
    print("矩形数量：" + (str)(gragh.getNum()))
