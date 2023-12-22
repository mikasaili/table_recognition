import cv2
import math
import imageio
import numpy as np
from scipy import ndimage
import numpy as np
import os

def picture_correction(picture_path,storage_path):
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

def picture_clipping(picture_path,storage_path):
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
                # 连接强边缘
                # if (mag[i-1, j-1] > high_threshold
                #         or mag[i-1, j] > high_threshold
                #         or mag[i-1, j+1] > high_threshold
                #         or mag[i, j-1] > high_threshold
                #         or mag[i, j+1] > high_threshold
                #         or mag[i+1, j-1] > high_threshold
                #         or mag[i+1, j] > high_threshold
                #         or mag[i+1, j+1] > high_threshold):
                #     mag[i, j] = 255
                # else:
                #     mag[i, j] = 0

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
        if len(approx) == 4 and area > 500 and w > 50 and h > 50 :  # 选择四边形并且面接以及高和宽有要求
            table_contours.append(approx)
        # if area > 1000 and w > 50 and h > 50:
        #     # and 0.5 < w / h < 2:
        #     # cv2.drawContours(img, table_contours, -1, (0, 0, 255), 2)
        #     table_contours.append(cnt)
        #     # table_contours.append((x,y,w,h))

    # # 遍历轮廓，筛选符合条件的轮廓，并进行图像切割
    # for i, contour in enumerate(contours):
    #     area = cv2.contourArea(contour)
    #     if area < 500:
    #         continue
    #     x, y, w, h = cv2.boundingRect(contour)
    #     if w < 50 or h < 50:
    #         continue
    #     if w > h:
    #         y1 = y
    #         y2 = y + w
    #         x1 = x
    #         x2 = x + w
    #     else:
    #         y1 = y
    #         y2 = y + h
    #         x1 = x
    #         x2 = x + h
    #     roi = img[y1:y2, x1:x2]
    #     cv2.imwrite(os.path.join('D:\\pythonProject\\region segmentation\\images', f'{i}.jpg'), roi)

    # 绘制表格边缘
    cv2.drawContours(gray_CV_8UC1, table_contours, -1, (0, 0, 255), 1)

    # # 创建掩膜
    # mask = np.zeros_like(gray_CV_8UC1)
    #
    # # 绘制表格轮廓
    # cv2.drawContours(mask, table_contours, -1, 255, -1)
    #
    # # 对比原始图像和掩膜
    # masked_img = cv2.bitwise_and(img, img, mask=mask)

    # 切割表格区域
    for i, contour in enumerate(table_contours):
        # 计算轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        # 裁剪出表格区域
        table_roi = img[y:y + h, x:x + w]
        # 保存裁剪出的表格区域
        cv2.imwrite(os.path.join(storage_path, f'{i}.jpg'), table_roi)

    # 显示结果
    # cv2.imshow("Original Image", img)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Masked Image", masked_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("Original Image", img)
    # cv2.imshow("Sobel Edge Detection", mag.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    picture_correction('D:/pythonProject/region segmentation/output02.png','D:/pythonProject/region segmentation')
    picture_clipping('D:/pythonProject/region segmentation/0.jpg','D:/pythonProject/region segmentation/images')