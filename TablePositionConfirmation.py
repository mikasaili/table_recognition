import math
import matplotlib.pyplot as plt
'''
单元格识别与位置关系确认
输入：线段集合
输出：表述单元格位置的图结构
'''
#通过横线和竖线集合，获得交点集
def getDotSet(horizontal_lineSeg,vertical_lineSeg):
    '''
    假设 有边界，标准。
    集合内为元组，元组表示坐标，最长的线段起止位置。
    形如  横：{((0,0),(3,0)),((0,1),(3，1)),...} 竖：{...}
    '''
    dotSet=set()
    for i in horizontal_lineSeg:
        for j in vertical_lineSeg:
            dot=findIntersection(i,j)
            if(dot!=None):
                dotSet.add(dot)
    return dotSet

#计算交点坐标
def findIntersection(address1,address2):
    x1=address1[0][0]
    x2=address1[1][0]
    x3 = address2[0][0]
    x4= address2[1][0]
    y1 = address1[0][1]
    y2 = address1[1][1]
    y3 = address2[0][1]
    y4 = address2[1][1]
    if(x1>x2):
        x1,x2=x2,x1
        y1,y2=y2,y1
    if(x3>x4):
        x3,x4=x4,x3
        y3,y4=y4,y3
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

    def judge(px,py,x1,x2,x3,x4,y1,y2,y3,y4):
        if(px<x1-2 or px<x3-2 or px>x2+2 or px>x4+2):
            return False
        if similar(y1,y2):
            if y3>y4:
                y3,y4=y4,y3
            if py<y3-2 or py>y4+2:
                return False
        if similar(y3,y4):
            if y1>y2:
                y1,y2=y2,y1
            if py<y1-2 or py>y2+2:
                return False
        return True
    if(judge(px,py,x1,x2,x3,x4,y1,y2,y3,y4)):
        return (px, py)
    else:return None

#从左上开始，每个交点，找到右下的、距离最近的其他三个交点（仅考虑右下方向），即构成一个矩形，创建一个新结点
def formRectangle(dotSet):
    nodeLst = []
    dotSet = sorted(list(dotSet))
    dotSet_copy = dotSet[:]
    # 排序后找纵坐标相同的结点比较容易，like[(0, 0), (0, 1), (1, 0), (1, 1.1)]
    # 相当于左下方的节点已经找到
    for i in range(len(dotSet)):
        # 第一版本假设横坐标严格相等，第二版加入近似的判断
        # 排序后下方交点为i+1
        if (similar(dotSet[i][0],dotSet[i+1][0])):#加入近似判断
            temp = []
            temp.append(dotSet[i])
            temp.append(dotSet[i + 1])
            if (i == 0):
                dotSet_copy.remove(dotSet[i])
                dotSet_copy.remove(dotSet[i + 1])
            else:
                dotSet_copy.remove(dotSet[i + 1])
            for j in dotSet_copy:
                if (similar(j[1],dotSet[i + 1][1])) and j[0]>dotSet[i + 1][0]:
                    temp.append(j)
                    break
            if (len(temp) == 2):
                break
            #异形情况下，合并横着的单元格会有影响
            # 还需判断此坐标是否存在。
            if (temp[2][0], temp[0][1]) not in dotSet:
                lst=recheck(temp[2],dotSet_copy)
                for k in lst:
                    if (k[0],temp[0][1]) in dotSet:
                        temp.remove(temp[2])
                        temp.append(k)
                        break
            temp.append((temp[2][0], temp[0][1]))  # 最后一个交点 计算得出
            # 新键结点
            newVertex = Vertex(temp[0], temp[1], temp[2], temp[3])
            newVertex.setName(str(len(nodeLst)+1))
            nodeLst.append(newVertex)
        # 其他情况，不相等则为最后的结点
    return nodeLst

def recheck(dot,dotSet):
    lst=[]
    for i in dotSet:
        if i[1]==dot[1]:
            lst.append(i)
    return lst

    #每个矩形为一个结点，找到有公共顶点的其他结点（仅找右下一圈的，形成图结构
def formGraph(nodelst):
    for i in range(len(nodelst)):
        lst1=nodelst[i].getNode()
        ll=lst1[1]
        lr=lst1[2]
        ur=lst1[3]
        for j in range(i+1,len(nodelst)):
            #左上交点
            l2=nodelst[j].getNode()[0]
            if(l2[0]==lr[0] and l2[1]>=ur[1] and l2[1]<=lr[1]) or (l2[1] == ll[1] and l2[0] >= ll[0] and l2[0] <= lr[0]):
                if(nodelst[j] not in nodelst[i].getConnections()):
                    nodelst[i].addNeighbor(nodelst[j])
    return nodelst

def similar(a,b):
    if b-2<=a and b+2>=a:
        return True
    else:
        return False
#异形则补充交点，变为标准表格，将补充的交点单独存放，重复标准化表格的操作
#根据补充的交点合并一些结点，形成异形表格的图结构
#有边界表格，无边界表格
#标准，异形

class Vertex:
    # 存放右方，下方的结点
    connectedTo = []
    name=''
    def __init__(self, up_left, low_left,  low_right,up_right):  # 初始化，四个顶点坐标形如（2.3，4.5）
        self.ul = up_left
        self.ur = up_right
        self.lr = low_right
        self.ll = low_left
        self.visitied = False
        self.connectedTo = []

    def setVisited(self):
        self.visitied = True

    def setName(self,nm):
        self.name=nm

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

    # 返回中心点坐标
    def getCenter(self):
        return ((self.ul[0]+self.ur[0])/2, (self.lr[1]+self.ur[1])/2)

class Graph:
    def __init__(self,nodelst):#属性包括一个表格里的结点list和结点数
        self.vertList = nodelst
        self.numVertices = len(nodelst)

    #新加结点
    def addVertex(self,vertex):
        self.numVertices = self.numVertices + 1
        self.vertList.append(vertex)

    #获取单元格列表
    def getVertList(self):
        return self.vertList

    #获取单元格数量
    def getNum(self):
        return self.numVertices

    #中心坐标集
    def dotCenter(self):
        ls = []  # 中心坐标
        for i in range(len(self.vertList)):
            ls.append(self.vertList[i].getCenter())
        return ls

    #左上坐标集
    def dotLeft(self):
        # 矩形左上角坐标
        leftdot=[]
        print("左上角坐标；")
        for i in range(len(self.vertList)):
            leftdot.append(self.vertList[i].getNode()[0])
        print(leftdot)
        return leftdot

    #显示图像
    def huitu(self):
        ax = plt.gca()
        ls=self.dotCenter()
        dotset=sorted(list(self.getDotset()))

        for i in self.vertList:
            fourdot=i.getNode()
            x_addr = []
            y_addr = []
            x_addr.append(fourdot[0][0])
            x_addr.append(fourdot[1][0])
            x_addr.append(fourdot[2][0])
            x_addr.append(fourdot[3][0])
            x_addr.append(fourdot[0][0])

            y_addr.append(fourdot[0][1])
            y_addr.append(fourdot[1][1])
            y_addr.append(fourdot[2][1])
            y_addr.append(fourdot[3][1])
            y_addr.append(fourdot[0][1])
            plt.plot(x_addr, y_addr, linewidth=1, color='grey')  # 点依次连成线

        for i in range(len(ls)):
            plt.scatter(ls[i][0], ls[i][1], c='b')
        ax.xaxis.set_ticks_position('top')  # 将x轴的位置设置在顶部
        plt.xlim(-5, dotset[-1][0]+5)  # x坐标轴范围
        plt.ylim(dotset[-1][1]+5, -5)
        plt.show()
        return

    def getDotset(self):
        dotset=set()
        for i in nodelst:
            fourDot=i.getNode()
            dotset.add(fourDot[0])
            dotset.add(fourDot[1])
            dotset.add(fourDot[2])
            dotset.add(fourDot[3])
        return dotset

if __name__ == '__main__' :
    lstLine =[[((3, 76), (417, 76)), ((4, 174), (417, 174)), ((3, 125), (417, 125)), ((3, 2), (417, 2)), ((3, 51), (417, 51)), ((3, 27), (417, 27)), ((4, 149), (417, 149)), ((3, 100), (417, 100))], [((209, 176), (209, 3)), ((415, 174), (415, 2)), ((3, 176), (3, 4)), ((297, 175), (297, 10)), ((91, 176), (91, 28))]]
    horizontal_lineSegment = set(lstLine[0])
    vertical_lineSegment = set(lstLine[1])

    dotset=sorted(getDotSet(horizontal_lineSegment, vertical_lineSegment))#交点集
    nodelst=formGraph(formRectangle(dotset))#结点集
    gragh=Graph(nodelst)#图结构

    gragh.dotLeft()
    gragh.huitu()



