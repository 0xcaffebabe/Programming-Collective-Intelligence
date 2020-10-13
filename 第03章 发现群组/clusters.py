from PIL import Image, ImageDraw
from math import sqrt


# read blog data
def readfile(filename):
    lines = [line for line in open(filename)]

    # First line is the column titles
    colnames = lines[0].strip().split('\t')[1:]
    rownames = []
    data = []
    for line in lines[1:]:
        p = line.strip().split('\t')
        # First column in each row is the rowname
        rownames.append(p[0])
        # The data for this row is the remainder of the row
        data.append([float(x) for x in p[1:]])
    return rownames, colnames, data


def pearson(v1, v2):
    # 简单求和
    sum1 = sum(v1)
    sum2 = sum(v2)

    # 平方根和
    sum1Sq = sum([pow(v, 2) for v in v1])
    sum2Sq = sum([pow(v, 2) for v in v2])

    # 乘积之和
    pSum = sum([v1[i] * v2[i] for i in range(len(v1))])

    # Calculate r (Pearson score)
    num = pSum - (sum1 * sum2 / len(v1))
    den = sqrt((sum1Sq - pow(sum1, 2) / len(v1)) * (sum2Sq - pow(sum2, 2) / len(v1)))
    if den == 0: return 0

    return 1.0 - num / den


class bicluster:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance


def hcluster(rows, distance=pearson):
    distances = {}
    currentclustid = -1

    # 最初的顶层聚类等于数据集中的行
    clust = [bicluster(rows[i], id=i) for i in range(len(rows))]

    # 不断聚合 直到只剩下一个顶层聚类
    while len(clust) > 1:
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        # 遍历每个配对 寻找最小距离
        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                # 缓存距离
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)

                d = distances[(clust[i].id, clust[j].id)]

                if d < closest:
                    closest = d
                    lowestpair = (i, j)

        # 计算两个聚类的平均值
        mergevec = [
            (clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i]) / 2.0
            for i in range(len(clust[0].vec))]

        # 创建新聚类
        newcluster = bicluster(mergevec, left=clust[lowestpair[0]],
                               right=clust[lowestpair[1]],
                               distance=closest, id=currentclustid)

        # 不在原始集合中的聚类 其id为负数
        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)

    return clust[0]


def printclust(clust, labels=None, n=0):
    # indent to make a hierarchy layout
    for i in range(n):
        print(' ', end='')
    if clust.id < 0:
        # negative id means that this is branch
        print('-')
    else:
        # positive id means that this is an endpoint
        if labels is None:
            print(clust.id)
        else:
            print(labels[clust.id])

    # now print the right and left branches
    if clust.left is not None:
        printclust(clust.left, labels=labels, n=n + 1)
    if clust.right is not None:
        printclust(clust.right, labels=labels, n=n + 1)

blognames, words, data = readfile('./data/blogdata.txt')
clust = hcluster(data)
# printclust(clust, labels=blognames)

# Drawing the Dendrogram
def getheight(clust):
    # Is this an endpoint? Then the height is just 1
    if clust.left is None and clust.right is None:
        return 1

    # Otherwise the height is the same of the heights of
    # each branch
    return getheight(clust.left) + getheight(clust.right)


def getdepth(clust):
    # The distance of an endpoint is 0.0
    if clust.left is None and clust.right is None:
        return 0

    # The distance of a branch is the greater of its two sides
    # plus its own distance
    return max(getdepth(clust.left), getdepth(clust.right)) + clust.distance


def drawdendrogram(clust, labels, jpeg='output/clusters.jpg'):
    # height and width
    h = getheight(clust) * 20
    w = 1200
    depth = getdepth(clust)

    # width is fixed, so scale distances accordingly
    scaling = float(w - 150) / depth

    # Create a new image with a white background
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.line((0, h / 2, 10, h / 2), fill=(255, 0, 0))

    # Draw the first node
    drawnode(draw, clust, 10, (h / 2), scaling, labels)
    img.save(jpeg, 'JPEG')


def drawnode(draw, clust, x, y, scaling, labels):
    if clust.id < 0:
        h1 = getheight(clust.left) * 20
        h2 = getheight(clust.right) * 20
        top = y - (h1 + h2) / 2
        bottom = y + (h1 + h2) / 2
        # Line length
        ll = clust.distance * scaling
        # Vertical line from this cluster to children
        draw.line((x, top + h1 / 2, x, bottom - h2 / 2), fill=(255, 0, 0))

        # Horizontal line to left item
        draw.line((x, top + h1 / 2, x + ll, top + h1 / 2), fill=(255, 0, 0))

        # Horizontal line to right item
        draw.line((x, bottom - h2 / 2, x + ll, bottom - h2 / 2), fill=(255, 0, 0))

        # Call the function to draw the left and right nodes
        drawnode(draw, clust.left, x + ll, top + h1 / 2, scaling, labels)
        drawnode(draw, clust.right, x + ll, bottom - h2 / 2, scaling, labels)
    else:
        # If this is an endpoint, draw the item label
        draw.text((x + 5, y - 7), labels[clust.id], (0, 0, 0))

# drawdendrogram(clust, blognames, jpeg='blogclust.jpg')

# Column Clustering
def rotatematrix(data):
    newdata = []
    for i in range(len(data[0])):
        newrow = [data[j][i] for j in range(len(data))]
        newdata.append(newrow)
    return newdata

rdata=rotatematrix(data)
wordclust=hcluster(rdata)
#drawdendrogram(wordclust, labels=words, jpeg='wordclust.jpg')

import random


# K-Means Clustering
def kcluster(rows, distance=pearson, k=4):
    # 确定每个点的最大最小值
    ranges = [(min([row[i] for row in rows]), max([row[i] for row in rows]))
              for i in range(len(rows[0]))]

    # 随机创建k个中心点
    clusters = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                 for i in range(len(rows[0]))] for j in range(k)]

    lastmatches = None

    for t in range(100):
        dis_sum = 0
        print('Iteration %d' % t)
        bestmatches = [[] for i in range(k)]

        # 在每一行中寻找距离最近的中心点
        for j in range(len(rows)):
            row = rows[j]
            bestmatch = 0
            for i in range(k):
                d = distance(clusters[i], row)
                if d < distance(clusters[bestmatch], row):
                    bestmatch = i
            dis_sum += d
            bestmatches[bestmatch].append(j)

        # 如果与上次结果相同 则结束
        if bestmatches == lastmatches:
            break
        lastmatches = bestmatches

        # 把中心移到其所有成员的平均位置处
        for i in range(k):
            avgs = [0.0] * len(rows[0])
            if len(bestmatches[i]) > 0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m] += rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i] = avgs

    return bestmatches, dis_sum

# kclust = kcluster(data,k=10)
# print(kclust)
# Clusters of Preference
def tanamoto(v1, v2):
    c1, c2, shr = 0, 0, 0

    for i in range(len(v1)):
        if v1[i] != 0: c1 += 1  # in v1
        if v2[i] != 0: c2 += 1  # in v2
        if v1[i] != 0 and v2[i] != 0: shr += 1  # in both

    return 1.0 - (float(shr) / (c1 + c2 - shr))


# ex04
def manhattan(v1, v2):
    dis = 0

    for i in range(len(v1)):
        dis += abs(v1[i] - v2[i])

    return dis


# Viewing Data in Two Dimensions
def scaledown(data, distance=pearson, rate=0.01):
    n = len(data)

    # 每一对数据项之间的距离
    realdist = [[distance(data[i], data[j]) for j in range(n)]
                for i in range(0, n)]

    # 随机初始化节点再二维空间的起始位置
    loc = [[random.random(), random.random()] for i in range(n)]
    fakedist = [[0.0 for j in range(n)] for i in range(n)]

    lasterror = None
    for m in range(0, 1000):
        # 寻找投影后的距离
        for i in range(n):
            for j in range(n):
                fakedist[i][j] = sqrt(sum([pow(loc[i][x] - loc[j][x], 2)
                                           for x in range(len(loc[i]))]))

        # 移动节点
        grad = [[0.0, 0.0] for i in range(n)]

        totalerror = 0
        for k in range(n):
            for j in range(n):
                if j == k: continue
                # The error is percent difference between the distances
                errorterm = (fakedist[j][k] - realdist[j][k]) / realdist[j][k]

                # Each point needs to be moved away from or towards the other
                # point in proportion to how much error it has
                grad[k][0] += ((loc[k][0] - loc[j][0]) / fakedist[j][k]) * errorterm
                grad[k][1] += ((loc[k][1] - loc[j][1]) / fakedist[j][k]) * errorterm

                # Keep track of the total error
                totalerror += abs(errorterm)
        print(totalerror)

        # If the answer got worse by moving the points, we are done
        if lasterror and lasterror < totalerror: break
        lasterror = totalerror

        # Move each of the points by the learning rate times the gradient
        for k in range(n):
            loc[k][0] -= rate * grad[k][0]
            loc[k][1] -= rate * grad[k][1]

    return loc


def draw2d(data, labels, jpeg='mds2d.jpg'):
    img = Image.new('RGB', (2000, 2000), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(len(data)):
        x = (data[i][0] + 0.5) * 1000
        y = (data[i][1] + 0.5) * 1000
        draw.text((x, y), labels[i], (0, 0, 0))
    img.save(jpeg, 'JPEG')

coords = scaledown(data)
draw2d(coords, blognames, jpeg="blogs2d.jpg")