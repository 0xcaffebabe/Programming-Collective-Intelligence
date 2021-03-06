# 一个涉及影评者及其对几部影片评分情况的字典
critics = {'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
                         'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
                         'The Night Listener': 3.0},
           'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
                            'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
                            'You, Me and Dupree': 3.5},
           'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
                                'Superman Returns': 3.5, 'The Night Listener': 4.0},
           'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
                            'The Night Listener': 4.5, 'Superman Returns': 4.0,
                            'You, Me and Dupree': 2.5},
           'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                            'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
                            'You, Me and Dupree': 2.0},
           'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                             'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
           'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0}}

from math import sqrt


# 返回一个有关person1与person2的基于距离的相似度评价
def sim_distance(prefs, person1, person2):
    # 得到shared_items的列表
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1

    # 如果两者没有共同之处，则返回0
    if len(si) == 0:
        return 0

    # 计算所有差值的平方和
    sum_of_squares = 0
    for item in si.keys():
        # 这里所做的就是等同于在计算二维中x y 的差值
        sum_of_squares = sum_of_squares + pow(prefs[person1][item] - prefs[person2][item], 2)

    return 1 / (1 + sqrt(sum_of_squares))

# print(sim_distance(critics, 'Lisa Rose', 'Gene Seymour'))

# 返回p1和p2的皮尔逊相关系数
def sim_pearson(prefs, p1, p2):
    # 得到双方都曾评价过的物品列表
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1

    # 如果两者没有共同之处，则返回0
    if len(si) == 0:
        return 0

    # 得到列表元素的个数
    n = len(si)

    # 对所有偏好求和
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])

    # 求平方和
    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])

    # 求乘积之和
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])

    # 计算皮尔逊评价值
    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    if den == 0:
        return 0

    r = num / den

    return r

#print(sim_pearson(critics, 'Lisa Rose', 'Gene Seymour'))

# 从反映偏好的词典中返回最为匹配者
# 返回的个数以及相似度算法都是可选的
def topMatches(prefs, person, n=5, similarity=sim_pearson):
    scores = [(similarity(prefs, person, other), other)
              for other in prefs if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]

# print(topMatches(critics, "Toby"))

# 利用他人评价值的就加权平均 获取推荐
def getRecommendations(prefs, person, similarity=sim_pearson):
    totals = {}
    simSums = {}
    for other in prefs:
        # 不和自己比较
        if other == person:
            continue
        sim = similarity(prefs, person, other)

        # 忽略相似度小于等于0的用户
        if sim <= 0: continue
        for item in prefs[other]:
            # 只对自己还未评价过的电影推荐
            if item not in prefs[person] or prefs[person][item] == 0:
                # 计算某部影片的分值
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                # 每部影片的评价人相似度总和
                simSums.setdefault(item, 0)
                simSums[item] += sim

    # 计算一个归一化列表
    rankings = [(total / simSums[item], item) for item, total in totals.items()]

    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings

# print(getRecommendations(critics, 'Toby'))

def transformPrefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})

            # Flip item and person
            result[item][person] = prefs[person][item]
    return result

# movies = transformPrefs(critics)
# print(topMatches(movies, 'Superman Returns'))

def calculateSimilarItems(prefs, n=10):
    # 建立一个key为物品 value为与其相似相近物品的的字典
    result = {}

    # 反转物品与人
    itemPrefs = transformPrefs(prefs)
    c = 0
    for item in itemPrefs:
        # 针对大数据更新状态变量
        c += 1
        if c % 100 == 0:
            print("%d / %d" % (c, len(itemPrefs)))
        # 寻找最为相近的物品
        scores = topMatches(itemPrefs, item, n=n, similarity=sim_distance)
        result[item] = scores
    return result

# print(calculateSimilarItems(critics))

def getRecommendedItems(prefs, itemMatch, user):
    userRatings = prefs[user]
    scores = {}
    totalSim = {}
    # 遍历当前用户评价过的物品
    for (item, rating) in userRatings.items():

        # 遍历与当前物品相近的物品
        for (similarity, item2) in itemMatch[item]:

            # 忽略已经被评价过的物品
            if item2 in userRatings: continue
            # 分数是通过相似度*评分
            scores.setdefault(item2, 0)
            scores[item2] += similarity * rating
            # Sum of all the similarities
            totalSim.setdefault(item2, 0)
            totalSim[item2] += similarity

    # Divide each total score by total weighting to get an average
    rankings = [(score / totalSim[item], item) for item, score in scores.items()]

    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    return rankings

# print(getRecommendedItems(critics, calculateSimilarItems(critics),'Toby'))

def loadMovieLens(path='/data/movielens'):
    # Get movie titles
    movies = {}
    for line in open(path + '/u.item'):
        (id, title) = line.split('|')[0:2]
        movies[id] = title

    # Load data
    prefs = {}
    for line in open(path + '/u.data'):
        (user, movieid, rating, ts) = line.split('\t')
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)
    return prefs
