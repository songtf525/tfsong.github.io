from __future__ import print_function
import operator
from math import  log

from collections import Counter


# 定义数据集
def createDataSet():
    dataSet = [[1,1,"yes"],
               [1,1,"yes"],
               [1,0,"no"],
               [0,1,"no"],
               [0,1,"no"]]

    labels = ["no surfacing", "flippers"]
    return dataSet, labels


# # 划分数据集
# def splitDataSet(dataSet, index, value):
#     """splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
#         就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中
#     Args:
#         dataSet 数据集                 待划分的数据集
#         index 表示每一行的index列        划分数据集的特征
#         value 表示index列对应的value值   需要返回的特征的值。
#     Returns:
#         index列为value的数据集【该数据集需要排除index列】
#     """
#     # -----------切分数据集的第一种方式 start------------------------------------
#     retDataSet = []
#     for featVec in dataSet: 
#         # index列为value的数据集【该数据集需要排除index列】
#         # 判断index列的值是否为value
#         if featVec[index] == value:
#             # chop out index used for splitting
#             # [:index]表示前index行，即若 index 为2，就是取 featVec 的前 index 行
#             reducedFeatVec = featVec[:index]
#             '''
#             请百度查询一下： extend和append的区别
#             list.append(object) 向列表中添加一个对象object
#             list.extend(sequence) 把一个序列seq的内容添加到列表中
#             1、使用append的时候，是将new_media看作一个对象，整体打包添加到music_media对象中。
#             2、使用extend的时候，是将new_media看作一个序列，将这个序列和music_media序列合并，并放在其后面。
#             result = []
#             result.extend([1,2,3])
#             print result
#             result.append([4,5,6])
#             print result
#             result.extend([7,8,9])
#             print result
#             结果：
#             [1, 2, 3]
#             [1, 2, 3, [4, 5, 6]]
#             [1, 2, 3, [4, 5, 6], 7, 8, 9]
#             '''
#             reducedFeatVec.extend(featVec[index+1:])
#             # [index+1:]表示从跳过 index 的 index+1行，取接下来的数据
#             # 收集结果值 index列为value的行【该行需要排除index列】
#             retDataSet.append(reducedFeatVec)
#     # -----------切分数据集的第一种方式 end------------------------------------

#     # # -----------切分数据集的第二种方式 start------------------------------------
#     # retDataSet = [data for data in dataSet for i, v in enumerate(data) if i == axis and v == value]
#     # # -----------切分数据集的第二种方式 end------------------------------------
#     return retDataSet


# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    # 求list的长度，计算参与训练的数据量
    numEntries = len(dataSet)

    # 计算分类标签label出现的次数
    labelCounts = {}
    for featVec in dataSet:
        # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
        currentLabel = featVec[-1]
        # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典
        # 并将当前键值加入字典，每个键值都记录了当前类别出现的次数
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # 对于label标签的占比，求出label标签的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率计算类别出现的概率
        prob = float(labelCounts[key]) / numEntries

        # log base 2
        # 计算香农熵，以2为底求对数
        shannonEnt -= prob * log(prob, 2)
    
    """#------第二种方法----------
    label_count = Counter(data[-1] for data in dataSet)
    probs = [p[1] / len(dataSet) for p in label_count.items()]
    shannonEnt = sum([-p * log(p, 2) for p in probs])
    """
    return shannonEnt

# 将指定特征的特征值等于value的行剩下列作为子数据集
def splitDataSet(dataSet, index, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    # 求第一行有多少列的feature，最后一列是label?
    numFeatures = len(dataSet[0]) - 1
    # 数据集的原始信息熵
    baseEntropy = calcShannonEnt(dataSet)

    # 最优的信息增益和最优的Feature编号
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeatures):
        featlist = [example[i] for example in dataSet]
        # 去除重复的
        uniqueVals = set(featlist)

        # 临时熵
        newEntropy =0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算概率
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 构建树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就是说只有一类，直接返回结果
    # 第一个停止条件：所有的类标签完全相同，则直接返回该标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 如果数据集只有一列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有的特征，仍然不能将数据集划分成仅包含为一类别的分组
    if len(dataSet[0]) == 1:
        pass

    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取label的名称
    bestFeatLabel = labels[bestFeat]

    # 初始化自定义树
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])

    # 取出最优列，然后它的branch做分类
    featValues = [example[bestFeat] for example in dataSet]

    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求剩余的标签的label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree


# 使用决策树进行分类
def classify(inputTree, featLabels, testVec):
    # 给输入节点进行分类
    # inputTree:决策树模型
    # FeatLabels：Feature标签对应的名称
    # testVec:测试输入的数据

    # 获取tree的根节点对应的key值
    firstStr = list(inputTree.keys())[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类




dataset, features = createDataSet()
print(dataset)
print(calcShannonEnt(dataset))

for i in range(2):
    fe = [ex[i] for ex in dataset]
    print("fe: ",fe)
print("xxx")

mytree = createTree(dataset, features)
print("mytree: ", mytree)