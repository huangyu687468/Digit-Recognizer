from numpy import *
import operator

def classify0(inx,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    #inx初始是1行的数组，通过tile函数进行数组广播，把数组变成行数和dataSet行数相等
    #对应位置相减之后通过平方测量距离
    diffMat = tile(inx,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat ** 2 #平方距离
    sqDistance = sqDiffMat.sum(axis=1)  #axis=0列　axis=1横
    distances = sqDistance ** 0.5  #开根号
    sortedDistances = distances.argsort()  #numpy.argsort() 函数返回的是数组值从小到大的索引值。

    classCount = {}
    #依据排序顺序从近到远依次查看ｋ个样例，记录类别和数量
    for i in range (k):
        voteLebel = labels[sortedDistances[i]]
        classCount[voteLebel]=classCount.get(voteLebel,0)+1  #(key,dafault) 如果key不存在，返回默认值

    #对记录类别数量按数量从大到小排序,并且将字典分解为元组列表
    #dict.items()返回可遍历的(键, 值) 元组数组。operator.itemgetter(1)代表按照键值对中的值进行排序．reverse=True从大到小排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
  
    return sortedClassCount[0][0]
def file2matrix(filename):
    fr = open(filename)#打开文件，返回对象
    next(fr)
    arrayOLines = fr.readlines()
    numberOfLines = 40000#得到文件行数
    returnMat = zeros((numberOfLines,784))#全0的矩阵，行是文件行数，列是3
    classLabelVector = [] #标签存在元组中
    index = 0
    for line in arrayOLines:
        line = line.strip()#Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）或字符序列。不能删除中间部分的字符。
        listFromLine = line.split(',') #split() 通过指定分隔符对字符串进行切片,此处是回车
        returnMat[index,:] = listFromLine[1:785] #将从文件中读取到的数据存放在矩阵中
        classLabelVector.append(int(listFromLine[0]))
        index+=1
        if index==40000:
            break
    fr.close()
    return returnMat,classLabelVector
def file2matrix2(filename):
    fr = open(filename)#打开文件，返回对象
    next(fr)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)#得到文件行数
    returnMat = zeros((numberOfLines,784))#全0的矩阵，行是文件行数，列是3
    index = 0
    for line in arrayOLines:
        line = line.strip()#Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）或字符序列。不能删除中间部分的字符。
        listFromLine = line.split(',') #split() 通过指定分隔符对字符串进行切片,此处是回车
        returnMat[index,:] = listFromLine[0:784] #将从文件中读取到的数据存放在矩阵中
        index+=1
    fr.close()
    return returnMat,index
def handwritingClassTest():
    fr = open('sub.csv','a+')
    dataMat,dataLabels = file2matrix('train.csv')
    normDataSet = dataMat/255
    testMat,n = file2matrix2('test.csv')
    print(n)
    normTestSet = testMat/255
    for j in range(n):
        result = int(classify0(normTestSet[j],normDataSet,dataLabels,3))
        fr.write('%d,%d\n' %(j+1,result))
        if(j%50==0):
            print("have process:%d" %(j))
    fr.close()
