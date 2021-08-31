import numpy as np
import pandas as pd
import math

# 首先计算熵H（D）
def Ent(D):
    num=len(D)
    labelCount={}
    for instance in D:
        currentLabel=instance[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel]=0
        labelCount[currentLabel]+=1
    Ent=0.0
    for key in labelCount:
        prob=float(labelCount[key])/num
        # 数据集的信息熵，也就是每一类在数据集中出现的概率
        Ent-=prob*math.log(prob,2)
    return Ent

# 使用分裂点，将数据集划分为DY,DN
def splitpoint(D,attribute,value):
    DY=[]
    DN=[]
    for instance in D:
        if instance[attribute]<=value:
            DY.append(instance)
        if instance[attribute]>value:
            DN.append(instance)
    return DY,DN

# 给每一个属性的每一个分裂点打分，分数最高的为分裂点，splitpoint（）
def evaluate_numeric_attribute(D,attribute,sigma):
    featureList = [instance[attribute] for instance in D]
    uniqueVals = set(featureList)
    uniqueVals=sorted(list(uniqueVals))
    split_point=0.0
    BestGain=0.0
    for i in range(len(uniqueVals)-1):
        v=(uniqueVals[i]+uniqueVals[i+1])/2.0
        # 使用分裂点，将数据集划分为DY,DN，并递归调用
        DY,DN=splitpoint(D,attribute,v)
        if len(DY)<=sigma or len(DN)<=sigma:
            continue
        baseEnt = Ent(D)
        # 计算信息增益gain，也就是使用分列前的熵减去分裂后的熵
        Gain=baseEnt-float(len(DY))/len(D)*Ent((DY))-float(len(DN))/len(D)*Ent(DN)
        # 更新gain，找到最大值bestgain
        if Gain>BestGain:
            BestGain=Gain
            split_point=v
    return split_point,BestGain


#所占比例最大的数据纯度
def Best_purity(classList):
    if len(classList)==0:
        return None,0.0,0
    classCount={}
    for n in classList:
        if n not in classCount.keys():
            classCount[n]=0
        classCount[n]+=1
    #先排序，再取最大
    sortedCC=sorted(classCount.items(),key=lambda x:x[1],reverse=True)
    majorityLabel=sortedCC[0][0]
    # 计算纯度
    purity=1.0*sortedCC[0][1]/len(classList)
    # 计算size
    num=len(classList)
    return [majorityLabel,purity,num]

#选择最优的属性,evaluate_numeric_attribute(),Best_purity(),splitpoint()
def Best(dataSet,sigma,pai):
    classList = [instance[-1] for instance in dataSet]
    majorityLabel,purity,num=Best_purity(classList)
    # 递归划分停止判断，集合只剩下一个元素；或者数据集的size小于阈值；或者纯度已经超过纯度阈值
    if len(set(classList))==1:
        return None,majorityLabel,purity,num
    if purity>=pai:
        return None,majorityLabel,purity,num
    featnum=len(dataSet[0])-1
    bestFeature=-1
    bestGain=0.0
    bestSplit_point=0.0
    # 表现最好
    for i in range(featnum):
        split_point,Gain = evaluate_numeric_attribute(dataSet,i,sigma)
        if Gain>bestGain:
            bestGain=Gain
            bestFeature=i
            bestSplit_point=split_point
    #划分DY,DN
    dY, dN = splitpoint(dataSet,bestFeature,bestSplit_point)
    if len(dY) < sigma or len(dN) < sigma:
        return None,majorityLabel,purity,num
    return bestFeature,bestSplit_point,bestGain,num

#决策树,输出决策树
def createTree(D,sigma,pai):
    bestFeature, split_point,purity,num= Best(D,sigma,pai)
    if bestFeature == None: return {'类别c1(标记为1) or 类别c2(标记为2)':split_point,'purity':purity,'number':num}
    DY, DN = splitpoint(D, bestFeature, split_point)
    Decision_tree = {}
    Decision_tree['属性X1(标记为0) or 属性X2(标记为1)'] = bestFeature
    Decision_tree['split_point'] = split_point
    Decision_tree['Gain']=purity
    Decision_tree['left_tree'] = createTree(DY, sigma, pai)
    Decision_tree['right_tree'] = createTree(DN, sigma, pai)
    return Decision_tree

data=pd.read_csv('iris.csv')
data=np.array(data)
data=data[:,[0,1,4]]
# 根据要求Iris-setosa为一个类，其他的为另一个类
data[data[:,2]=='Iris-setosa',2]=1
data[data[:,2]=='Iris-versicolor',2]=2
data[data[:,2]=='Iris-virginica',2]=2
data=data.tolist()
myTree = createTree(data,2,0.95)
print(myTree)