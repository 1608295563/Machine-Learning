#/usr/bin/python
#coding=utf8
'''
Created on Jan 30, 2018
@author: zzm
'''
import LoadMnist
import numpy as np
import operator

def classify(inx,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inx,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distance=sqDistances**0.5
    sortDistance=distance.argsort()
    ClassCount={}
    for i in range(k):
        votelabel=labels[sortDistance[i]]
        ClassCount[votelabel]=ClassCount.get(votelabel,0)+1
    sortedClassCount=sorted(ClassCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

trainImages=LoadMnist.loadMNISTImages('../Data/train-images-idx3-ubyte')
trainLabels=LoadMnist.loadMNISTLables('../Data/train-labels-idx1-ubyte')

testImages=LoadMnist.loadMNISTImages('../Data/t10k-images-idx3-ubyte')
testLabels=LoadMnist.loadMNISTLables('../Data/t10k-labels-idx1-ubyte')

K=100
errorCount=0.0
for i in range(len(testLabels)):
    curImage=testImages[i]
    result=classify(curImage,trainImages,trainLabels,K)
    #print result ,testLabels[i]
    if(result!=testLabels[i]): errorCount+=1.0
    
print "\n error count: %d" % errorCount
print "\n error rate: %f" % (errorCount/float(len(testLabels)))
