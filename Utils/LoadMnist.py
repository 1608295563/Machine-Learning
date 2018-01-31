#/usr/bin/python
#coding=utf8
'''
Created on Jan 30, 2018
@author: zzm
'''

import struct
import matplotlib.pyplot as plt
import numpy as np

'''load Mnist Image file，return two-dimensional matrix，
row is the number of pictures，
column is rows*columns of every picture,
entry of the matrix is pixel/255.0'''
def loadMNISTImages(filename):
    fp=open(filename,'rb')
    magic=struct.unpack_from(">I",fp.read(4),0)[0]   
    number=struct.unpack_from(">I",fp.read(4))[0]      
    rows=struct.unpack_from(">I",fp.read(4))[0]      
    columns=struct.unpack_from(">I",fp.read(4))[0]      
    print magic,number,rows,columns
    img=struct.unpack_from('>'+str(rows*columns*number)+'B',fp.read())
    img=np.array(img)/255.0
    img=img.reshape(number,rows*columns)    
    fp.close()
    return img
'''load Mnist Label file，return one-dimensional matrix,
entry of matrix is label of picture'''
def loadMNISTLables(filename):
    fp=open(filename,'rb')
    magic=struct.unpack_from(">I",fp.read(4),0)[0] 
    number=struct.unpack_from(">I",fp.read(4),0)[0] 
    print magic,number
    lables= struct.unpack_from('>'+str(number)+'B',fp.read())
    fp.close()
    return lables


if __name__ == '__main__':
    testimages=loadMNISTImages('../Data/t10k-images-idx3-ubyte')
    testlabels=loadMNISTLables('../Data/t10k-labels-idx1-ubyte')
    #trainImages=loadMNISTImages('../Data/train-images-idx3-ubyte')
    print testlabels[1]
    fig=plt.figure()
    plotwindow=fig.add_subplot(111)
    plt.imshow(testimages[1].reshape(28,28),cmap='gray')
    plt.show()