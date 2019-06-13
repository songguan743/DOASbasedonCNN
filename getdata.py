import numpy as np
import random

##读取混合气体光谱及强吸收气体的吸收截面
##参数：filename   混合光谱
##      point       数据数
##      sample      样本数
def loaddata(filename,point,sample):
    f = open(filename,'r')
    line = f.readline()
    data = np.zeros((point, sample))
    i = 0
    while line:
        num = np.array([float(x) for x in line.split(',')])
        data[i, :] = num
        line = f.readline()
        i = i+1
    f.close()
    ##对混合光谱数据做shuffle,打乱各气体样本分布
    a,b=data.shape
    index=[i for i in range(b)]
    random.shuffle(index)
    data=data[:,index]
    return data

##数据size调整
##参数：   data    sec1    sec2    数据
##          sec     波长点数据
##          sample  训练样本数
def datasize(data,sec1,sec2,sec,sample):
    # datas为5000组混合光谱数据，labels为浓度标签
    # 一列为一组数据，5000列

    train_3datas = np.transpose(data[0:sec, 0:sample])
    train_labels = np.transpose(data[sec:, 0:sample])
    train_nh3so2labels = np.transpose(data[sec:sec+2, 0:sample])
    train_no2nocolabels = np.transpose(data[sec+2:, 0:sample])
    nh3sec = np.transpose(sec1)
    nh3sec = np.tile(nh3sec, [sample, 1])
    so2sec=np.transpose(sec2)
    so2sec=np.tile(so2sec,[sample,1])

    ##卷积层要求数据类型为三维
    train_3datas = train_3datas.reshape((train_3datas.shape[0],train_3datas.shape[1],1))#(5000,100,1)
    nh3sec = nh3sec.reshape((nh3sec.shape[0],nh3sec.shape[1],1))#(5000,100,1)
    so2sec = so2sec.reshape((so2sec.shape[0],so2sec.shape[1],1))#(5000,100,1)
    return train_3datas,train_nh3so2labels,train_no2nocolabels,nh3sec,so2sec

##读取强吸收气体的吸收截面
##      filesec       吸收截面文件
def loadsec(filesec,point):
    f = open(filesec, 'r')
    line = f.readline()
    sec = np.zeros((point, 1))
    j = 0
    while line:
        num = np.array([float(x) for x in line.split(',')])
        sec[j, :] = num
        line = f.readline()
        j = j + 1
    f.close()
    return sec

##数据size调整
##参数：   data    sec1    sec2    数据
##          sec     波长点数据
##          sample  总样本数
def testdatasize(data,sec1,sec2,sec,sample):
    # datas为5000组混合光谱数据，labels为浓度标签
    # 一列为一组数据，5000列
    test_sample = sample - 1000
    test_datas = np.transpose(data[0:sec, test_sample:])
    test_labels = np.transpose(data[sec:, test_sample:])##labels[test_sample:,sec:]
    test_nh3so2labels = np.transpose(data[sec:sec+2, test_sample:])
    test_no2nocolabels = np.transpose(data[sec+2:, test_sample:])
    nh3sec = np.transpose(sec1)
    nh3sec = np.tile(nh3sec, [1000, 1])
    so2sec=np.transpose(sec2)
    so2sec=np.tile(so2sec,[1000,1])

    ##卷积层要求数据类型为三维
    test_datas = test_datas.reshape((test_datas.shape[0],test_datas.shape[1],1))#(5000,100,1)
    nh3sec = nh3sec.reshape((nh3sec.shape[0],nh3sec.shape[1],1))#(5000,100,1)
    so2sec = so2sec.reshape((so2sec.shape[0],so2sec.shape[1],1))#(5000,100,1)
    return test_datas,test_labels

