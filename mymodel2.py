from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.layers import Input,Multiply,Subtract,Dot,Lambda,Add
from keras.layers import Flatten,Convolution1D,MaxPool1D,ZeroPadding2D,Reshape
from keras.layers.core import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import random
import keras.backend as K
from keras.callbacks import LearningRateScheduler,TensorBoard

'''
#  数据读取  #
# 输入： 10000组数值拟合的混合气体光谱 min_concentration.txt
# 输出： train集train_ABS test集test_ABS
f = open(r"mix_concentration5gas_jixian.txt")
line = f.readline()
data = np.zeros((130, 20000))
i = 0
while line:
    num = np.array([float(x) for x in line.split(',')])
    data[i, :] = num
    line = f.readline()
    i = i+1
f.close()
a,b=data.shape
index=[i for i in range(b)]
random.shuffle(index)
data=data[:,index]

f = open(r"nh3diffsec.txt")
line = f.readline()
nh3sec = np.zeros((125, 1))
j = 0
while line:
    num = np.array([float(x) for x in line.split(',')])
    nh3sec[j, :] = num
    line = f.readline()
    j = j+1
f.close()

f = open(r"so2diffsec.txt")
line = f.readline()
so2sec = np.zeros((125, 1))
j = 0
while line:
    num = np.array([float(x) for x in line.split(',')])
    so2sec[j, :] = num
    line = f.readline()
    j = j+1
f.close()
batchsize=32

f = open(r"no2diffsec.txt")
line = f.readline()
no2sec = np.zeros((125, 1))
j = 0
while line:
    num = np.array([float(x) for x in line.split(',')])
    no2sec[j, :] = num
    line = f.readline()
    j = j+1
f.close()

f = open(r"nodiffsec.txt")
line = f.readline()
nosec = np.zeros((125, 1))
j = 0
while line:
    num = np.array([float(x) for x in line.split(',')])
    nosec[j, :] = num
    line = f.readline()
    j = j+1
f.close()

f = open(r"mix_concentration5gas_jixian_1000.txt")
line = f.readline()
test2_data = np.zeros((130, 1000))
i = 0
while line:
    num = np.array([float(x) for x in line.split(',')])
    test2_data[i, :] = num
    line = f.readline()
    i = i+1
f.close()
a,b=test2_data.shape
index=[i for i in range(b)]
random.shuffle(index)
test2_data=test2_data[:,index]


# datas为10000组混合光谱数据，labels为浓度标签
# 一列为一组数据，10000列
train_3datas = np.transpose(data[0:125, 0:19000])#（9000，100）
train_labels = np.transpose(data[125:, 0:19000])#（9000，5）
train_nh3so2labels = np.transpose(data[125:127, 0:19000])#（9000,2）
train_no2nocolabels = np.transpose(data[127:, 0:19000])#（9000,2）
train_colabels = np.transpose(data[129:130, 0:19000])#（9000,1）
so2sec=np.transpose(so2sec)#（1,100）
so2sec=np.tile(so2sec,[19000,1])#（9000,100）
nh3sec=np.transpose(nh3sec)
nh3sec=np.tile(nh3sec,[19000,1])
no2sec=np.transpose(no2sec)
no2sec=np.tile(no2sec,[19000,1])
nosec=np.transpose(nosec)
nosec=np.tile(nosec,[19000,1])
train_3datas = train_3datas.reshape((train_3datas.shape[0],train_3datas.shape[1],1))#(9000,100,1)
nh3sec = nh3sec.reshape((nh3sec.shape[0],nh3sec.shape[1],1))#(9000,100,1)
so2sec = so2sec.reshape((so2sec.shape[0],so2sec.shape[1],1))#(9000,100,1)
no2sec = no2sec.reshape((no2sec.shape[0],no2sec.shape[1],1))#(9000,100,1)
nosec = nosec.reshape((nosec.shape[0],nosec.shape[1],1))#(9000,100,1)

test_3datas = np.transpose(data[0:125, 19000:])#（1000,100）
test_labels = np.transpose(data[125:, 19000:])#（1000,5）
test_nh3labels = np.transpose(data[125:126, 19000:])#（1000,1）
test_so2labels = np.transpose(data[126:127, 19000:])#（1000,1）
test_no2labels = np.transpose(data[127:128, 19000:])#（1000,1）
test_nolabels = np.transpose(data[128:129, 19000:])#（1000,1）
test_colabels = np.transpose(data[129:130, 19000:])#（1000,1）
test_3datas = test_3datas.reshape((test_3datas.shape[0],test_3datas.shape[1],1))#(1000,100,1)

test2_3datas = np.transpose(test2_data[0:125, :])
test2_labels = np.transpose(test2_data[125:, :])
test2_nh3labels = np.transpose(test2_data[125:126, :])
test2_so2labels = np.transpose(test2_data[126:127, :])
test2_no2labels = np.transpose(test2_data[127:128, :])
test2_nolabels = np.transpose(test2_data[128:129, :])
test2_colabels = np.transpose(test2_data[129:130, :])
test2_3datas = test2_3datas.reshape((test2_3datas.shape[0],test2_3datas.shape[1],1))#(6000,64,1)
'''
def slice(x,w1,w2):
    return x[:,w1:w2]

def sliceh(x,h1,h2):
    return x[:,h1:h2,:]


#aux_output为so2浓度，main_output为nox浓度
def mymodel2(data,test_data,sec1,sec2,label1,label2,test_label,epoch,loadfile,savefile):
     main_input = Input((125,1), dtype='float32', name='main_input')#（None,100,1）
     #c = BatchNormalization()(main_input)
     c = Convolution1D(2,2, activation='linear', border_mode='same', name='conv1_1')(main_input)#(None,100,2)
     #c = MaxPool1D(pool_size=2, stride=1)(c)
     #c = BatchNormalization()(c)
     c = Convolution1D(5, 3, activation='linear', border_mode='same', name='conv1_2')(c)#(None,100,5)
     c = MaxPool1D(pool_size=4, stride=1)(c)#（None,97,5)
     c = Flatten()(c)#(None,485)
     x = Dense(100,activation='linear',name='dense1')(c)#(None,100)
     #x= Dropout(0.2)(x)
     x = Dense(19,activation='linear',name='Dense2')(x)#(None,19)
     #x = Dropout(0.2)(x)
     #x = Dense(19, activation='linear', name='Dense4')(x)
     aux_output1=Dense(2,activation='linear',name='aux_output1')(x)#nh3,so2(None,2)
     slice_1=Lambda(slice,arguments={'w1':0,'w2':1})(aux_output1)#(None,1)
     slice_2 = Lambda(slice, arguments={'w1': 1, 'w2': 2})(aux_output1)#(None,1)
     aux_input1 = Input((125,1), name='aux_input1')#nh3section#(None,100,1)
     aux_input2 = Input((125,1), name='aux_input2')  # so2section#(None,100,1)
     y1=Multiply()([aux_input1,slice_1])
     y2 = Multiply()([aux_input2, slice_2])
     y3=Add()([y1,y2])
     y_so2=Subtract()([main_input,y3])#abs-c*so2sec
     #c = BatchNormalization()(y_so2)
     c = Convolution1D(2,2, activation='linear', border_mode='same', name='conv2_1')(y_so2)
     #c = MaxPool1D(pool_size=2, stride=1)(c)
     #c = BatchNormalization()(c)
     c = Convolution1D(5, 3, activation='linear', border_mode='same', name='conv2_2')(c)
     c = MaxPool1D(pool_size=4, stride=1)(c)
     c = Flatten()(c)
     y=Dense(100,activation='linear',name='Dense5')(c)
     #y = Dropout(0.2)(y)
     y=Dense(10,activation='linear',name='Dense6')(y)
     #y = Dropout(0.2)(y)
     #y = Dense(7, activation='linear', name='Dense8')(y)
     main_output = Dense(3, activation='linear', name='main_output')(y)#no2,no


     def scheduler(epoch):
        # 每隔100个epoch，学习率减小为原来的1/10
        if epoch % 100 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(model.optimizer.lr)

     lr = 0.0005
     adam = Adam(lr)
     model = Model(inputs=[main_input, aux_input1, aux_input2],
                      outputs=[main_output, aux_output1])
     model.compile(optimizer=adam, loss='mse', loss_weights=[0.05,8],metrics=['accuracy'])
     print('summary:', end='\n')
     model.summary()
     model.load_weights(loadfile)

     print('-' * 44, 'training', '-' * 44)
     reduce_lr = LearningRateScheduler(scheduler)
     history = model.fit(x={'main_input': data, 'aux_input1': sec1, 'aux_input2': sec2},
              y={'main_output': label2 , 'aux_output1': label1},
              batch_size=32, validation_split=0.33, epochs=epoch, verbose=2,callbacks=[reduce_lr])
     model.save(savefile)
     print('网络结构保存文件：' + savefile)


     # summarize history for loss
     plt.plot(history.history['main_output_acc'])
     plt.plot(history.history['val_main_output_acc'])
     plt.title('main_output acc')
     plt.ylabel('acc')
     plt.xlabel('epoch')
     plt.legend(['train', 'val'], loc='upper left')
     plt.show()

     plt.plot(history.history['aux_output1_acc'])
     plt.plot(history.history['val_aux_output1_acc'])
     plt.title('aux_output1 acc')
     plt.ylabel('acc')
     plt.xlabel('epoch')
     plt.legend(['train', 'val'], loc='upper left')
     plt.show()

     print('-' * 44, 'testing', '-' * 44)
     # cost = model.evaluate(y_test,A_test,batch_size= 5)
     [main_cost, aux_cost1] = model.predict([test_data, sec1, sec2])
     concentration = np.concatenate((aux_cost1, main_cost), axis=1)
     np.savetxt('Cconcentration.txt', concentration)
     concentration1 = test_label
     np.savetxt('Crealconcentration.txt', concentration1)
     print('pre_A\n', concentration[0:10, :])
     print('test_A\n', concentration1[0:10, :])
     error1 = abs(aux_cost1[:, 0:1] - test_label[:,0:1])
     error2 = abs(aux_cost1[:, 1:2] - test_label[:,1:2])
     error3 = abs(main_cost[:, 0:1] - test_label[:,2:3])
     error4 = abs(main_cost[:, 1:2] - test_label[:,3:4])
     error5 = abs(main_cost[:, 2:3] - test_label[:,4:])
     error = np.concatenate((error1, error2, error3, error4, error5), axis=1)
     np.savetxt('error.txt', error)
     print('反演结果保存：'+'Cconcentration.txt')
     print('真实结果保存：'+'Crealconcentration.txt')
     print('绝对误差保存：'+'error.txt')

##myfunction(train_3datas,so2sec,nh3sec,train_no2nocolabels,train_nh3so2labels)







