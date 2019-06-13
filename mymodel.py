from keras.models import Model
from keras.layers import Input,Multiply,Subtract,Lambda,Add
from keras.optimizers import Adam
from keras.layers import Dense,Flatten,Convolution1D,MaxPool1D
import matplotlib.pyplot as plt


##网络中间数据切割，二维
def slice(x,w1,w2):
    return x[:,w1:w2]

##网络中间数据切割，三维
def sliceh(x,h1,h2):
    return x[:,h1:h2,:]

##网络构架
def mymodel(data,sec1,sec2,label2,label1,epoch,savefile):
 main_input = Input((125,1), dtype='float32', name='main_input')
 #c = BatchNormalization()(main_input)
 c = Convolution1D(2,2, activation='linear', border_mode='valid', name='conv1_1')(main_input)
 #c = MaxPool1D(pool_size=2, stride=1)(c)
 #c = BatchNormalization()(c)
 c = Convolution1D(5, 3, activation='linear', border_mode='valid', name='conv1_2')(c)
 c = MaxPool1D(pool_size=4, stride=1)(c)
 c = Flatten()(c)
 x = Dense(100,activation='linear',name='dense1')(c)
 #x= Dropout(0.2)(x)
 x = Dense(19,activation='linear',name='Dense2')(x)
 #x = Dropout(0.2)(x)
 #x = Dense(19, activation='linear', name='Dense4')(x)
 aux_output1=Dense(2,activation='linear',name='aux_output1')(x)#nh3,so2
 slice_1=Lambda(slice,arguments={'w1':0,'w2':1})(aux_output1)
 slice_2 = Lambda(slice, arguments={'w1': 1, 'w2': 2})(aux_output1)
 aux_input1 = Input((125,1), name='aux_input1')#nh3section
 aux_input2 = Input((125,1), name='aux_input2')  # so2section
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

 lr = 0.0005
 adam = Adam(lr)
 model = Model(inputs=[main_input,aux_input1,aux_input2], outputs=[main_output, aux_output1])
 model.compile(optimizer=adam,loss='mse',loss_weights=[0.002,8],metrics=['accuracy'])
 print('summary:',end='\n')
 model.summary()
 print('-'*44,'training','-'*44)
 history = model.fit(x={'main_input': data,'aux_input1':sec1,'aux_input2':sec2},
           y={'main_output': label2, 'aux_output1': label1},
           batch_size=32, validation_split=0.33, epochs=epoch, verbose=2)
 print(history.history.keys())
 model.save(savefile)
 print('网络结构保存文件：'+ savefile)
 return history




def modelsp(history):
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









