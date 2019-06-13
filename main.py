import getdata
import mymodel
import mymodel2

data = getdata.loaddata("mix_concentration5gas_nh3so2no2noco.txt",130,5000)
nh3sec = getdata.loadsec("nh3diffsec.txt",125)
so2sec = getdata.loadsec("so2diffsec.txt",125)
train_datas1,train_nh3so2labels1,train_no2nocolabels1,nh3sec1,so2sec1 = getdata.datasize(data,nh3sec,so2sec,125,5000)
history = mymodel.mymodel(train_datas1,nh3sec1,so2sec1,train_no2nocolabels1,train_nh3so2labels1,5,'model1.h5')
data2 = getdata.loaddata("mix_concentration5gas_jixian.txt",130,20000)
train_datas2,train_nh3so2labels2,train_no2nocolabels2,nh3sec2,so2sec2 = getdata.datasize(data2,nh3sec,so2sec,125,19000)
test_data2,test_labels = getdata.testdatasize(data2,nh3sec,so2sec,125,20000)
history2 = mymodel2.mymodel2(train_datas2,test_data2,nh3sec2,so2sec2,train_nh3so2labels2,train_no2nocolabels2,test_labels,10,'model1.h5','model2.h5')
#mymodel.modelsp(history)







