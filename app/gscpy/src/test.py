from GSC import*
from utils.file_manager import *
from dataset_builder import*
from keras.src.datasets import mnist
from sklearn import metrics

(train_data,train_labels),(test_data,test_labels)=mnist.load_data()
#Reshape the train data (28*28 to 784)
data = test_data.reshape(len(test_data),-1)
#Normalize data : [0;255] -> [0;1]
data=data.astype(float) / 255
#data,labels=build_dataset(save=True,name='wsh',path='Dossier/Dossier2')

model=GSC(10)
model.fit(data)
print(metrics.normalized_mutual_info_score(train_labels,model.labels))
save_data_and_labels(data,model.labels,'mnist')