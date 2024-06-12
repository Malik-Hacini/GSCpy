from GSC import*
from utils.file_manager import *
from dataset_builder import*

data,labels=build_dataset(save=True,name='wsh',path='Dossier/Dossier2')
model=GSC(2)
model.fit(data,true_labels=labels)
print(model.cluster_centers, model.eigenvals, model.adj_matrix, model.ch_index)
save_data_and_labels(data,model.labels,'testtt')