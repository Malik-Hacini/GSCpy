import gscpy

data,labels_true=gscpy.build_dataset(save=True,name='test')
k=2 #The number of clusters you built
model=gscpy.GSC(k)
model.fit(data)
print('Clustering  NMI score' ,model.nmi(labels_true))