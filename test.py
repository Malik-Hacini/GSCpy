import gscpy

data,labels=gscpy.data_files_managing.load_data_n_labels('test_package')
labels_spectral=gscpy.spectral_clustering(data,n_clusters=2,true_labels=labels)
print(labels,labels_spectral)