# GSCpy

This is a package for performing Spectral Clustering.

 It works a near-fully unsupervised way : the only required information is the number of clusters .

The clustering is done using the Generalized Spectral Clustering (GSC) framework developped by Jonckheere et al. in [arXiv:2203.03221](https://arxiv.org/abs/2203.03221).  It has been shown experimentally that this framework regularly outperforms classical spectral clustering for synthetic and real datasets.

Classical spectral clustering can also be performed by tweaking the parameters, as the clustering algorithm used is fully customizable.

## Usage

Interacting with the package is done trough the GSC class, representing a GSC model. To use :

* Create a GSC object with the parameters of your choice
* Cluster your data using the fit method of the class
* Retrieve the clustering using the labels attribute
* Get more information on the clustering by using the available instance attributes (cluster centers, eigenvalues of the graph laplacian, adjacency matrix, Calinski-Harabasz index)
* Evaluate the performance of the clustering using the nmi method.

To help you manage your datasets, GSCpy includes a file manager allowing to easily load and save datasets with their labels.The package also includes an interactive 2D dataset builder, powered by matplotlib.

## Installation

GSCpy is entirely written in Python and requires the following libraries to run correctly :

* NumPy
* Matplotlib
* SciPy
* Scikit-learn

You can install GSCpy and every required library using pip :

```
pip install GSCpy
```

Pypi repository of the project : [GSCpy · PyPI](https://pypi.org/project/GSCpy/)

### Test

You can test your GSCpy installation using this simple script :

```
import gscpy

data,labels_true=gscpy.build_dataset(save=True,name='test')
k=2 #The number of clusters you built
model=gscpy.GSC(n_clusters=k)
model.fit(data)
print('Clustering NMI score :' ,model.nmi(labels_true))
```

This project was carried out as part of an internship at LAAS-CNRS, Toulouse.
