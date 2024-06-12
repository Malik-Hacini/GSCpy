from graphs import*
from utils.labels_ordering import*
from utils.error_handling import*
from scipy.linalg import eigh,eig,issymmetric
from sklearn.cluster import KMeans,MiniBatchKMeans
import matplotlib.pyplot as plt
import sys


    
def normalize_vec(vector):
    """Normalizes a vector
    Inputs :
        vector(ndarray): n-d vector.
    
    Returns :
        vec_normalized (ndarray) : n-d vector with norm 1."""
    
    return (1/np.linalg.norm(vector)) * vector

def compute_centers(data,labels):
    clusters=[[datum for j,datum in enumerate(data) if labels[j]==i] for i in range(len(set(labels)))]
    centers=[[sum(i)/len(cluster) for i in zip(*cluster)] for cluster in clusters]
    return centers

def kmeans(data,k,use_minibatch=False):
    '''Performs the k-means clustering algorithm on a dataset of n-dimensional points. 
    Use LLoyd's algorithm to compute the clustering.
    
    Inputs :
        data (ndarray): dataset.
        k (int): Number of clusters (or centroids) to construct.
    
    Returns :
        labels (ndarray) : labels of the points after k-means, ordered in the same way as the dataset.'''
    if use_minibatch:
        est=MiniBatchKMeans(n_clusters=k)
    else:
        est=KMeans(n_clusters=k)
    est.fit(data)

    return est.labels_


def cluster(n_clusters,n_eig,laplacian_matrix,use_minibatch,print_progress=True):
    vals,u_full=eigenvectors(n_eig,*laplacian_matrix)
    
    #In the case of L_sym being used, there is an additional normalization step.
    if laplacian_matrix=='sym':
        u_full=np.apply_along_axis(normalize_vec, axis=0, arr=u_full)
    u=u_full[:,:n_clusters]
    if print_progress:
        print("k-means clustering the spectral embedded data...")
    labels_unord=kmeans(u,n_clusters,use_minibatch)

    return vals,labels_unord


def eigenvectors(i,a,b=None):
    """Computes the first i eigenvals and eigenvecs of a symmetric matrix for the generalized problem
    a*X=lambda*b*X
    Inputs :
        a(ndarray): Needs to be real symmetric.
        i (int): Number of eigenvals and eigenvecs to compute (in ascending order)
        b (ndarray/str) : The second matrix / a string.
    Returns :
        vecs (ndarray) : Matrix with the i eigenvecs as columns."""
    if isinstance(b,str):
        vals,vecs=eig(a)
        vals,vecs=vals.real,vecs.real
        idx = vals.argsort()
        vals = vals[idx]
        vecs = vecs[:,idx]
        vals,vecs=vals[:i],vecs[:,:i]
    else:
        vals,vecs=eigh(a,b,subset_by_index=[0, i-1])
        vals,vecs=vals.real,vecs.real
    return vals,vecs


def ch_index(data,clustering_labels):
    labels_unique=list(set(clustering_labels))
    N=len(data)
    k=len(labels_unique)
    cluster_centers=np.array(compute_centers(data,clustering_labels))
    set_center=np.array([sum(i) for i in zip(*data)])/N
    vols_dist=[len([i for i in clustering_labels if i==cluster])*np.linalg.norm(cluster_centers[j]-set_center) for j,cluster in enumerate(labels_unique)]
    intra_dist=[sum([np.linalg.norm(datum-cluster_centers[j]) for i,datum in enumerate(data) if clustering_labels[i]==j]) for j in labels_unique]
    ch=(N-k)/(k-1)*(sum(vols_dist))/sum(intra_dist)
    
    return ch


def unsupervised_gsc(data,n_eig,graph,laplacian,max_it,n_clusters,use_minibatch):
   ch_list,vals_list,labels_list=[],[],[]

   for j in range(max_it):
       gsc_params=(2**j,1,0.99)
       vals,labels_unord=cluster(n_clusters,n_eig,graph.laplacian(laplacian,gsc_params),use_minibatch,print_progress=False)
       vals_list.append(vals)
       labels_list.append(labels_unord)
       ch_list.append(ch_index(data,labels_unord))
   argmax=np.argmax(ch_list)
   return vals_list[argmax],labels_list[argmax]




def spectral_clustering(data,n_clusters,k_neighbors=None,n_eig=None,laplacian='g_rw',
                        g_method='knn',sym_method=None,sigma=1,gsc_params=None,
                        use_minibatch=True,return_labels=True,return_eigvals=True,return_matrix=False,
                        max_it=5,true_labels=np.array([None]),print_progress=False):
    """Performs spectral clustering on a dataset of n-dimensional points.

    Inputs :
        data (list / ndarray): The dataset, a list or an ndarray of n-dimensional points
        k_neighbors (int): Number of neighbors you want to connect in the case of a k-nn graph.
        n_eig (int): Number of eigenvectors to calculate. Used to compute the number of clusters
        laplacian (string) : The laplacian to use between [un_norm , sym , rw] 
        sym_method (string): The method used to symmetrize the graph matrix in the case of an asymmetric adjacency matrix.
        sigma (float): Standard deviation for the gaussian kernel.
        gsc_params (3-uple): (t,alpha,gamma) for the gsc laplacians.
        use_minibatch (bool) : Choice of the k-means algorithm. True might lead to better performance on large datasets. Default = False.
        eigen_only (bool) : If True, the function will only returns the eigenvalues and eigenvectors and not compute the full clustering. Default = False.
        clusters_fixed (int) : The number of clusters in your data. If unknown, leave by default and the eigengap heuristic will be used. Default = False.
        return_matrix (bool) : True <=> returns the adjacency matrix alongside the clustering results. Use if you want to visualize the graph. Default = False.
        labels_given (ndarray) : The correct labels of your data. If given, used to reorder the labels obtained by clustering. Leave empty if unknown. Default = False 
    
    Returns :
        vals (ndarray): the computed eigenvalues pf the graph laplacian.
        labels (ndarray) : labels of the points after spectral clustering, ordered in the same way as the dataset.
        matrix (ndarray) : the adjacency matrix of the graph
        """
    
    
    #Choosing the optimal  base parameters
    try:
        N=len(data)
        data=np.array(data)
    except:
        print("Error : Your dataset is not in the corect format. Please format it as a list or ndarray of n-dimensional points")
        sys.exit()
    if n_eig==None: n_eig=N
    if k_neighbors==None: k_neighbors=int(np.floor(np.log(N)))
    if sym_method is None and laplacian not in ['g','g_rw']: sym_method='mean'

    #Error handling
    handle_errors(n_clusters,k_neighbors,n_eig,laplacian,
                        g_method,sym_method,sigma,gsc_params,
                        use_minibatch,return_labels,return_eigvals,return_matrix,print_progress,
                        max_it,true_labels)

    if print_progress: print("Building dataset graph...")
    graph=Graph(data,k_neighbors,g_method,sym_method,sigma)
    dir_status=['directed','undirected'][int(issymmetric(graph.m))]
    if print_progress: print(f"Dataset's {dir_status} graph built. ")
    
    if print_progress: print("Performing spectral embedding on the data.")


    if laplacian in ['g','g_rw'] and gsc_params is None:
            vals,labels_unord=unsupervised_gsc(data,n_eig,graph,laplacian,max_it,n_clusters,use_minibatch)
    else:
        vals,labels_unord=cluster(n_clusters,n_eig,graph.laplacian(laplacian,gsc_params),use_minibatch,print_progress=print_progress)
 
    
    if np.all(true_labels)==None:
        #If the labels aren't given, we order labels based on cluster centroids.
        #We do not use the KMeans clusters_centers attribute because we 
        #need to compute them on the original dataset, not in the spectral embedding.
        centers=compute_centers(data,labels_unord)
        labels_ordered = reorder_labels(centers,labels_unord)
    else:
        #If the true labels are given, we order our clustering labels by inference.
        try:
            true_labels=labels_to_ints(true_labels)
        except:
            print("Error : The given labels are not in the correct format. Please format it as a list or ndarray with length=len(data)")

        cluster_labels = infer_cluster_labels(n_clusters, true_labels,labels_unord)
        labels_ordered = infer_data_labels(labels_unord,cluster_labels)
        
    results = []
    if return_labels:
        results.append(labels_ordered)
    if return_eigvals:
        results.append(vals)
    if return_matrix:
        results.append(graph.m)
    
    return tuple(results)
    
    

