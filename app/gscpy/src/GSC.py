from utils.spectral_clustering import*

class GSC:
    """Generalized Spectral Clustering model.

    Attributes :
    labels : The labels assigned to each data point by the clustering algorithm.

    cluster_centers : Coordinates of cluster centers. 

    eigenvals: The eigenvalues of the similarity matrix.

    adj_matrix: The similarity matrix used in spectral clustering.

    ch_index: The Calinski-Harabasz index, a metric for evaluating the quality of the clustering.
    
    Methods : 
        fit : Clusters a given dataset using the model."""
    def __init__(self,n_clusters,k_neighbors=None,n_eig=None,laplacian='g_rw',
                        g_method='knn',sym_method=None,sigma=1,gsc_params=(1,1,0.99),
                        max_it=10,use_minibatch=True) :
        
        """Creates a GSC model with the specified parameters.

    Inputs :
        n_clusters (int): The number of clusters in your data. Default = floor(log(N)) with N=len(data).

        k_neighbors (int, optional): Number of neighbors you want to connect in the case of a k-nn graph.

        n_eig (int, optional): Number of eigenvectors to calculate. Used to compute the number of clusters. 
                               Leaving default is recommended. Default = N
        laplacian (string, optional) : The graph laplacian used, in [un_norm , sym , rw,g,g_rw]. Default = 'g_rw'

        sym_method (string, optional): The method used to symmetrize the graph matrix in the case of an asymmetric adjacency matrix. 
                                       Default='mean' or 'None' if laplacian in ['g','g_rw]

        sigma (float, optional): Standard deviation for the gaussian kernel similarity function used. Default = '1'

        gsc_params (3-uple, optional): (t (int),alpha (float [0,1]),gamma (float [0,1])) for the generalized laplacians.

        max_it (int, optional) : Max number of iterations for the grid-search of the optimal t parameter. Default = 10.

        use_minibatch (bool, optional) : Choice of the k-means algorithm implementation. 
                                         True might lead to better performance on large datasets. Default = True.
        
    """
        
        self.n_clusters=n_clusters
        self.k=k_neighbors
        self.n_eig=n_eig
        self.laplacian=laplacian
        self.g_method=g_method
        self.sym_method=sym_method
        self.sigma=sigma
        self.gsc_params=gsc_params
        self.use_minibatch=use_minibatch
        self.max_it=max_it
    

    def fit(self,data,true_labels=np.array([None]),print_progress=False): 
        """
        Cluster data using generalized spectral clustering.

        This method performs generalized spectral clustering on the provided data and sets the related attributes for the GSC instance.
    
        Inputs :
            
            data (list or ndarray) :The n-dimensional real-valued points representing the data to be clustered.
            
            true_labels (list or ndarray, optional): 
                The true labels of the data, if available. This can be used for evaluation purposes.default=None
        
            print_progress( bool, optional) :
                A boolean flag that determines if the function should print the progress of the clustering in the terminal during the execution.default=False

        Returns:
        None
           
        
        """
        labels,vals,matrix=spectral_clustering(data,n_clusters=self.n_clusters,k_neighbors=self.k,n_eig=self.n_eig,laplacian=self.laplacian,
                        g_method=self.g_method,sym_method=self.sym_method,sigma=self.sigma,gsc_params=self.gsc_params,
                        use_minibatch=self.use_minibatch,print_progress=print_progress,
                        max_it=self.max_it,true_labels=true_labels)
        setattr(self,'labels',labels)
        setattr(self,'eigenvals',vals)
        setattr(self,'adj_matrix',matrix)
        setattr(self,'cluster_centers',np.array(compute_centers(data,labels)))
        setattr(self,'ch_index',ch_index(data,labels))

        return None
    



