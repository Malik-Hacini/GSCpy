import numpy as np

def full_kernel(N,nodes,similarity):
     matrix=np.zeros((N,N))
     for i in range(N):
          for j in range(i+1):
               if j!=i: #No node is connected to itself (no impact on clustering)
                matrix[i,j]=similarity.gaussian(nodes[i],nodes[j])
                matrix[j,i]=matrix[i,j]
    
     return matrix + matrix.T


def knn(k: int, nodes, similarity):
    """Constructs a k-nearest neighbor graph of the given nodes (n-dimensional points) using the similarity function given.
    
        Inputs :
            k (int): Number of neighbors you want to connect.
            nodes (list) : Nodes of the graph as n-dimensional points.
            similarity (Similarity) : Object of the Similarity class (contains multiples similarity functions)
        
        Returns :
            matrix (ndarray) : k-nearest neighbors non-weighted similarity matrix."""
    N = len(nodes)
    matrix = np.zeros((N, N))


    distances=np.zeros((N,N))
    #Calculate distances
    for i in range(N):
        for j in range(i+1, N):
            distances[i, j] = similarity.gaussian(nodes[i], nodes[j])
    distances=distances+distances.T

    
    # Find k-nearest neighbors
    for i in range(N):
        knn_indices = np.argpartition(distances[i], -k)[-k:]
        for j in knn_indices:
            if j != i:
                matrix[i, j] = 1
       

    return matrix

def knn_gaussian(k: int, N, nodes, similarity):
        """Constructs a k-nearest neighbor graph of the given nodes (n-dimensional points) using the gaussian similarity function.
        The graph is weighted : W(i,j)=0 if j isn't in the k-nearest neighbors of i and W(i,j)=gaussian(i,j) else.
    Inputs :
        k (int): Number of neighbors you want to connect.
        N (int): Number of nodes
        nodes (list) : Nodes of the graph as n-dimensional points.
        similarity (Similarity) : Object of the Similarity class (contains multiples similarity functions)
    
    Returns :
        matrix (ndarray) : k-nearest neighbors weighted similarity matrix."""
    
        N = len(nodes)
        matrix = np.zeros((N, N))

    
        distances=np.zeros((N,N))
        # Calculate distances
        for i in range(N):
            for j in range(i+1, N):
                distances[i, j] = similarity.gaussian(nodes[i], nodes[j])
        
        distances=distances+distances.T


        # Find k-nearest neighbors
        for i in range(N):
            knn_indices = np.argpartition(distances[i], -k)[-k:]
            for j in knn_indices:
                if j != i:
                    matrix[i, j] = distances[i,j]

        return matrix


def symmetrize(matrix,method):
        """Symmetrizes k-nn matrix using different methods.
        Inputs :
            matrix (ndarray): the matrix to symmetrize (must be real valued)
            method (string): the method used, must be in ['mean','and','or']
        Returns :
            matrix (ndarray) : The symmetrized matrix."""
        if method==None:
            #Do not symmetrize. Used for GSC.
            return matrix
        if method=='mean' or method=='or':
            return (1/2)*(matrix+matrix.T)

        return matrix





class Similarity:
    """Used to store different similarity functions between n-dimensional points."""
    def __init__(self,sigma=None):
        """Sigma is a standard deviation parameter used for certain fucntions."""
        self.sigma=sigma

    def euclidean(self,x1 : list,x2: list,*args)->float:
        '''Calculates the euclidean distance between two n-dimensional points x1 and x2'''
        return np.linalg.norm(np.subtract(x1,x2))

    def gaussian(self,x1: list,x2: list)->float:
        '''Returns the gaussian kernel of standard deviation sigma between two n-dimensional points x1 and x2.'''
        return np.exp(-(self.euclidean(x1,x2)**2)/(2*(self.sigma**2)))
    


class Graph:
    """Similarity graphs based on a dataset of n-dimensional vectors"""
    def __init__(self,data,k,g_method,sym_method=None,sigma=None) -> None:
        """
    Inputs :
        data (ndarray): the dataset as an array of n-dimensional points.
        k (int): Number of neighbors you want to connect.
        sigma (float): standard deviation (only if you use a similarity function with this parameter)
     """
    
        self.dim=len(data[0])
        self.k=k
        self.N=len(data)
        self.nodes=data
        self.similarity=Similarity(sigma)
        if g_method=='knn':
             self.m=symmetrize(knn(self.k,self.nodes,self.similarity),sym_method)
        if g_method=='g_knn':
            self.m=symmetrize(knn_gaussian(k,self.N,self.nodes,self.similarity),sym_method)
        if g_method=='f_kernel':
            #Lower sigma needed for good performance.
            self.m=full_kernel(self.N,self.nodes,self.similarity)
        self.degree_m=np.diag([sum(self.m[i]) for i in range(self.N)])

    
    def laplacian(self,choice='un_norm',gsc_params=None):
        """Constructs the chosen graph laplacian based on the graph matrix W."""
        L=np.subtract(self.degree_m, self.m)

        if choice=='un_norm':
            '''Important : We return a tuple of matrices (A,B), we are then going to solve the generalized eigenproblem 
            AX=lambdaBX. If B=None we consider B=Identity.
            '''
            return (L,None) 
        if choice=='sym':
            inv_sqrt_d=np.diag([1/np.sqrt(sum(self.m[i])) for i in range(self.N)])
            return (np.matmul(inv_sqrt_d,np.matmul(L,inv_sqrt_d)),None)
        if choice=='rw':
            return (L,self.degree_m)
        
        if choice in ['g','g_rw']:
                
                t,alpha,gamma=gsc_params
                #Basic matrices of GSC
                P=self.m/self.k #Transition matrix of the natural random walk on a knn graph.
                P_gamma=gamma*P + ((1-gamma)*1/self.N)*np.ones((self.N,self.N))
                v=((1/self.N)*np.matmul(np.ones((1,self.N)),np.linalg.matrix_power(P_gamma,t)))**alpha  
                xi=np.array([sum(v[0,i]*P[i,k] for i in range(self.N)) for k in range(self.N)])
                #Other matrices used for the generalized laplacians
                
                N=np.diag(v[0])
                O=np.diag(xi)
                I=np.identity(self.N)
                N_inv=np.linalg.inv(N)
               
                #Computation
                if choice=='g':
                     return (N + O - np.matmul(N,P) - np.matmul(P.T,N),None)
                if choice=='g_rw':
                     """Return not_sym as the generalized L_rw isn't symmetric and isn't related to a generalized eigenproblem
                     of a symmetric matrix, thus the computation of the eigenvals and vecs is done differently"""
                     return (I - np.matmul(np.linalg.inv((I+np.matmul(N_inv,O))),P+np.matmul(N_inv,np.matmul(P.T,N))),'not_sym')