import sys


def handle_errors(n_clusters,k_neighbors,n_eig,laplacian,
                        g_method,sym_method,sigma,gsc_params,
                        use_minibatch,return_labels,return_eigvals,return_matrix,print_progress,
                        max_it,true_labels):
    int_options={'number of clusters':n_clusters,'k_neighbors': k_neighbors,'n_eig':n_eig,'sigma':sigma,'max_it':max_it}

    for key,item in int_options.items():
        if  not isinstance(item,int): 
            print(f"Error : The given {key} is not an integer.")
            sys.exit()
    
    list_options={'laplacian': (['un_norm','sym','rw','g','g_rw'],laplacian), 'sym_method':(['mean','or',None],sym_method),'g_method': (['knn','g_knn','full_kernel'],g_method)}

    for key,item in list_options.items():
        if  item[1] not in item[0]:
                print(f"Error : The given {key} is not supported. Please enter a {key} in {item[1]}")
                sys.exit()

    bool_options={'use_minibatch':use_minibatch,'return_labels':return_labels,'return_eigvals':return_eigvals,
                  'return_matrix':return_matrix,'print_progress':print_progress}
    for key,item in bool_options.items():
        if not isinstance(item,bool):
                print(f"Error : The given {key} is not a bool.")
                sys.exit()
    
    if not isinstance(gsc_params,tuple) :
        print("Error : gsc_params is not in the correct format. It should be a tuple of 3 parameters :(t (int),alpha (float [0,1]),gamma (float [0,1]))")
        sys.exit()
    else:
        if len(gsc_params!=3):
            print("Error : gsc_params is not in the correct format. It should be a tuple of 3 parameters :(t (int),alpha (float [0,1]),gamma (float [0,1]))")
            sys.exit()