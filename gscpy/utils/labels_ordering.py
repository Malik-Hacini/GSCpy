import numpy as np


def labels_to_ints(labels):
    labels_unique=list(set(labels))
    for i,label in enumerate(labels):
            labels[i]=labels_unique.index(label)
    return np.array(labels)


def reorder_labels(centers,labels):
    ordered_labels=[]
    centers_norm=np.array([np.linalg.norm(center) for center in centers])
    clusters_order=np.argsort(centers_norm)
    
    for label in labels:
        idx=np.where(clusters_order==label)[0][0]
        ordered_labels.append(idx)
    
    return np.array(ordered_labels)

def infer_cluster_labels(n_clusters, actual_labels,labels_spectral):
    """
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    """

    inferred_labels = {}

    for i in range(n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(labels_spectral == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(np.array(labels[0]).astype(int))
        else:
            counts = np.bincount(np.squeeze(np.array(labels).astype(int)))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

        #print(labels)
        #print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))
        
    return inferred_labels  


def infer_data_labels(labels_spectral, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """
    
    # empty array of len(X)
    predicted_labels = np.zeros(len(labels_spectral)).astype(np.uint8)
    
    for i, cluster in enumerate(labels_spectral):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                
    return predicted_labels
