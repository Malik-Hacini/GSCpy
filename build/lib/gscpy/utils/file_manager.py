import numpy as np
from pathlib import Path

def save_data_and_labels(data,labels,name,path=None):
    """Saves data and labels of a dataset under two files : name_data.txt and name_labels.txt.
      The save files are compatible with numpy.loadtxt

    Inputs :
        data (list or ndarray): The data to save.
        labels (list or ndarray) : Labels of each point
        name (str): The name of your dataset, without file extension. Example: 'myset'.
        path (str): The path to save the dataset. Example : 'Datasets/2D'."""
    data,labels=list(data),list(labels)

    if path==None:
        full_path_data=f'{name}_data.txt'
        full_path_labels=f'{name}_labels.txt'
    else:
        Path(path).mkdir(parents=True, exist_ok=True)
        full_path_data=f'{path}/{name}_data.txt'
        full_path_labels=f'{path}/{name}_labels.txt'

    with open(full_path_data, 'w') as file:
    # Iterate through each sublist
        for datum in data:
            # Join the elements of the sublist with a space and write to the file
            file.write(' '.join(map(str, datum)) + '\n')
    with open(full_path_labels, 'w') as file:
        file.write(' '.join(map(str,labels)))
    print('Data and labels saved successfully.')

def load_data_and_labels(name,path=None):
    """Loads data and labels of a dataset. Files need to be compatible with numpy.loadtxt method. 
    
    
       Example :

            Input files :
                myset_data.txt : "1 2
                                        3 4"
                myset_labels.txt : "1 0"

            Output :
                data = [[1,2],[3,4]]
                labels=[1,0]


    Inputs :
        name (str): The name of your dataset, without file extension. Example: 'myset'.
        path (str): The path where the dataset files are stored Example : 'Datasets/2D'.
    Returns :
        data (ndarray): The loaded data
        labels (list or ndarray) : The loaded labels
    """

    if path==None:
        full_path_data=f'{name}_data.txt'
        full_path_labels=f'{name}_labels.txt'
    else:
        full_path_data=f'{path}/{name}_data.txt'
        full_path_labels=f'{path}/{name}_labels.txt'
        
    return np.loadtxt(full_path_data),np.loadtxt(full_path_labels)