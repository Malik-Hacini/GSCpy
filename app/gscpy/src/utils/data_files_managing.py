import numpy as np

def save_data_n_labels(data,labels,name,path=''):
    """Name should be 'set' and not 'set.txt'."""
    data,labels=list(data),list(labels)
    with open(f'{path}/{name}_data.txt', 'w') as file:
    # Iterate through each sublist
        for datum in data:
            # Join the elements of the sublist with a space and write to the file
            file.write(' '.join(map(str, datum)) + '\n')
    with open(f'{path}/{name}_labels.txt', 'w') as file:
        file.write(' '.join(map(str,labels)))
    print('Data and labels saved successfully.')

def load_data_n_labels(name,path='src/utils/Datasets'):
    """Name should be 'set' and not 'set.txt'."""
    return np.loadtxt(f'{path}/{name}_data.txt'),np.loadtxt(f'{path}/{name}_labels.txt'),name