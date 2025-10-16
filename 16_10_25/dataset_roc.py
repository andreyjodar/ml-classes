import pandas as pd
import numpy as np
from normalize_roc import remove_columns
from normalize_roc import normalize_numcol

def data_set(fname):
    result = {}
    result['nome-arquivo'] = fname
    print("Carregando dataset")
    data = pd.read_csv(fname, skipinitialspace=True, skip_blank_lines=True)

    cols = list(data.columns)
    drop_columns = ['id']
    number_columns = ['ri','sodium','magnesium','aluminum','silicon','potassium','calcium','barium','iron']

    print("Removendo colunas enviesadas")
    remove_columns(data, drop_columns)
    print("Normalizando dados quantitativos")
    normalize_numcol(data, number_columns)

    targ = cols[-1]
    targ_orig = data[targ]
    cls_orig, classes, cls_count = np.unique(targ_orig, return_inverse=True, return_counts=True)
    data = data.drop(columns=targ)

    result['dados'] = np.array(data)
    result['classes'] = classes
    result['cls-orig'] = cls_orig
    result['cls-count'] = cls_count
    print("-" * 52)
    return result

def dataset_info(data):
    ###################
    data.info(verbose=True)
    print(data.describe())
    print('tipos:', data.dtypes)
    print('dimensoes:', data.ndim)
    print('linhas x colunas:', data.shape)
    ###################

def show_dataset(data):
    print('-'*40)

    dataset_info(data['dados'])
    ncls = len( data['cls-orig'] )
    print(f'1- possui {ncls} classes')
    print('2- numero de itens para cada classe:', data['cls-count'])
    print('-'*40)

def show_unbalanced(data):
    soma = np.sum( data['cls-count'] )
    result = []
    for vlr in data['cls-count']:
        result.append( soma / vlr )
    max_val = np.max( result )
    print('desbalanceamento -->', max_val)

FNAME = 'datasets/glass.csv'

if __name__ == '__main__':
    data = data_set(FNAME)
    
    show_dataset(data)
    show_unbalanced(data)