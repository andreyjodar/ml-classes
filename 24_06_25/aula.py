import pandas as pd
import numpy as np


def data_set(fname):
    result = {}
    result['nome-arquivo'] = fname
    data = pd.read_csv(fname)
    cols = data.columns
    ultima = cols[-1]

    nome_orig = data[ultima]

    cls_orig, classes, cls_cnt = np.unique(
        nome_orig, return_inverse=True, return_counts=True)

    df = data.drop(columns=ultima)

    result['dados'] = df
    result['classes'] = classes
    result['cls-orig'] = cls_orig
    result['cls-count'] = cls_cnt

    return result

def data_set_v2(fname):
    result = {}
    result['nome-arquivo'] = fname
    data = pd.read_csv(fname, skipinitialspace=True)
    process = ['workclass', 'native-country',
               'race', 'relationship', 'education', 'marital-status', 'occupation', 'sex','class']
    for colname in process:
        transform_values(data, colname)

    return result

def transform_values(data, column):
    values = data[column]
    vlr_orig, value, count = np.unique(values, return_inverse=True, return_counts=True)
    print('valor-original -->', vlr_orig)
    print('valor -->', value)
    print('contagem -->', count)
    
    data.drop(columns=column)
    data[column] = value
    
    return data

def print_data(data):
    print('-'*40)
    ncls = len(data['cls-orig'])
    print(f'1- possui {ncls} classes')
    print('2- numero de itens para cada classe:', data['cls-count'])

    print('-'*40)

def print_imbalance(data):
    soma = np.sum(data['cls-count'])
    result = []
    for vlr in data['cls-count']:
        result.append(soma / vlr)

    max = np.max(result)
    print('desbalanceamento -->', max)

FNAME = '24_06_25/adult.data'

if __name__ == '__main__':
    data = data_set(FNAME)
    print_data(data)
    print_imbalance(data)
    