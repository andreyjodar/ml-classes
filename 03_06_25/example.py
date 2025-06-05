import pandas as pd
import numpy as np

def data_set(fname):
    result = {}
    result['fname'] = fname
    data = pd.read_csv(fname)
    cols = data.columns

    last = cols[-1]
    cls_name = data[last]
    cls_orig, cls, cls_count = np.unique(cls_name, return_inverse=True, return_counts = True) 
    # np.unique() --> encontra elementos únicos de um array
    # param return_inverse --> retorna índices das classes
    # param return_counts --> retorna a contagem das classes
    print('classes -->', cls)
    print('classe-original -->', cls_orig)

    df = data.drop(columns=last)
    print(cls_name)
    print(df)
    result['dados'] = df
    result['classes'] = cls
    result['cls-orig'] = cls_orig
    result['cls-count'] = cls_count

    # first = cols[0]
    # second = cols[1]

    # print(data[first])
    # print(data[second])
    # print(data[last])

    return result

FNAME = '03_06_25/iris.csv'
if __name__ == '__main__': ## main (executa apenas se for o arquivo princial)
    data = data_set(FNAME)
    print('----------------------------------------')
    ncls = len(data['cls-orig'])
    print(f'fname --> {data['fname']}')
    print(f'1. nclasses --> {ncls}')
    print(f'2. número de itens de cada classe: {data['cls-count']}')
    print(f'')

    soma = np.sum(data['cls-count'])
    result = []

    for vlr in data['cls-count']:
        result.append(soma/vlr)

    max = np.max(result)
    print(f'valor máximo --> {max}')


# Índice de Desbalanceamento (ID)
#   total_registros / contagem_classe

# Site Treino Dataset
#   https://archive.ics.uci.edu/ml/index.php
