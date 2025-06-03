import pandas as pd
import numpy as np

def data_set(fname):
    result = {}
    result['fname'] = fname
    data = pd.read_csv(fname)
    cols = data.columns

    last = cols[-1]
    cls_name = data[last]
    cls_orig, cls = np.unique(cls_name, return_inverse=True) # encontra elementos únicos
    print('classes -->', cls)

    df = data.drop(columns=last)
    print(cls_name)
    print(df)
    result['dados'] = df

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
    print('fname -->', data['fname'])