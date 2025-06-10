import pandas as pd
import numpy as np

def data_set(fname):
    result = {}
    data = pd.read_csv(fname)
    cols = data.columns
    last = cols[-1]
    cls_name = data[last]
    cls_orig, cls_index, cls_count = np.unique(cls_name, return_inverse=True, return_counts=True)
    df = data.drop(columns=last)

    result['dataframe'] = df
    result['fname'] = fname
    result['classes'] = cls_orig
    result['n-classes'] = len(cls_orig)
    result['cls-index'] = cls_index
    result['cls-count'] = cls_count
    return result

FNAME = 'review-dataset/wine.data'
if __name__ == '__main__':
    dataset = data_set(FNAME)
    print(f"fname --> {dataset['fname']}")
    print(f"classes --> {dataset['classes']}")
    print(f"n-classes --> {dataset['n-classes']}")
    print(f"cls-index --> {dataset['cls-index']}")
    print(f"cls-count --> {dataset['cls-count']}")
    print(f"dataframe --> {dataset['dataframe']}")

    total = np.sum(dataset['cls-count'])
    result = []

    for value in dataset['cls-count']:
        result.append(total/value)
    
    maximum = np.max(result)
    print(f'max-value --> {maximum}')