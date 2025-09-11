import numpy as np

def transform_col( data ):
    vlr_orig, values, count = np.unique(data, return_inverse=True, return_counts=True)
    result = {}
    result['vlr-orig'] = list(vlr_orig)
    result['values'] = list(values)
    result['vlr-count'] = list(count)
    return result

def normalize_stringcol(data, columns):
    cols = list(data.columns)
    for colname in columns:
        if colname not in cols: continue
        dados = data[ colname ]
        ret = transform_col( dados )
        data[colname] = ret['values']

def normalize_numcol(data, columns):
    cols = list(data.columns)
    for colname in columns: 
        if colname not in cols: continue
        col_min = data[colname].min()
        col_max = data[colname].max()
        data[colname] = (data[colname] - col_min) / (col_max - col_min)

def remove_columns(data, columns):
    cols = list(data.columns)
    for colname in columns:
        if colname not in cols: continue
        data.drop(columns=colname, inplace=True)