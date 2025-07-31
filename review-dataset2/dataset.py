import pandas as pd
import numpy as np

def _transform_column( data ):
    result = {}
    orig, values, count = np.unique(data, return_inverse=True, return_counts=True)
    result['orig'] = list(orig)
    result['values'] = list(values)
    result['count'] = list(count)
    return result

def _transform_data( data, column_list ):
    for colname in list(data.columns):
        if colname not in column_list: continue
        column = data[ colname ]
        convert = _transform_column( column )
        data[ colname ] = convert['values']
    return data

def dataset_info( data ):
    data.info(verbose=True)
    print(data.describe())
    print('types:', data.dtypes)
    print('dimensions:', data.ndim)
    print('linha x column:', data.shape)

def data_set ( fname ):
    result = {}
    result['fname'] = fname
    data = pd.read_csv(fname, sep=';', skipinitialspace=True, skip_blank_lines=True)
    dataset_info(data)

    mystr = 'school; sex; address; famsize; Pstatus; Mjob; Fjob; reason; guardian; schoolsup; famsup; paid; activities; nursery; higher; internet; romantic; G1; G2'
    process = [x.strip() for x in mystr.split(';')]
    data = _transform_data(data, process)

    last = data.columns[-1]
    classes = list(data[last])
    dataframe = data.drop(columns=last)
    result['dataframe'] = dataframe
    result['classes'] = classes

    return result
