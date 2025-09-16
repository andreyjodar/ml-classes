import pandas as pd
import numpy as np
from normalize_optm import remove_columns;
from normalize_optm import normalize_stringcol;
from normalize_optm import normalize_numcol;


# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous. (final_weight)
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

# age, workclass, final_weight, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, class

def data_set(fname):
    result = {}
    result['nome-arquivo'] = fname
    print("Carregando dataset")
    data = pd.read_csv(fname, skipinitialspace=True, skip_blank_lines=True)

    cols = list(data.columns)
    drop_columns = ['capital-gain', 'capital-loss']
    string_columns = ['workclass', 'education', 'occupation', 'marital-status', 'relationship', 'race', 'sex', 'native-country']
    number_columns = ['age', 'final_weight', 'education-num', 'hours-per-week']

    print("Removendo colunas enviesadas")
    remove_columns(data, drop_columns)
    print("Normalizando dados qualitativos")
    normalize_stringcol(data, string_columns)
    print("Normalizando dados quantitativos")
    normalize_numcol(data, number_columns)

    last = cols[-1]
    last_orig = data[last]
    cls_orig, classes, cls_count = np.unique(last_orig, return_inverse=True, return_counts=True)
    data = data.drop(columns=last)

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

FNAME = '04_09_25/adult.csv'

if __name__ == '__main__':
    data = data_set(FNAME)
    
    show_dataset(data)
    show_unbalanced(data)
