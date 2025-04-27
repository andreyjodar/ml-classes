import pandas as pd

# count classification values
dataframe = pd.read_csv('pedrinhas.csv')
last_column = dataframe.iloc[:, -1]
occurrences = last_column.value_counts()
print(occurrences)

# find max distance for joao
dataframe_joao = dataframe[dataframe['classificacao'] == 1]
max_distance = dataframe_joao['distancia'].max()
print(f'Maior Distância para João: {max_distance}')

# find max depth for maria
dataframe_maria = dataframe[dataframe['classificacao'] == 2]
max_depth = dataframe_maria['desceu'].max()
print(f'Maior Profundidade para Maria: {max_depth}')