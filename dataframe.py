import pandas as pd

# count classification values
dataframe = pd.read_csv('pedrinhas.csv')
last_column = dataframe.iloc[:, -1]
occurrences = last_column.value_counts()
print(occurrences)

list_d = list(dataframe['desceu'])
list_c = list(dataframe['classificacao'])

count = [0, 0]

for des,cls in zip(list_d, list_c):
    pass