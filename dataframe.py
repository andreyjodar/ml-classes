import pandas as pd

dataframe = pd.read_csv('pedrinhas.csv')
last_column = dataframe.iloc[:, -1]
occurrences = last_column.value_counts()
print(occurrences)