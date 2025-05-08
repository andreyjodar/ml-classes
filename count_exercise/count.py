import pandas as pd

dataframe = pd.read_json('count_exercise/pessoas.json')

names = dataframe['nomes-pessoas']

dictionary = {}
for nam in names:
    dictionary[nam] = 0


