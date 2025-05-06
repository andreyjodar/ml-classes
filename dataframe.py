import pandas as pd

dataframe = pd.read_csv('pedrinhas.csv')
list_d = list(dataframe['desceu'])
list_dis = list(dataframe['distancia'])
list_c = list(dataframe['classificacao'])

sum_desc = {'joao': 0, 'maria': 0}
sum_dist = {'joao': 0, 'maria': 0}
count = {'joao': 0, 'maria': 0}

for dis,des,cls in zip(list_dis, list_d, list_c):
    if(cls == 1): key = 'joao'
    if(cls == 2): key = 'maria'
    sum_desc[key] += des
    sum_dist[key] += dis
    count[key] += 1
    
media_desc = {'joao': 0, 'maria': 0}
media_dist = {'joao': 0, 'maria': 0}
media_desc['joao'] = sum_desc['joao']/count['joao']
media_desc['maria'] = sum_desc['maria']/count['maria']
media_dist['joao'] = sum_dist['joao']/count['joao']
media_dist['maria'] = sum_dist['maria']/count['maria']

print(f'Ocorrências de João: {count['joao']}')
print(f'Ocorrências de Maria: {count['maria']}')
print(f'Média Desceu João: {media_desc['joao']:.2f}')
print(f'Média Desceu Maria: {media_desc['maria']:.2f}')
print(f'Média Distância João: {media_dist['joao']:.2f}')
print(f'Média Distância Maria: {media_dist['maria']:.2f}')
