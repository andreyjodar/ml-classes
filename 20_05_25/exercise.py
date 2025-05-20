import numpy as np

letras = 'abcdefghijklmnopqrstuvwxyz@#*+'

lista = list(letras)
print('len(lista):', len(lista))
print(lista)

lista = np.array(lista)

# dado a lista anterior, faça os exercícios:

# 1- capturar os primeiros 10 elementos e imprimir na tela
print(lista[:10])

# 2- capturar os últimos 10 elementos e imprimir na tela
print(lista[-10:])

# 3- capturar os 10 elementos do meio e imprimir na tela
middle = int(len(lista)/2)
print(lista[middle - 5: middle + 5])

# 4- imprimir o 21o elemento apenas
print(lista[20])

# 5- imprimir todos elementos, menos os 5 últimos
print(lista[:-5])

# 6- imprimir todos elementos do início até o meio
print(lista[:middle])

# 7- imprimir todos elementos do meio até o final
print(lista[middle:])

# 7.1 - imprimir todos os elementos do meio até o fim, em ordem reversa
print(lista[:middle:-1])

# 8- imprimir todos elementos a partir do 5 , menos os 5 últimos
print(lista[5:-5])

# 9- imprimir o 12 elemento
print(lista[11])

# 10- fazer um laço que repita 10 vezes, imprimindo cada vez 3 elementos
for idx in range(10):
    ii = idx*3
    print(lista[ii:ii+3])
