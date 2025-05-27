import numpy as np

lista = [1, 2, 3, 4, 5, 6, 7, 8, 9]
lista = np.array(lista)

print(lista)
lista = lista.reshape(-1,2)
print(lista)
print(lista.T)
