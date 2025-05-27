import numpy as np

letras = 'abcdefghijklmnopqrstuvwxyz@#*+'

lista = list(letras)
print('len(lista):', len(lista))
print(lista)

lista = np.array(lista)

# dado a lista anterior, faça os exercícios:

# 1- capturar os primeiros 10 elementos e imprimir na tela
print('10 Primeiros: \n', lista[:10])

# 2- capturar os últimos 10 elementos e imprimir na tela
print('10 Últimos: \n', lista[-10:])

# 3- capturar os 10 elementos do meio e imprimir na tela
middle = int(len(lista)/2)
print('10 do Meio: \n', lista[middle - 5: middle + 5])

# 4- imprimir o 21o elemento apenas
print('21° Elemento: ', lista[20])

# 5- imprimir todos elementos, menos os 5 últimos
print('Menos os 5 últimos: \n', lista[:-5])

# 6- imprimir todos elementos do início até o meio
print('Elementos do Início ao Meio: \n', lista[:middle])

# 7- imprimir todos elementos do meio até o final
print('Elementos do Meio ao Fim: \n', lista[middle:])

# 7.1 - imprimir todos os elementos do meio até o fim, em ordem reversa
print('Elementos do Meio ao Fim (Reverso): \n', lista[:middle:-1])

# 8- imprimir todos elementos a partir do 5 , menos os 5 últimos
print('Elementos do 5 menos os 5 últimos: \n', lista[5:-5])

# 9 - imprimir o 12 elemento
print('12° Elemento: ', lista[11])

# 10 - fazer um laço que repita 10 vezes, imprimindo cada vez 3 elementos
for idx in range(10):
    ii = idx*3
    print(lista[ii:ii+3])
    
# 11- verificar o que significa 'TRANSPOSE' na internet
# transposição é uma maneira de girar uma matriz em 90 graus, trocando as linhas pelas colunas e vice-versa.
    
# 12- fazer o transpose da tabela e armazenar em outra variável: tabela_t
#       imprimir a tabela normal e sua transposta
tabela = np.matrix([['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                    ['k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'],
                    ['u', 'v', 'w', 'x', 'y', 'z', '@', '#', '*', '+']])
tabela_t = np.transpose(tabela)

print('Tabela\n', tabela)
print('Tabela Transposta\n', tabela_t)
    
# 13 - capturar da tabela o elemento linha=2 e coluna=3, e imprimir na tela
print((tabela.item(1, 2)))

# 14- transformar a tabela em um shape (10, 3), armazenar em tabela2.
#       Imprimir cada linha da tabela2
#       Comparar o resultado com a pergunta:
tabela2 = tabela.reshape((10, 3))
for linha in tabela2:
    print(linha)
print('Quantidade de linhas da tabela 2:', tabela2.shape[0])

# 15- imprimir as colunas da tabela2
print('Imprimindo colunas', np.transpose(tabela2))

# 16- capturar da tabela, os elementos do meio, e colocar na variável: tabela3
#       Imprimir a tabela3. Abaixo o que deve aparecer:
#       ['h' 'i' 'j' 'k']
#       ['n' 'o' 'p' 'q']
#       ['t' 'u' 'v' 'w']
tabela3 = tabela[1:-1, 1:-1]
print('Tabela 3', tabela3)

# Extra - pegar a tabela, deixar em 5 colunas e deixar só os valores do meio
teste = tabela.reshape(-1, 5)
meio = teste[1:-1, 1:-1]
print(teste)
print('Valores do Meio:\n', meio)

# 17- imprimir o shape da tabela3
print('Shape', tabela.shape)

# 18- imprimir todas colunas da tabela3
tabela3_t = np.transpose(tabela3)
print('Colunas tabela 3', tabela3_t)

# 19- transformar a tabela 3 em uma lista, e colocar dentro da variável: lista3
#       imprimir a lista3
lista3 = tabela3.tolist()
# para transformar a lista em uma lista simples
lista3 = [item for sublist in lista3 for item in sublist]
print('Transformando tabela 3 em lista 3:', lista3)

# 20- imprimir na tela, da lista3, os elementos de índice: 1, 4, 7 e 8
#       OBS: todos estes itens devem ser impressos todos em uma única linha
# print('Imprimindo índices:',lista3[1],lista3[4],lista3[7],lista3[8])
