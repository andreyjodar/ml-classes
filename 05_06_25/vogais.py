def contar_vogais(string):
    soma = 0
    vogais = {'a', 'e', 'i', 'o', 'u'}
    for i in range(0, len(string)):
        if(string[i].lower() in vogais):
            soma += 1
    return soma

vogais = contar_vogais('mateus')
print('vogais -->', vogais)