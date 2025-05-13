import pandas as pd

def count_class(list):
    result = {}
    for item in list:
        if(item not in result):
            result[item] = 0
        result[item] += 1
    print(result)

def count_list(list):
    result = [0, 0, 0]
    for id in list:
        if id not in (1, 2, 3): continue
        result[id-1] += 1
    print(result)
        
count_class([1, 2, 3, 2, '1', 2, 3, 1, 2, 4, 4, 'mar', 'banana', 4, 'mar'])
count_list([1, 2, 3, 2, '1', 2, 3, 1, 2, 4, 4, 'mar', 'banana', 4, 'mar'])
