list1 = [1, 2, 3, 3, 2, 1, 2, 3, 1 , 2]
list2 = ["mar", "marca", "banana", "mar", ]

def count(_list):
    dictionary = {}
    for item in _list:
        if(item not in dictionary): dictionary[item] = 0
        dictionary[item] += 1
    print("result -> ", dictionary)
    

    
count(list1)
count(list2)