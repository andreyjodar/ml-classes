import json

def count(_list):
    result = {}
    for person in _list:
        if (person not in result): result[person] = 0
        result[person] += 1
    print(result)


with open("count_people/pessoas.json") as file:
    dictionary = json.load(file)
    
people = dictionary["nomes-pessoas"]

count(people)

