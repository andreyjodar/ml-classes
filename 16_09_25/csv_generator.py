import csv

def generate_csv(dict):
    columns = dict.keys()
    lines = dict.zip(*dict.values())

    with open("final-result.csv", "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(lines) 

if __name__ == '__main__':
    dict = {
        'perceptron': [0.1, 0.3, 0.4],
        'svm': [0.1, 0.3, 0.4],
        'knn': [0.1, 0.3, 0.4],
        'tree': [0.1, 0.3, 0.4],
        'bayes': [0.1, 0.3, 0.4]
    }

    generate_csv(dict)