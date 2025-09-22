import csv

def generate_csv(results_dict, dataset_name="dataset_name", output_file="final-result.csv"):

    header = ["dataset_name", "classifier_name", "metric"] + [f"v{i+1}" for i in range(20)]

    rows = []
    for classifier, metrics in results_dict.items():
        for metric_name, values in metrics.items():
            row = [dataset_name, classifier, metric_name] + values[:20]
            rows.append(row)

    with open(f"16_09_25/csvs/{output_file}", "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows) 

if __name__ == '__main__':
    final_results = {
        'perceptron': {"f1-score": [0.9]*20, "accuracy": [0.8]*20},
        'svm': {"f1-score": [0.9]*20, "accuracy": [0.8]*20},
        'knn': {"f1-score": [0.9]*20, "accuracy": [0.8]*20},
        'tree': {"f1-score": [0.9]*20, "accuracy": [0.8]*20},
        'bayes': {"f1-score": [0.9]*20, "accuracy": [0.8]*20}
    }

    generate_csv(final_results)