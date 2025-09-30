import os
import csv
import pandas as pd

def generate_pivot_csv(results_dict, dataset_name="dataset_name", output_file="pivot-result.csv"):

    header = ["dataset_name", "classifier_name", "metric"] + [f"result{i+1}" for i in range(20)]

    rows = []
    for classifier, metrics in results_dict.items():
        for metric_name, values in metrics.items():
            row = [dataset_name, classifier, metric_name] + values[:20]
            rows.append(row)

    os.makedirs("16_09_25/csvs", exist_ok=True)
    with open(f"16_09_25/csvs/{output_file}", "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows) 

def generate_melt_csv(results_dict, dataset_name="dataset_name", output_file="melt-result.csv"):
    header = ["dataset_name", "classifier_name", "metric", "result_name", "result_value"]

    rows = []
    for classifier, metrics in results_dict.items():
        for metric_name, values in metrics.items():
            for i, value in enumerate(values, start=1):
                row = [dataset_name, classifier, metric_name, f"result{i}", value]
                rows.append(row)

    os.makedirs("16_09_25/csvs", exist_ok=True)
    with open(f"16_09_25/csvs/{output_file}", "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)

def generate_melt_from_pivot(pivot_file="pivot-result.csv", melt_file="melt-from-pivot-result.csv"):
    df = pd.read_csv(f"16_09_25/csvs/{pivot_file}")

    df_melt = df.melt(
        id_vars=["dataset_name", "classifier_name", "metric"],
        var_name="result_name",
        value_name="result_value"
    )

    os.makedirs("16_09_25/csvs", exist_ok=True)
    df_melt.to_csv(f"16_09_25/csvs/{melt_file}", index=False)

def generate_pivot_from_melt(melt_file="melt-result.csv", pivot_file="pivot-from-melt-result.csv"):
    df = pd.read_csv(f"16_09_25/csvs/{melt_file}")

    df_pivot = df.pivot_table(
        index=["dataset_name", "classifier_name", "metric"],
        columns="result_name",
        values="result_value",
        aggfunc="mean"
    ).reset_index()

    result_cols = sorted([col for col in df_pivot.columns if col.startswith("result")],
                         key=lambda x: int(x.replace("result", "")))
    df_pivot = df_pivot[["dataset_name", "classifier_name", "metric"] + result_cols]

    os.makedirs("16_09_25/csvs", exist_ok=True)
    df_pivot.to_csv(f"16_09_25/csvs/{pivot_file}", index=False)

if __name__ == '__main__':
    final_results = {
        'perceptron': {"f1-score": [0.9]*20, "accuracy": [0.8]*20},
        'svm': {"f1-score": [0.9]*20, "accuracy": [0.8]*20},
        'knn': {"f1-score": [0.9]*20, "accuracy": [0.8]*20},
        'tree': {"f1-score": [0.9]*20, "accuracy": [0.8]*20},
        'bayes': {"f1-score": [0.9]*20, "accuracy": [0.8]*20}
    }

    generate_pivot_csv(final_results)
    generate_melt_csv(final_results)
    generate_melt_from_pivot(pivot_file="pivot-result.csv", melt_file="melt-from-pivot-result.csv")
    generate_pivot_from_melt(melt_file="melt-result.csv", pivot_file="pivot-from-melt-result.csv")