import pandas as pd
import matplotlib.pyplot as plt

def plot_boxplot(fname, metric): 
    dataframe = pd.read_csv(fname)
    metric_dataframe = dataframe[dataframe["metric"] == metric]

    value_columns = [col for col in metric_dataframe.columns if col.startswith("v")]
    classifiers = metric_dataframe["classifier_name"].unique()
    clf_values = [metric_dataframe.loc[metric_dataframe['classifier_name'] == clf, value_columns].values.flatten() for clf in classifiers]

    plt.figure(figsize=(8, 6))
    plt.boxplot(clf_values, labels=classifiers)
    plt.title(f"Boxplot - {metric.capitalize()} per Classifier")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Classifiers")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig(f"16_09_25/imgs/{metric}-boxplot.png")

if __name__ == '__main__':
    plot_boxplot("16_09_25/csvs/final-result.csv", "f1-score")
    plot_boxplot("16_09_25/csvs/final-result.csv", "accuracy")