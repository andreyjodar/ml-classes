# pip install scikit-learn
# pip install pandas

import numpy as np
from random import shuffle
from sklearn import metrics

from dataset_plot import data_set
from csv_generator import generate_pivot_csv
from boxplot_generator import plot_boxplot

from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

FNAME = "datasets/glass.csv"

def load_dataset(fname):
    data = data_set(fname)
    return data

def initalize_classifiers():
    rng = np.random.RandomState()

    perceptron = Perceptron(max_iter=100,random_state=rng)
    model_svc = SVC(probability=True, gamma='auto',random_state=rng)
    model_bayes = GaussianNB()
    model_tree = DecisionTreeClassifier(random_state=rng, max_depth=10)
    model_knn = KNeighborsClassifier(n_neighbors=7)

    classifiers = {    
        'perceptron':   perceptron,
        'svm':          model_svc,
        'bayes':        model_bayes,
        'trees':        model_tree,
        'knn':          model_knn
    }

    return classifiers

def calculate_crossvalid(xdata, ytarg, classifiers):
    num_folds = 5
    fold_size = len(ytarg) // num_folds
    parcial_result = {clfs_name: {"f1-score": [], "accuracy": []} for clfs_name in classifiers.keys()}

    for fold in range(num_folds):
        print(f"Fold {fold + 1}")

        test_mask = np.full(len(ytarg), False, dtype=bool)
        start = fold * fold_size
        end = (fold + 1) * fold_size
        test_mask[start:end] = True
        
        x_test = xdata[test_mask]
        y_test = ytarg[test_mask]

        x_train = xdata[~test_mask]
        y_train = ytarg[~test_mask]

        for clf_name, classific in classifiers.items():
            classific.fit(x_train, y_train)
            ypred = classific.predict(x_test)
            
            f1_score = metrics.f1_score(y_test, ypred, average='macro')
            accuracy = metrics.accuracy_score(y_test, ypred)

            parcial_result[clf_name]["f1-score"].append(f1_score)
            parcial_result[clf_name]["accuracy"].append(accuracy)

    return parcial_result

def print_final_mean(final_result):
    print("=" * 52)
    for clfs_name, mean in final_result.items():
        print(f"Classificador: {clfs_name} | F1 Score (Mean): {mean:.4f}")
    print("=" * 52)
    
if __name__ == '__main__':
    print(f"Carregando Dataset Adult.csv")
    data = load_dataset(FNAME)
    xdata = data['dados']
    ytarg = data['classes']
    print(f"Inicializando Classificadores")
    classifiers = initalize_classifiers()
    final_results = {clfs_name: {"f1-score": [], "accuracy": []} for clfs_name in classifiers.keys()}

    for turns in range(4):
        print("-" * 52)
        print(f"Início da Execução {turns + 1}")
        print(f"Embaralhando Dataset")
        idx = list(range(len(ytarg)))
        shuffle(idx)
        xdata_shuffle = xdata[idx]
        ytarg_shuffle = ytarg[idx]

        print(f"Calculando Cross Validation (F1 Score)")
        parcial_result = calculate_crossvalid(xdata_shuffle, ytarg_shuffle, classifiers)
        print("-" * 52)

        for clfs_name, results in parcial_result.items():
            final_results[clfs_name]["f1-score"].extend(results["f1-score"])
            final_results[clfs_name]["accuracy"].extend(results["accuracy"])

    generate_pivot_csv(final_results, dataset_name="glass_identification", output_file="final-result.csv")
    plot_boxplot(fname="16_09_25/csvs/final-result.csv", metric="f1-score")
    plot_boxplot(fname="16_09_25/csvs/final-result.csv", metric="accuracy")