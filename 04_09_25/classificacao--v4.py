# pip install scikit-learn
# pip install pandas

import numpy as np
from random import shuffle
from sklearn import metrics

from dataset_v2 import data_set_v2

from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

FNAME = "04_09_25/adult.csv"

def load_dataset(fname):
    data = data_set_v2(fname)
    norm_columns = ['age', 'final_weight', 'education-num', 'hours-per-week']
    drop_columns = ['capital-gain', 'capital-loss']

    dataframe = data['dados']
    dataframe.drop( columns=drop_columns, inplace=True)

    for colname in norm_columns:
        col_min = dataframe[colname].min()
        col_max = dataframe[colname].max()

        dataframe[colname] = (dataframe[colname] - col_min) / (col_max - col_min)

    data['dados'] = np.array(dataframe)
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
    parcial_result = {clfs_name: [] for clfs_name in classifiers.keys()}

    for fold in range(num_folds):
        print(f"Fold {fold + 1}")
        start = fold * fold_size
        end = (fold + 1) * fold_size
        
        x_test = xdata[start:end]
        y_test = ytarg[start:end]

        x_train = np.concatenate((xdata[:start], xdata[end:]))
        y_train = np.concatenate((ytarg[:start], ytarg[end:]))

        for clf_name, classific in classifiers.items():
            classific.fit(x_train, y_train)
            ypred = classific.predict(x_test)
            
            f1_score = metrics.f1_score(y_test, ypred, average='macro')
            parcial_result[clf_name].append(f1_score)

    return {clf_name: np.mean(scores) for clf_name, scores in parcial_result.items()}

def print_final_mean(final_result):
    print("=" * 50)
    for clfs_name, mean in final_result.items():
        print(f"Classificador: {clfs_name} | F1-Score (Mean): {mean:.4f}")
    print("=" * 50)
    
if __name__ == '__main__':
    data = load_dataset(FNAME)
    xdata = data['dados']
    ytarg = data['classes']
    classifiers = initalize_classifiers()
    turns_result = {clfs_name: [] for clfs_name in classifiers.keys()}

    for turns in range(3):
        idx = list(range(len(ytarg)))
        shuffle(idx)
        xdata_shuffle = xdata[idx]
        ytarg_shuffle = ytarg[idx]

        parcial_result = calculate_crossvalid(xdata_shuffle, ytarg_shuffle, classifiers)
        for clfs_name, result in parcial_result.items():
            turns_result[clfs_name].append(result)

    final_result = {clfs_name: np.mean(results) for clfs_name, results in turns_result.items()}
    print_final_mean(final_result)