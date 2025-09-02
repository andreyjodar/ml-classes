
# pip install scikit-learn
# pip install pandas

import numpy as np
from sklearn import metrics

from dataset import data_set

from sklearn.linear_model import Perceptron
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

FNAME = "datasets/student-performance.csv"

def load_dataset():
    return data_set(FNAME)

def cross_validation(data, clfs, k):
    xdata = data["dados"]
    ytarget = data["classes"]

    kf = KFold(n_splits=k, shuffle=True, random_state=None)
    f1_scores = {clf_name: [] for clf_name in clfs.keys()}

    for train_index, test_index in kf.split(xdata):
        x_train, x_test = xdata[train_index], xdata[test_index]
        y_train, y_test = ytarget[train_index], ytarget[test_index]

        for clf_name, clf in clfs.items():
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            f1 = metrics.f1_score(y_test, y_pred, average='macro')
            f1_scores[clf_name].append(f1)

    mean_scores = {clf_name: np.mean(scores) for clf_name, scores in f1_scores.items()}
    return mean_scores

def cross_validation_turns(data, clfs, k, turns=3):
    results = {clf_name: [] for clf_name in clfs.keys()}

    for i in range(turns):
        mean_scores = cross_validation(data, clfs, k)
        for clf_name, score in mean_scores.items():
            results[clf_name].append(score)

    final_means = {clf_name: np.mean(scores) for clf_name, scores in results.items()}
    return final_means

def initialize_classifiers():
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

def print_means(final_means):
    print("================ Final F1 Scores ================")
    for clf_name, final_score in final_means.items():
        print(f"{clf_name} mean: {final_score}")
    print("=================================================")


data = load_dataset()
classifiers = initialize_classifiers()
percent_test = 0.2  
kfolds = int(1 / percent_test)
final_means = cross_validation_turns(data, classifiers, kfolds, 3)
print_means(final_means)
