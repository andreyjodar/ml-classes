import numpy as np

from random import shuffle
from dataset_roc import data_set

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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

def simple_classific(xdata, ytarg):
    idx = list(range(len(ytarg)))
    shuffle(idx)

    xdata_shuffle = xdata[idx]
    ytarg_shuffle = ytarg[idx]

    xtrain, xtest, ytrain, ytest = train_test_split(xdata_shuffle, ytarg_shuffle, test_size=0.2)


if __name__ == "__main__":
    classifiers = initalize_classifiers()

    FNAME = "/datasets/glass.csv"
    dataset = data_set(FNAME)
    xdata = dataset['dados']
    ytarg = dataset['classes']


