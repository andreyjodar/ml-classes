import numpy as np

from random import shuffle
from dataset_roc import data_set
from generate_plot import generate_roc_plot

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

def initalize_classifiers():
    rng = np.random.RandomState()

    perceptron = CalibratedClassifierCV(Perceptron(max_iter=100,random_state=rng))
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

def simple_classific(xdata, ytarg, classifiers):
    idx = list(range(len(ytarg)))
    shuffle(idx)

    xdata_shuffle = xdata[idx]
    ytarg_shuffle = ytarg[idx]

    xtrain, xtest, ytrain, ytest = train_test_split(xdata_shuffle, ytarg_shuffle, test_size=0.2)

    classes = np.unique(ytarg)
    ytest_bin = label_binarize(ytest, classes=classes)
    n_classes = ytest_bin.shape[1]

    fprs = {}
    tprs = {}
    roc_aucs = {}

    for name, model in classifiers.items():
        model.fit(xtrain, ytrain)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(xtest)
        else:
            y_score = model.decision_function(xtest)
            if y_score.ndim == 1:
                y_score = y_score.reshape(-1, 1)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(ytest_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(ytest_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        fprs[name] = fpr["macro"]
        tprs[name] = tpr["macro"]
        roc_aucs[name] = roc_auc["macro"]

    return fprs, tprs, roc_aucs

if __name__ == "__main__":
    classifiers = initalize_classifiers()

    FNAME = "datasets/glass.csv"
    dataset = data_set(FNAME)
    xdata = dataset['dados']
    ytarg = dataset['classes']

    fprs, tprs, roc_aucs = simple_classific(xdata, ytarg, classifiers)
    generate_roc_plot(fprs, tprs, roc_aucs)
