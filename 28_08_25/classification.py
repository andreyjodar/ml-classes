
# pip install scikit-learn
# pip install pandas


import numpy as np
from random import shuffle
from sklearn import metrics

from sklearn.datasets import fetch_openml
from dataset import data_set

from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

FNAME = "datasets/student-performance.csv"

def load_dataset():
    return data_set(FNAME)


data = load_dataset()
xdata = data["dados"]
ytarg = data["classes"]


# embaralhar os dados
idx = list(range(len(ytarg)))
shuffle(idx)
part = int(len(ytarg)*0.8) # assumindo 80% treino

# xtr --> x_treino  ;  xte --> x_teste
xtr = xdata[ :part ]
ytr = ytarg[ :part ]
xte = xdata[ part: ]
yte = ytarg[ part: ]


rng = np.random.RandomState()

perceptron = Perceptron(max_iter=100,random_state=rng)
model_svc = SVC(probability=True, gamma='auto',random_state=rng)
model_bayes = GaussianNB()
model_tree = DecisionTreeClassifier(random_state=rng, max_depth=10)
model_knn = KNeighborsClassifier(n_neighbors=7)

# colocando todos classificadores criados em um dicionario
clfs = {    'perceptron':   perceptron,
            'svm':          model_svc,
            'bayes':        model_bayes,
            'trees':        model_tree,
            'knn':          model_knn
        }

f1_scores = {
            'perceptron':   [],
            'svm':          [],
            'bayes':        [],
            'trees':        [],
            'knn':          []
}

mean_scores = {
            'perceptron':   0,
            'svm':          0,
            'bayes':        0,
            'trees':        0,
            'knn':          0
}

ytrue = yte
print('Treinando cada classificador e encontrando o score')

for i in range(21):
    for clf_name, classific in clfs.items():
        classific.fit(xtr, ytr)
        ypred = classific.predict(xte)
        matrconf = metrics.confusion_matrix(ytrue, ypred)
        acc = metrics.accuracy_score(ytrue, ypred)
        f1 = metrics.f1_score(ytrue, ypred, average='macro')
        f1_scores[clf_name].append(f1)

for clf_name, vector in f1_scores.items():
    mean_scores[clf_name] = np.mean(f1_scores[clf_name])
    print(clf_name, " mean: ", mean_scores[clf_name])
