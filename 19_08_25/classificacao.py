# pip install scikit-learn
# pip install pandas

from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.svm import SVC
from random import shuffle
import numpy as np
from dataset import data_set 

FNAME = "datasets/student-performance.csv"

data = data_set(FNAME)
xdata = data['dados']
ytarg = data['classes']

xdata = np.array( xdata ) # np.array -> para embaralhar dados
ytarg = np.array( ytarg )

# embaralhar os dados
nums = list(range(len(ytarg)))
print(nums)
shuffle(nums)
print(nums)

xdata = xdata[ nums ]
ytarg = ytarg[ nums ]

size = len(ytarg)
particao = int(size*0.6) # treino -> 50%

xtreino = xdata[ : particao ]
ytreino = ytarg[ : particao ]

xteste = xdata[ particao : ]
yteste = ytarg[ particao : ]


print(xtreino)
print(ytreino)

print(xteste)
print(yteste)

scores = []
for _ in range(20):
    perceptron = Perceptron(max_iter=100,random_state=None)
    perceptron.fit(xtreino, ytreino)
    yhat = perceptron.predict( xteste )

    precision = metrics.precision_score(yteste, yhat)
    recall = metrics.recall_score(yteste, yhat)

    f1_score = (2*precision*recall) / (precision + recall)
    scores.append(f1_score)

mean_score = np.mean(scores)

scores_svc = []
for _ in range(20):
    svc = SVC(kernel="linear", random_state=None)
    svc.fit(xtreino, ytreino)
    yhat = svc.predict(xteste)

    precision = metrics.precision_score(yteste, yhat)
    recall = metrics.recall_score(yteste, yhat)

    f1_score = (2*precision*recall) / (precision + recall)
    scores_svc.append(f1_score)
    
mean_svc_score = np.mean(scores_svc)
