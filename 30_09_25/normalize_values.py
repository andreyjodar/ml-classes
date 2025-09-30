import pandas as pd

def _handle_matric(value):
    f1_values = ['f1-score', 'f1_score', 'F1-Score', 'F1', 'f1', 'F1-Measure', 'F1_Score']
    acc_values = ['accuracy', 'Acurácia', 'ACC', 'acc', ' acc', 'Acc', 'Acuracia']

    if(value in f1_values):
        return 'f1'
    if(value in acc_values):
        return 'acc'


def _convert_classifier(dataset):
    clumns_name = ['']

def _convert_metric(dataset): 
    dataset['metrica'] = _handle_matric(dataset['metrica'])
    return dataset

def convert_all(dataset):
    dataset = _convert_metric(dataset)