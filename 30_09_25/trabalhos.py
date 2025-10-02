
import os
import matplotlib.pyplot as plt
import pandas as pd

def _handle_metric(value):
    f1_values = ['f1-score', 'f1_score', 'F1-Score', 'F1', 'f1', 'F1-Measure', 'F1_Score']
    acc_values = ['accuracy', 'Accuracy', 'Acurácia', 'ACC', 'acc', ' acc', 'Acc', 'Acuracia']

    if value in f1_values:
        return 'f1'
    elif value in acc_values:
        return 'acc'
    return value

def _handle_classifier(value):
    perceptron_values = ['perceptron', 'Perceptron', ' perceptron', 'LogisticRegression']
    svm_values = ['svm', 'SVM', 'SVC', ' svm']
    tree_values = ['tree', 'trees', 'DecisionTree', 'Decision Tree', ' trees', 'RandomForest']
    bayes_values = ['bayes', 'NaiveBayes', 'Naive Bayes', 'GaussianNB', ' bayes']
    knn_values = ['knn', 'KNN', 'KNeighbors', ' knn']

    if value in perceptron_values:
        return 'perceptron'
    elif value in svm_values:
        return 'svm'
    elif value in tree_values:
        return 'tree'
    elif value in bayes_values:
        return 'bayes'
    elif value in knn_values:
        return 'knn'
    return value 

def apllay_pattern(df):
    df['metric'] = df['metric'].apply(_handle_metric)
    df['classifier'] = df['classifier'].apply(_handle_classifier)
    return df

def load_csvs(basedir, result_file):
    result = []
    for fname in os.listdir(basedir):
        if not fname.endswith('.csv'):
            continue

        print(f'Lendo arquivo: {fname}...')
        df = pd.read_csv(os.path.join(basedir, fname), skipinitialspace=True)

        df['author'] = fname[:-4]
        df.columns = MYCOLS
        df = apllay_pattern(df)

        value_cols = [c for c in MYCOLS if c.startswith("v")]
        df[value_cols] = df[value_cols].apply(
            lambda x: pd.to_numeric(x.astype(str).str.strip(), errors="coerce")
        )

        result.append(df)

    result = pd.concat(result, axis=0, ignore_index=True)
    output_path = os.path.join(basedir, result_file)
    print(f'Salvando arquivo: {result_file}...')

    result.to_csv(output_path, index=False, float_format="%.10f")

def generate_filtered_boxplot(file, classifier, metric):
    df = pd.read_csv(file)
    df = df[(df['classifier'] == classifier) & (df['metric'] == metric)]

    if df.empty:
        print("Nenhuma ocorrência encontrada")
        return

    value_cols = [col for col in df.columns if col.startswith("v")]
    df_long = df.melt(id_vars=["dataset"], value_vars=value_cols,
                      var_name="fold", value_name="score")

    plt.figure(figsize=(22, 10))
    df_long.boxplot(column="score", by="dataset", grid=False)
    plt.title("Boxplot Classifier (KNN)")
    plt.suptitle("")
    plt.xlabel("Dataset")
    plt.ylabel("F1 Score")
    plt.xticks(fontsize=8, rotation=60, ha="right")
    plt.tight_layout()
    print(f'Salvando imagem: {classifier}-{metric}-boxplot.png"')
    plt.savefig(file.replace('.csv', f'-{classifier}-{metric}-boxplot.png'))

if __name__ == '__main__':
    BASEDIR='C:/Users/Andrey/Documents/ml-classes/30_09_25/trabalhos'
    RESULT='all.csv'
    MYCOLS=['dataset', 'classifier', 'metric'] +[f'v{i}' for i in range (1,21)] + ['author']
    load_csvs(BASEDIR, RESULT)
    generate_filtered_boxplot(os.path.join(BASEDIR, RESULT), 'knn', 'f1')

# ------------------ abaixo, valores de CLASSIFICADOR desregulado
# [
#   'perceptron' 'svm' 'bayes' 'trees' 'knn' 'Perceptron' 'SVM' 
#   'NaiveBayes' 'KNN' 'DecisionTree' 'Naive Bayes' 'Decision Tree' 
#   ' perceptron' ' svm' ' bayes' ' trees' ' knn' 'GaussianNB' 'SVC' 
#   'LogisticRegression' 'RandomForest' 'KNeighbors'
# ]


# ------------------ abaixo, valores de METRICAS desregulado
# [
#   'f1-score' 'accuracy' 'f1' 'F1-Score' 'Acurácia' 'ACC' 'F1' 
#   'f1_score' 'acc' ' f1' ' acc' 'F1-Measure' 'Accuracy' 'Acc' 
#   'F1_Score' 'Acuracia'
# ]
