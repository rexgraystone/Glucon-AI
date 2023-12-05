import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv('diabetes_prediction_dataset.csv')

def normalize(df):
    for column in df:
        df = df.replace({f'{column}': {"Female": 1, "Male": 0, "No Info": 0, "never": 1, "former": 2, "current": 3, "not current": 4}})
    df = df[~df.apply(lambda row: any(row == ''), axis=1)]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.astype(float)
    df = df.dropna()
    y = df.pop(df.columns[-1])
    X = df
    return X, y

X, y = normalize(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

CLASSIFIERS = {
    SVC(): 'Support Vector Machine', 
    KNeighborsClassifier(): 'K-Nearest Neighbors', 
    DecisionTreeClassifier(): 'Decision Tree', 
    RandomForestClassifier(): 'Random Forest', 
    MultinomialNB(): 'Naive Bayes'
}

def evaluate_classifier(clf, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test) -> float:
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    prec = precision_score(y_true=y_test, y_pred=y_pred, zero_division=0)
    rec = recall_score(y_true=y_test, y_pred=y_pred, zero_division=0)
    return acc, prec, rec

def train() -> list:
    results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall'])
    for index, clf in enumerate(CLASSIFIERS.keys()):
        acc, prec, rec = evaluate_classifier(clf)
        row = [CLASSIFIERS[clf], acc, prec, rec]
        results.loc[index] = row
    return results

def run():
    results = train()
    results = results.set_index('Model')
    results.to_csv('results.csv', index=True)
    ax = results.plot(kind='bar')
    ax.set_xlabel('Models', ha='center', fontsize=10)
    ax.set_ylabel('Values')
    ax.set_title('Performance Metrics')
    plt.legend(loc='upper right', fontsize='small')
    plt.xticks(rotation=0, fontsize=5)
    plt.savefig('Images/results.png', format='png')