from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_models_and_grids():
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Support Vector Machine': SVC(probability=True),
        'AdaBoost': AdaBoostClassifier(),
    }

    param_grids = {
        'Decision Tree': {'model__max_depth': [3, 5, 7, None]},
        'Random Forest': {'model__n_estimators': [50, 100, 200], 'model__max_depth': [3, 5, 7, None]},
        'K-Nearest Neighbors': {'model__n_neighbors': [3, 5, 7]},
        'Logistic Regression': {'model__C': [0.01, 0.1, 1, 10]},
        'Support Vector Machine': {'model__C': [0.1, 1, 10], 'model__kernel': ['rbf', 'linear']},
        'AdaBoost': {'model__n_estimators': [50, 100]},
    }

    return models, param_grids
