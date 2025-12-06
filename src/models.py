from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_models_and_grids():
    models = {
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(random_state=40),
        'Decision Tree': DecisionTreeClassifier(random_state=40),
        'Logistic Regression': LogisticRegression(random_state=40, max_iter=1000),
        'Support Vector Machine': SVC(random_state=40, probability=True),
        'AdaBoost': AdaBoostClassifier(random_state=40),
    }

    param_grids = {
        'K-Nearest Neighbors': {'model__n_neighbors': [3, 5, 7, 9]},
        'Random Forest': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 7, None],
            'model__min_samples_split': [2, 5]
        },
        'Decision Tree': {
            'model__max_depth': [3, 5, 7, 10, None],
            'model__min_samples_split': [2, 5, 10]
        },
        'Logistic Regression': {
            'model__C': [0.1, 1, 10, 100],
            'model__penalty': ['l2'],  # l1 requires liblinear, l2 works with both
            'model__solver': ['liblinear', 'lbfgs']
        },
        'Support Vector Machine': {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['linear', 'rbf', 'poly'],
            'model__gamma': ['scale', 'auto']
        },
        'AdaBoost': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.1, 0.5, 1.0]
        },
    }

    return models, param_grids
