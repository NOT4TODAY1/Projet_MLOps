from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_models_and_grids():
    models = {
        'Decision Tree': DecisionTreeClassifier(),
    }

    param_grids = {
        'Decision Tree': {'model__max_depth': [3, 5, 7, None]},
    }

    return models, param_grids
