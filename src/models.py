from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_models_and_grids():
    models = {
        'K-Nearest Neighbors': KNeighborsClassifier(),

    }

    param_grids = {

        'K-Nearest Neighbors': {'model__n_neighbors': [3, 5, 7]},

    }

    return models, param_grids
