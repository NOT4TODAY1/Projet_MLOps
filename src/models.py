from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_models_and_grids():
    models = {
       
        'Random Forest': RandomForestClassifier(random_state=40),
        
    }

    param_grids = {
        
        'Random Forest': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 7, None],
            'model__min_samples_split': [2, 5]
        },
        
    }

    return models, param_grids
