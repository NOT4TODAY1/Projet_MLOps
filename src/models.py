
from sklearn.ensemble import  AdaBoostClassifier


def get_models_and_grids():
    models = {
    
        'AdaBoost': AdaBoostClassifier(),
    }

    param_grids = {

        'AdaBoost': {'model__n_estimators': [50, 100]},
    }

    return models, param_grids
