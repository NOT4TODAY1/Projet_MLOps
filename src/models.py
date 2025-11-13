from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_models_and_grids():
    models = {
        
        
        
        'Logistic Regression': LogisticRegression(max_iter=1000),
        
        
    }

    param_grids = {
        
        
        
        'Logistic Regression': {'model__C': [0.01, 0.1, 1, 10]},
        
        
    }

    return models, param_grids
