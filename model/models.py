from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn.svm import SVC

def get_models():
    models = dict()

    # Gradient boosting
    models['gradient_boosting'] = {
        'params' : {
            'learning_rate' : [1, 10, 100],
            'max_depth' : [5, 25, None],
            'max_features' : ['sqrt', 'log2', None]
            },
        'classifier' : GradientBoostingClassifier()
        }
        
    # Random forest
    models['random_forest'] = {
        'params' : {
            'max_features' : ['sqrt', 'log2', None],
            'max_depth' : [5, 25, None]
            },
        'classifier' : RandomForestClassifier()
        }

    # SVM
    models['svm'] = {
        'params' : {
            'kernel' : ['linear', 'poly', 'rbf'],
            'degree' : [1, 2, 5, 10]
            },
        'classifier' : SVC()
        }

    # Logistic regression
    models['logistic_regression'] = {
        'params' : {
            'tol' : [1e-1, 1e-2, 1e-4, 1e-6],
            'C' : [1e2, 10, 1e0, 1e-1, 1e-3]
            },
        'classifier' : LogisticRegression()
        }

    return models
