from sklearn.neural_network import MLPClassifier 
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier

def get_model(name):
    if name == 'rf':
        return RandomForestClassifier()
    elif name == 'svm':
        return SVC() 
    elif name == 'mlp':
        return MLPClassifier() 
    elif name == 'xgb':
        return XGBClassifier()
    else:
        raise ValueError(f'modelo {name} no soportado')
