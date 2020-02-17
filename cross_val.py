import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# from models.graph_baseline import GraphBaseline as Model
# from models.text_baseline import TextBaseline as Model
from models.doc_embedding.model import DocEmbeddingModel as Model
# from models.combined import CombinedBaseline as Model

from utils.read_data import get_data_train

X_all, y_all = get_data_train()
print('Data loaded...')
model = Model()

def custom_cross_val(model, X_all, y_all):
    train_score, test_score = list(), list()
    for fold in range(5):
        print(f'Fold {fold + 1}...')
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all)
        model.fit(X_train, y_train)
        train_score.append(log_loss(y_train, model.predict_proba(X_train)))
        test_score.append(log_loss(y_test, model.predict_proba(X_test)))
    return train_score, test_score

train_score, test_score = custom_cross_val(model, X_all, y_all)
print(f'Cross val scores for train: mean {np.mean(train_score):06} - std {np.std(train_score):06}')
print(f'Cross val scores for test : mean {np.mean(test_score):06} - std {np.std(test_score):06}')

