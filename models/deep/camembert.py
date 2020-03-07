import pandas as pd
import re
import numpy as np
from sklearn.base import BaseEstimator
from sklearn import preprocessing




from transformers import CamembertTokenizer
from simpletransformers.classification import ClassificationModel

labels = ['business/finance', 'education/research', 'entertainment', 'health/medical',
          'news/press', 'politics/government/law', 'sports', 'tech/science']
pattern = re.compile('[\W_]+')

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


args = {
    'num_train_epochs': 4,
    'early_stopping_patience': 10,
    'early_stopping_delta': 0,
    'train_batch_size': 32,
    "use_cached_eval_features": True,
    "cache_dir": "cache/",
    "no_cache": False,
    "reprocess_input_data": False,
    "overwrite_output_dir": True,
}

class CamembertModel(BaseEstimator):
    def __init__(self):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(labels)
        self.tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        self.model = ClassificationModel('camembert', 'camembert-base', num_labels=8)

    def fit(self, X_train, y_train):
        df_train = pd.DataFrame({'text': [pattern.sub(' ', u[1]) for u in X_train], 'labels': self.le.transform(y_train)})
        res = self.model.train_model(df_train, tokenizer=self.tokenizer, output_dir="./model/camembert",
                                     show_running_loss=True, args=args)
        return res

    def predict_proba(self, X_test):
        test_texts = [pattern.sub(' ', u[1]) for u in X_test]
        _, raw_outputs = self.model.predict(test_texts)
        return [softmax(u) for u in raw_outputs]
