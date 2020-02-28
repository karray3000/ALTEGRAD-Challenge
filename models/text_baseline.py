from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from lightgbm import LGBMClassifier
import pickle

labels = ['business/finance', 'education/research', 'entertainment', 'health/medical',
          'news/press', 'politics/government/law', 'sports', 'tech/science']

class TextBaseline(BaseEstimator):
    def __init__(self):
        self.vec = TfidfVectorizer(decode_error='ignore', strip_accents='unicode', encoding='latin-1', min_df=10, max_df=1000, max_features=1000)
        self.clf = SVC(probability=True, class_weight='balanced', C=1e8)
        # self.clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000, class_weight='balanced')

    def fit(self, X_train, y_train):
        train_data = [u[1] for u in X_train]
        try:
            self.vec = pickles.load(open('models/tfidf_full.pkl', 'rb'))
        except:
            self.vec.fit(train_data)
        X_train_text = self.vec.transform(train_data)
        res = self.clf.fit(X_train_text, y_train)
        self.classes_ = self.clf.classes_
        return res

    def predict_proba(self, X_test):
        test_data = [u[1] for u in X_test]
        X_test_text = self.vec.transform(test_data)
        return self.clf.predict_proba(X_test_text)
