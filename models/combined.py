import csv
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator

labels = ['business/finance', 'education/research', 'entertainment', 'health/medical',
          'news/press', 'politics/government/law', 'sports', 'tech/science']

class CombinedBaseline(BaseEstimator):
    def __init__(self, **params):
        self.G = nx.read_weighted_edgelist('./data/edgelist.txt', create_using=nx.DiGraph())
        self.vec = TfidfVectorizer(decode_error='ignore', strip_accents='unicode', encoding='latin-1', min_df=10, max_df=1000, max_features=100)
        self.clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=2000)

    def fit(self, X_train, y_train):
        train_data = [u[1] for u in X_train]
        train_hosts = [u[0] for u in X_train]

        X_train_graph = np.zeros((len(train_hosts), 3))
        avg_neig_deg = nx.average_neighbor_degree(self.G, nodes=train_hosts)
        for i in range(len(train_hosts)):
            X_train_graph[i, 0] = self.G.in_degree(train_hosts[i])
            X_train_graph[i, 1] = self.G.out_degree(train_hosts[i])
            X_train_graph[i, 2] = avg_neig_deg[train_hosts[i]]

        X_train_text = self.vec.fit_transform(train_data).todense()

        X_train_combined = np.concatenate((X_train_graph, X_train_text), axis=1)

        res = self.clf.fit(X_train_combined, y_train)
        self.classes_ = self.clf.classes_
        return res

    def predict_proba(self, X_test):
        test_hosts = [u[0] for u in X_test]
        test_data = [u[1] for u in X_test]

        X_test_graph = np.zeros((len(test_hosts), 3))
        avg_neig_deg = nx.average_neighbor_degree(self.G, nodes=test_hosts)
        for i in range(len(test_hosts)):
            X_test_graph[i, 0] = self.G.in_degree(test_hosts[i])
            X_test_graph[i, 1] = self.G.out_degree(test_hosts[i])
            X_test_graph[i, 2] = avg_neig_deg[test_hosts[i]]

        X_test_text = self.vec.transform(test_data).todense()

        X_test_combined = np.c_[X_test_graph, X_test_text]

        return self.clf.predict_proba(X_test_combined)
