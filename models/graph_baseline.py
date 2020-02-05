import csv
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator

labels = ['business/finance', 'education/research', 'entertainment', 'health/medical',
          'news/press', 'politics/government/law', 'sports', 'tech/science']

class GraphBaseline(BaseEstimator):
    def __init__(self, **params):
        self.G = nx.read_weighted_edgelist('./data/edgelist.txt', create_using=nx.DiGraph())
        self.clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=2000)

    def get_params(self, deep=True):
        return self.clf.get_params(deep=deep)

    def set_params(self, **params):
        return self.clf.set_params(**params)

    def fit(self, X_train, y_train):
        train_hosts = [u[0] for u in X_train]
        X_train_graph = np.zeros((len(train_hosts), 3))
        avg_neig_deg = nx.average_neighbor_degree(self.G, nodes=train_hosts)
        for i in range(len(train_hosts)):
            X_train_graph[i, 0] = self.G.in_degree(train_hosts[i])
            X_train_graph[i, 1] = self.G.out_degree(train_hosts[i])
            X_train_graph[i, 2] = avg_neig_deg[train_hosts[i]]
        res = self.clf.fit(X_train_graph, y_train)
        self.classes_ = self.clf.classes_
        return res

    def predict(self, X_test):
        test_hosts = [u[0] for u in X_test]
        X_test_graph = np.zeros((len(test_hosts), 3))
        avg_neig_deg = nx.average_neighbor_degree(self.G, nodes=test_hosts)
        for i in range(len(test_hosts)):
            X_test_graph[i, 0] = self.G.in_degree(test_hosts[i])
            X_test_graph[i, 1] = self.G.out_degree(test_hosts[i])
            X_test_graph[i, 2] = avg_neig_deg[test_hosts[i]]
        return self.clf.predict(X_test_graph)

    def predict_proba(self, X_test):
        test_hosts = [u[0] for u in X_test]
        X_test_graph = np.zeros((len(test_hosts), 3))
        avg_neig_deg = nx.average_neighbor_degree(self.G, nodes=test_hosts)
        for i in range(len(test_hosts)):
            X_test_graph[i, 0] = self.G.in_degree(test_hosts[i])
            X_test_graph[i, 1] = self.G.out_degree(test_hosts[i])
            X_test_graph[i, 2] = avg_neig_deg[test_hosts[i]]
        return self.clf.predict_proba(X_test_graph)
