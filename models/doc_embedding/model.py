from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
import pickle
import networkx as nx
import numpy as np
from .utils import get_shortest_paths


labels = ['business/finance', 'education/research', 'entertainment', 'health/medical',
          'news/press', 'politics/government/law', 'sports', 'tech/science']

class DocEmbeddingModel(BaseEstimator):
    def __init__(self, **params):
        self.G = nx.read_weighted_edgelist('./data/edgelist.txt', create_using=nx.DiGraph())
        self.shortest_paths = get_shortest_paths()
        self.embed = pickle.load(open('models/doc_embedding/doc_embeds.pkl', 'rb'))
        self.train_hosts_with_labels = dict()
        # self.clf = StackingClassifier([
        #     ('rf', RandomForestClassifier(n_estimators=20, max_depth=5)),
        #     ('lgbm', LGBMClassifier(max_depth=4, num_leaves=10,
        #                             learning_rate=0.1, reg_lambda=10)),
        #     ('mlp', MLPClassifier(hidden_layer_sizes=(50), max_iter=2000))
        # ])
        # self.clf = LGBMClassifier(max_depth=7, num_leaves=32,
        #                           learning_rate=0.1, reg_lambda=10, reg_alpha=10)
        self.clf = MLPClassifier(hidden_layer_sizes=(20,), max_iter=500)
        self.full = Pipeline(steps=[
            ('ss', StandardScaler()),
            # ('pca', PCA(50)),
            ('clf', self.clf),
        ])

    def fit(self, X_train, y_train):
        train_hosts = [u[0] for u in X_train]
        self.train_hosts_with_labels = {X_train[i][0]: y_train[i] for i in range(len(y_train))}
        X_train_graph = np.zeros((len(train_hosts), 16))
        for i in range(len(train_hosts)):
            for j in range(8):
                curr_host = train_hosts[i]
                shortest_paths = [1 / (1 + self.shortest_paths[curr_host][target_host])
                                  for target_host, target_label in self.train_hosts_with_labels.items()
                                  if target_host in self.shortest_paths[curr_host] and target_label == labels[j] and target_host != curr_host]
                if len(shortest_paths) == 0: shortest_paths = [0]
                X_train_graph[i, j] = np.max(shortest_paths)
                X_train_graph[i, 8 + j] = np.mean(shortest_paths)

        X_train_text = [self.embed[host] for host in train_hosts]
        X_train_combined = np.concatenate((X_train_graph, X_train_text), axis=1)

        res = self.full.fit(X_train_combined, y_train)

        self.classes_ = self.clf.classes_
        return res

    def predict_proba(self, X_test):
        test_hosts = [u[0] for u in X_test]
        X_test_graph = np.zeros((len(test_hosts), 16))

        for i in range(len(test_hosts)):
            for j in range(8):
                curr_host = test_hosts[i]
                shortest_paths = [1 / (1 + self.shortest_paths[curr_host][target_host])
                                  for target_host, target_label in self.train_hosts_with_labels.items()
                                  if target_host in self.shortest_paths[curr_host] and target_label == labels[j] and target_host != curr_host]
                if len(shortest_paths) == 0: shortest_paths = [0]
                X_test_graph[i, j] = np.max(shortest_paths)
                X_test_graph[i, 8 + j] = np.mean(shortest_paths)

        X_test_text = [self.embed[host] for host in test_hosts]
        X_test_combined = np.c_[X_test_graph, X_test_text]
        return self.full.predict_proba(X_test_combined)
