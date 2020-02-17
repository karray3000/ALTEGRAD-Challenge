import networkx as nx
import codecs
import os
from os import path
import pandas as pd

def get_data_train(data_path='./data/'):
    # with open(path.join(data_path, "train.csv"), 'r') as f:
    #     train_data = f.read().splitlines()
    train_data = pd.read_csv(path.join(data_path, "train.csv"), names=['host', 'label'],  header=0).drop_duplicates(subset='host')

    train_hosts = list()
    y_train = list()
    for row in train_data.iterrows():
        host, label = str(row[1]['host']), row[1]['label']

        # host, label = row.split(",")
        train_hosts.append(host)
        y_train.append(label.lower())

    # Text data
    # Load the textual content of a set of webpages for each host into the dictionary "data".
    # The encoding parameter is required since the majority of our data is french.
    text = dict()
    text_path = path.join(data_path, 'text/')
    filenames = os.listdir(text_path)

    for filename in filenames:
        try:
            with codecs.open(path.join(text_path, filename), encoding='utf-8') as f:
                text[filename] = f.read().replace("\n", "").lower()
        except:
            with codecs.open(path.join(text_path, filename), encoding='latin-1') as f:
                text[filename] = f.read().replace("\n", "").lower()

    X_train = list()
    for host in train_hosts:
        if host in text:
            X_train.append([host, text[host]])
        else:
            X_train.append([host, ''])

    return X_train, y_train


def get_data_full(data_path='./data/'):
    # with open(path.join(data_path, "train.csv"), 'r') as f:
    #     train_data = f.read().splitlines()
    train_data = pd.read_csv(path.join(data_path, "train.csv"), names=['host', 'label'],  header=0).drop_duplicates(subset='host')
    with open(path.join(data_path, "test.csv"), 'r') as f:
        test_hosts = f.read().splitlines()

    train_hosts = list()
    y_train = list()
    for row in train_data.iterrows():
        # host, label = row.split(",")
        host, label = str(row[1]['host']), row[1]['label']
        train_hosts.append(host)
        y_train.append(label.lower())

    # Text data
    # Load the textual content of a set of webpages for each host into the dictionary "data".
    # The encoding parameter is required since the majority of our data is french.
    text = dict()
    text_path = path.join(data_path, 'text/')
    filenames = os.listdir(text_path)

    for filename in filenames:
        try:
            with codecs.open(path.join(text_path, filename), encoding='utf-8') as f:
                text[filename] = f.read().replace("\n", "").lower()
        except:
            with codecs.open(path.join(text_path, filename), encoding='latin-1') as f:
                text[filename] = f.read().replace("\n", "").lower()

    X_train = list()
    for host in train_hosts:
        if host in text:
            X_train.append([host, text[host]])
        else:
            X_train.append([host, ''])
    # Get textual content of web hosts of the test set
    X_test = list()
    for host in test_hosts:
        if host in text:
            X_test.append([host, text[host]])
        else:
            X_test.append([host, ''])
    return X_train, y_train, X_test, test_hosts
