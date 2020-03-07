import csv
import numpy as np

from models.doc_embedding.model import DocEmbeddingModel
from models.deep.cnn import CNNModel
from models.deep.camembert import CamembertModel

from utils.read_data import get_data_full

labels = ['business/finance', 'education/research', 'entertainment', 'health/medical',
          'news/press', 'politics/government/law', 'sports', 'tech/science']

X_train, y_train, X_test, test_hosts = get_data_full()

models = [
  DocEmbeddingModel(),
  CNNModel(),
  CamembertModel(),
]
coefs = [0.3, 0.5, 0.2]

for model in models:
    model.fit(X_train, y_train)
print('Models trained')

y_preds = []
for model in models:
    y_preds.append(model.predict_proba(X_test))

with open('./submissions.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    try:
        lst = model.classes_.tolist()
    except:
        lst = labels
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i,test_host in enumerate(test_hosts):
        lst = np.zeros(8)
        for u in range(len(models)):
            lst += coefs[u] * y_preds[u][i]
        lst = lst.tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)

print('Submission written successfully')