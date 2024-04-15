from sklearn.datasets import load_iris
from sklearn import tree, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


data = load_iris()

features =data.data
target = data.target

plt.subplot(2,2,2)
plt.scatter(features[:,0], features[:,1], c=target,marker='o',cmap='viridis')


Classificador = MLPClassifier(hidden_layer_sizes = (100), alpha=2, max_iter=2000)
Classificador.fit(features,target)


predicao = Classificador.predict(features)

plt.figure(figsize=(16,8))
plt.subplot(2,2,4)
plt.scatter(features[:,0], features[:,1], c=predicao,marker='d',cmap='viridis',s=150)
plt.scatter(features[:,0], features[:,1], c=target,marker='o',cmap='viridis',s=15)
plt.show()

metrics.ConfusionMatrixDisplay.from_estimator(Classificador, features, target,include_values=True,display_labels=data.target_names)
plt.show()
