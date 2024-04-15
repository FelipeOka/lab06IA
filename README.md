# CC7711 - Inteligência Artificial e Robótica

## Laboratório 6 - RNA Classificação de Padrões

Alunos:
- Fabio Augusto Schiavi Morpanini | RA: 22.121.094-1
- Felipe de Campos Oka            | RA: 22.121.001-6

### Imports e Data
```python
from sklearn.datasets import load_iris
from sklearn import tree, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = load_iris()
features =data.data
target = data.target
```
### Código Iris data com PCA

```python
pca = PCA(n_components=2, whiten=True, svd_solver='randomized')
pca = pca.fit(features)
pca_features = pca.transform(features)
print('Mantida %5.2f%% da informação do conjunto inicial de dados'%(sum(pca.explained_variance_ratio_)*100))

plt.subplot(2,2,2)
plt.scatter(pca_features[:,0], pca_features[:,1], c=target,marker='o',cmap='viridis')

ClassificadorPCA = MLPClassifier(hidden_layer_sizes = (100), alpha=1.5, max_iter=600)
ClassificadorPCA.fit(pca_features,target)

predicao = ClassificadorPCA.predict(pca_features)

plt.figure(figsize=(16,8))
plt.subplot(2,2,4)
plt.scatter(pca_features[:,0], pca_features[:,1], c=predicao,marker='d',cmap='viridis',s=150)
plt.scatter(pca_features[:,0], pca_features[:,1], c=target,marker='o',cmap='viridis',s=15)
plt.show()

metrics.ConfusionMatrixDisplay.from_estimator(ClassificadorPCA, pca_features, target,include_values=True,display_labels=data.target_names)
plt.show()
```
### Resultados
<div align="center">
<img src="https://github.com/KaburauNero/lab06IA/assets/92650933/d605674b-270e-497c-aff9-f44334f57131" width="300px" />
<img src="https://github.com/KaburauNero/lab06IA/assets/92650933/191db6d6-3ab3-4461-8cad-c9e0d8613924" width="500px" />
</div>

### Matriz de Confusão sem PCA
<div align="center">
<img src="https://github.com/KaburauNero/lab06IA/assets/92650933/afcbeba4-d5d3-4285-b33e-be30db588527" width="600px" />
</div>

### Cógido Iris Data sem PCA

```python
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
```
### Resultados
<div align="center">
<img src="https://github.com/KaburauNero/lab06IA/assets/92650933/f5e3853d-2db0-4ecb-a7e8-004fe508b98f" width="300px" />
<img src="https://github.com/KaburauNero/lab06IA/assets/92650933/60fc0aa7-30b7-4ef3-853e-ef3137d61da0" width="500px" />
</div>

### Matriz de Confusão sem PCA
<div align="center">
<img src="https://github.com/KaburauNero/lab06IA/assets/92650933/d3ce3e9c-bb1f-43de-9581-f35933a9a98a" width="600px" />
</div>

## Conclusão
Os resultados que utilizam da PCA podem ser considerados mais promissores, uma vez que os pontos gerados estão dispersos em um espaço menor de dimensionalidade. <br>
As diferenças dos pontos dispersos acabam causando uma pequena diferença na relação virginica - versicolor, para ambas linhas e colunas. Tamanha diferença serve como prova da acurácia dos dados.

