# %%
"""
# IFT870 - TP4

Auteur : Aurélien Vauthier (19 126 456)
"""

# %%
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, make_scorer, silhouette_score
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import cdist, pdist, squareform
from itertools import product
from tqdm import tqdm
import numpy as np
import pandas as pd
# %matplotlib notebook
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Chargement d'un ensemble de données de faces de personnages connus
from sklearn.datasets import fetch_lfw_people

# %%
faces = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

# %%
# format des images et nombres de clusters
print("Format des images: {}".format(faces.images.shape))
print("Nombre de classes: {}".format(len(faces.target_names)))

# %%
# nombre de données par cluster
nombres = np.bincount(faces.target)
for i, (nb, nom) in enumerate(zip(nombres, faces.target_names)):
    print("{0:25} {1:3}".format(nom, nb), end='   ')
    if (i + 1) % 3 == 0:
        print()

# %%
# Affichage des 10 premières faces
fig, axes = plt.subplots(2, 5, figsize=(10, 6),
                         subplot_kw={'xticks': (), 'yticks': ()})
for nom, image, ax in zip(faces.target, faces.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(faces.target_names[nom])

# %%
"""
*Pour commencer, les données ne sont pas équilibrées car certains personnages sont beaucoup
plus représentés que d’autres. Pour pallier à cela, filter les données pour ne conserver que
40 visages au maximum par personne.*
"""

# %%
# Convert data array to DataFrame and append targets
data = pd.DataFrame(faces.data)
data["target"] = faces.target

# keep the first 40 data for each target
data = data.groupby("target").head(40)

# show results
data.head()

# %%
"""
*Ensuite, appliquer une réduction de la dimension à 100 composantes et une normalisation
en utilisant le modèle `PCA()` de `sklearn` avec les options `whiten=True` et `random_state=0`.*
"""

# %%
pca = PCA(100, whiten=True, random_state=0)

X = pca.fit_transform(data.drop("target", axis=1))
y = data["target"]
data = pd.concat([pd.DataFrame(X), y], axis=1)

# show results
data.head()

# %%
"""
## Analyse avec K-Means

*Implémenter la méthode du coude (Elbow method) pour essayer de déterminer un
nombre de clusters optimals dans l’ensemble suivant [40, 45, 50, 55, 60, ..., 80] sans
utiliser les données réelles (noms associés aux images). La mesure de score à utiliser
pour tout nombre de clusters k est la suivante : moyenne des distances euclidiennes des
données à leur plus proche centre de cluster pour le modèle à k clusters. Analyser le
résultat et donner vos conclusions.*
"""

# %%
# https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
Ks = range(40, 81, 5)
mean_min_dists = []

for k in tqdm(Ks, desc="Computing mean of min distances to closest centroid..."):
    kmean = KMeans(n_clusters=k, n_jobs=-1)
    kmean.fit(X)

    distances = cdist(X, kmean.cluster_centers_, "euclidean")
    mean_min_dist = np.mean(np.min(distances, axis=1))
    mean_min_dists.append(mean_min_dist)

plt.plot(Ks, mean_min_dists, 'bx-')
plt.xlabel("Nombre de cluster K")
plt.ylabel("Moyenne des distances")
plt.title("Moyennes des distances des données à leur plus proche centre par rapport au nombre de cluster")
plt.show()

# %%
"""
D'après les résultats du graphique ci-dessus, nous pouvons constater que le nombre de cluster optimal ne semble pas
pouvoir être deviné par la méthode du coude. En effet, nous n'observons pas ici de "coude", la distance moyenne des
données au centre le plus proche décroit de façon linéaire avec l'augmentation du nombre de cluster. Cela pourrait être
dû à la [malédiction de la dimensionnalité](https://fr.wikipedia.org/wiki/Fl%C3%A9au_de_la_dimension). En effet, on peut
constater que lorsque le nombre de dimension augmente la distance entre les points tend à s'uniformiser.
"""

# %%
"""
*Appliquer une approche de validation croisée en divisant les données en 10 parties et en
utilisant les données réelles et le score `Adjusted_Rand_Index` (ARI) pour déterminer
un nombre de clusters optimal dans l’ensemble [40, 45, 50, 55, 60, ..., 80]. Analyser le
résultat et donner vos conclusions.*
"""

# %%
model = KMeans(n_jobs=-1)
param_grid = {
    "n_clusters": Ks
}

# convert metric to scorer
scorer_ARI = make_scorer(adjusted_rand_score)

best_kmean = GridSearchCV(model, param_grid, cv=10, scoring=scorer_ARI, n_jobs=-1, verbose=1)
best_kmean.fit(X, y)

print(f"Best k fond : {best_kmean.best_params_['n_clusters']}")

plt.plot(Ks, best_kmean.cv_results_["mean_test_score"], 'bx-')
plt.xlabel("Nombre de cluster K")
plt.ylabel("Moyenne des scores")
plt.title("Moyennes des scores en fonction du nombre de cluster")
plt.show()

# %%
"""
Test
"""

# %%
"""
# Analyse avec DBSCAN

*Utiliser le coéfficient de silhouette pour déterminer les meilleurs valeurs de paramètres
(nombre minimum d’éléments dans un cluster `min_samples`, et rayon du voisinage autour
de chaque donnée `eps`) pour la méthode DBSCAN avec `min_samples` dans l’intervalle
[1, ..., 10] et eps dans l’intervalle [5, ..., 15]*
"""

# %%
model = DBSCAN(n_jobs=-1)
param_grid = {
    "min_samples": range(11),
    "eps": range(5, 16)
}
distances = squareform(pdist(X))

best_dbscan = None
best_dbscan_score = 0
best_dbscan_params = None
for eps, min_samples in product(range(5, 16), range(11)):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    predict = dbscan.fit_predict(X, y)

    score = silhouette_score(distances, predict) if len(np.unique(predict)) != 1 else 0

    if score > best_dbscan_score:
        best_dbscan_score = score
        best_dbscan = dbscan
        best_dbscan_params = (eps, min_samples)

print(f"Best eps = {best_dbscan_params[0]}, best min_samples = {best_dbscan_params[1]}")
print(f"Number of cluster found : {len(best_dbscan.classes_) - 1}")


# %%
"""
*En fixant le paramètre `min_samples = 3`, appliquer DBSCAN en faisant varier le paramètre
`eps` dans l’intervalle [5, ..., 15]. Observer des échantillons d’images des clusters
pour chaque rayon dans l’intervalle [5, ..., 15], et tenter de déterminer la signification
sémantique des clusterings estimés. Elle peut correspondre à un clustering suivant les
personnages, ou suivant d’autres caractéristiques commune comme l’orientation du visage,
l’arrière plan, le port de lunette, etc. Lister vos conclusions pour chaque valeur de
`eps`.*
"""

# %%

