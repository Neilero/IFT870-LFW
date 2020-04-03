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
data = pd.concat([pd.DataFrame(X, index=y.index), y], axis=1)

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
Nous pouvons remarquer que les résultats changent à chaque exécution et que ces résultats sont parfoit oscillant,
rendant la précision de la validation croisée douteuse. Puisque nous utilisons ici un autre score mais que les résultats
semble toujours aussi incertains, nous pouvons imaginer que c'est le modèle `KMeans` qui est inadapté à nos données.
"""

# %%
"""
## Analyse avec DBSCAN

*Utiliser le coéfficient de silhouette pour déterminer les meilleurs valeurs de paramètres
(nombre minimum d’éléments dans un cluster `min_samples`, et rayon du voisinage autour
de chaque donnée `eps`) pour la méthode DBSCAN avec `min_samples` dans l’intervalle
[1, ..., 10] et eps dans l’intervalle [5, ..., 15]*
"""

# %%
# pre-compute pair-wise distances
distances = squareform(pdist(X))

eps_list = range(5, 16)
min_samples_list = range(11)

scores = []
best_dbscan = None
best_dbscan_score = 0
best_dbscan_params = None
for eps, min_samples in tqdm(product(eps_list, min_samples_list), total=len(eps_list)*len(min_samples_list),
                             desc="Searching best eps and min_samples for DBSCAN"):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    predict = dbscan.fit_predict(X, y)

    score = silhouette_score(distances, predict) if len(np.unique(predict)) > 1 else 0  # if only noise found, score = 0

    scores.append(score)
    if score > best_dbscan_score:
        best_dbscan_score = score
        best_dbscan = dbscan
        best_dbscan_params = (eps, min_samples)

print(f"Best eps = {best_dbscan_params[0]}, best min_samples = {best_dbscan_params[1]}")
print(f"Number of cluster(s) found : {len(np.unique(best_dbscan.labels_)) - (1 if -1 in best_dbscan.labels_ else 0)}")

scores = np.reshape(scores, (len(eps_list), len(min_samples_list)))
sns.heatmap(scores, xticklabels=min_samples_list, yticklabels=eps_list)
plt.xlabel("min_samples")
plt.ylabel("eps")
plt.show()

# %%
"""
D'après les résultats du graph ci-dessus, on peut remarquer que les meilleurs scores de silhouette sont obtenu pour un
`eps` élevé et un `min_sample` moyen à élevé. Nous pouvons ainsi déduire de ces résultats que les meilleurs clusters au
sens du coéfficient de silhouette sont les plus gros. Il semble aussi que les valeurs de score soient plus sensible aux
modifications d'`eps` que de `min_samples`.
"""

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
# remove warning printing more than 20 images
plt.rcParams.update({'figure.max_open_warning': 0})

def show_cluster_samples(data, X, y, eps):
    predict = DBSCAN(eps=eps, min_samples=3, n_jobs=-1).fit_predict(X, y)
    data["cluster"] = predict

    image_count = 5

    for i, (cluster, clusterData) in enumerate(data.groupby("cluster")):
        if cluster == -1:
            continue

        fig, axes = plt.subplots(ncols=image_count, figsize=(10, 3), subplot_kw={'xticks': (), 'yticks': ()})
        fig.suptitle(f"Échantillon d'images du cluster {cluster} (taille réel : {clusterData.shape[0]})", fontsize=16)

        if cluster == 0:
            fig.text(0.5, 1, f"Clusters pour eps={eps}", horizontalalignment='center', fontsize=20)

        for j in range(image_count):
            if j >= clusterData.shape[0]:
                axes[j].set_visible(False)
                continue

            row_index = clusterData.index[j]
            axes[j].imshow(faces.images[row_index])
            axes[j].set_title(faces.target_names[clusterData["target"][row_index]])


for eps in eps_list:
    show_cluster_samples(data, X, y, eps)

# %%
"""
Pour tous les clusters nous avons choisi de ne pas afficher les bruits (données pour lesquelles la valeur du 
cluster est égale à `-1`). En effet, bien qu'elles soient regroupées dans un même ensemble, ces données ne présentent 
pas de caractéristiques commune mis à part d'être considérée différentes de toutes les autres données par `DBSCAN` 

### Analyse pour `eps=5`

Nous observons ici un seul cluster composé de 4 photos de la même personne : `Junichiro Koizumi`.

### Analyse pour `eps=6`

Là encore, nous observons le cluster trouvé précédemment mais cette fois-ci, avec une photo de plus de la même personne.

### Analyse pour `eps=7`

Pour cette valeur d'`eps` nous observons une nette amélioration. En effet, l'algorithme a réussi à identifier 12 
groupes de photos. `DBSCAN` semble ici avoir regroupé les photos selon l'orientation du visage. On note tout de même 
que les clusters sont tous assez petits (6 photos pour le gros cluster). 

### Analyse pour `eps=8`

À partir de cette valeur d'`eps`, il semble que `DBSCAN` ait plus de mal à séparer les visages. En effet, bien qu'on
puisse noter la présence de 4 petits clusters (avec 5 photos au maximum), on remarque aussi l'émergence d'un grand
cluster (avec 256 photos). De plus, les petits clusters semble semble ne plus uniquement regrouper des personnes selon
l'orientation du visage mais aussi selon la personne (seul le dernier cluster est composé de deux personnalités
différentes). A contrario le premier cluster regroupe beaucoup de personnalités différentes dans des orientations du
visage différents et avec des traits de caractères (comme la présence de rides / cernes, la bouche ouverte...)
différents. On peut donc imaginer que la valeur d'`eps` est trop grande et induit de trop grand regroupement.

### Analyse pour `eps=9`

Les résultats observé précédemment se confirme avec la présence ici d'un cluster encore plus grand (694 images) et d'un
petit cluster (3 images) composé uniquement de la personnalité `Jiang Zemin` possédant des traits assez spécifique (en
particulier de très grandes lunettes bien visible).

### Analyse pour `eps=10`

Légère amélioration ici des résultats avec 2 petits clusters (3 photos chacun) avec une orientation du visage
intra-cluster similaire et avec le premier cluster composé de différentes personnalité. En revanche, nous notons aussi
que le gros cluster a quasiement doublé de taille.

### Analyse pour `eps=11`, `eps=12`, `eps=13`, `eps=14`, `eps=15`

Pour toutes ces valeurs d'`eps`, nous n'avons désormais plus qu'un très gros cluster qui croit avec l'augmentation
d'`eps`.

### Bilan

Contrairement à ce que nous montrait le score de silhouette, les résultats que nous observons ici semblent nous indiquer
que de nombreux petits clusters seraient préférables à de gros clusters avec le modèle `DBSCAN`.
"""