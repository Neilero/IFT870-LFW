# %%
"""
# IFT870 - TP4

Auteur : Aurélien Vauthier (19 126 456)
"""

# %%
import sklearn
import numpy as np
import scipy as sp
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


# %%
"""
*Ensuite, appliquer une réduction de la dimension à 100 composantes et une normalisation
en utilisant le modèle `PCA()` de `sklearn` avec les options `whiten=True` et `random_state=0`.*
"""

# %%


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

# %%
"""
*Appliquer une approche de validation croisée en divisant les données en 10 parties et en
utilisant les données réelles et le score `Adjusted_Rand_Index` (ARI) pour déterminer
un nombre de clusters optimal dans l’ensemble [40, 45, 50, 55, 60, ..., 80]. Analyser le
résultat et donner vos conclusions.*
"""

# %%

# %%
"""
# Analyse avec DBSCAN

*Utiliser le coéfficient de silhouette pour déterminer les meilleurs valeurs de paramètres
(nombre minimum d’éléments dans un cluster `min_samples`, et rayon du voisinage autour
de chaque donnée `eps`) pour la méthode DBSCAN avec `min_samples` dans l’intervalle
[1, ..., 10] et eps dans l’intervalle [5, ..., 15]*
"""

# %%

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

# %%
"""
**
"""

# %%
