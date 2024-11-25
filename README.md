# IA_Object_Detection
# Ryad BELARBI
# Ameni FEKI
# Projet Mask R-CNN : Classification des Fleurs

Ce projet utilise Mask R-CNN pour classifier les images de fleurs en deux catégories : **Tulipe** et **Non-Tulipe**. Ce README fournit des détails complets sur les tests d'entraînement effectués avec différents nombres d'époques (10, 20, 30), les résultats obtenus, les observations sur le surapprentissage (overfitting), et des visualisations clés pour évaluer les performances du modèle.


## voici les 10 lignes et de chaques execution d'epoques
## 10 lignes de l'entrainement sur 10 epoques
![Texte alternatif](./lign10_10.png "Texte au survol")

## Courbes de perte d'entrainement
![Texte alternatif](./loss_10.png "Texte au survol")

## Courbes de pertes de validation
![Texte alternatif](./val_loss_10.png "Texte au survol")

## 10 lignes de l'entrainement sur 20 epoques 

![Texte alternatif](./lign10_20.png "Texte au survol")
## courbes de pertes d'entrainement 
0
![Texte alternatif](./loss_20.png "Texte au survol")
## courbes de pertes de validation 

![Texte alternatif](./val_loss_20.png "Texte au survol")
## 10 lignes de l'entrainement sur 30 epoques 

![Texte alternatif](./lign10_30.png "Texte au survol")
## courbes de pertes d'entrainement

![Texte alternatif](./loss_30.png "Texte au survol")
## courbes de pertes de validation 

![Texte alternatif](./val_loss_30.png "Texte au survol")




## Installation des Dépendances

Ce projet utilise plusieurs bibliothèques pour l’entraînement du modèle Mask R-CNN, la manipulation d’images, et la visualisation de données. Suivez les instructions ci-dessous pour installer les packages nécessaires avec les versions compatibles.

### Commandes d'Installation

Exécutez les commandes suivantes pour installer les dépendances :

```bash
# Installer Keras
python -m pip install Keras==2.2.4 Keras-Applications==1.0.8 Keras-Preprocessing==1.1.2

# Installer TensorFlow
python -m pip install tensorflow==1.15.0 tensorflow-estimator==1.15.1 tensorboard==1.15.0

# Installer Scikit-Image (en installant également ses dépendances courantes)
python -m pip install scikit-image==0.16.2

# Autres dépendances essentielles
python -m pip install numpy==1.21.6 scipy==1.7.3 pandas==1.0.3 matplotlib==3.5.3 Pillow==9.5.0

# Packages supplémentaires pour la manipulation et l'affichage des données
python -m pip install seaborn==0.11.2 tqdm==4.66.6 requests==2.31.0

# Pour la manipulation JSON et les fichiers de labels (Labelme2COCO, Pybboxes)
python -m pip install jsonschema==4.17.3 labelme2coco==0.2.6 pybboxes==0.1.6

# Installations pour des modules auxiliaires comme l'interface utilisateur et le client Jupyter
python -m pip install jupyter-client==7.4.9 jupyter-core==4.12.0 nest-asyncio==1.6.0

# Packages pour le développement d'algorithmes et calculs mathématiques avancés
python -m pip install absl-py==2.1.0 cloudpickle==2.2.1 dask==2022.2.0

# Packages de manipulation d'images et de fichiers multimédia
python -m pip install opencv-python==3.4.13.47 imageio==2.10.4

# Installations pour les réseaux de neurones et autres applications IA
python -m pip install torch==1.13.1 sahi==0.11.18

# Installations pour gérer les fichiers et le système
python -m pip install fsspec==2023.1.0 importlib-resources==5.12.0 importlib-metadata==6.7.0

# Installations pour la manipulation de JSON et la gestion des chemins de fichiers
python -m pip install attrs==24.2.0 pyparsing==3.1.4 typing-extensions==4.7.1

# Gestion des demandes réseau et de l'interface utilisateur
python -m pip install urllib3==1.26.16 termcolor==2.3.0 colorama==0.4.6

# Autres utilitaires importants pour le système et les structures de données
python -m pip install setuptools==68.0.0 wheel==0.42.0 wrapt==1.16.0
```
---

## Structure du Projet

- **Dossier `data_t`** : Contient les images et annotations JSON pour les ensembles d'entraînement et de validation. Les images sont annotées avec les classes "tulipe" et "non_tulipe".
- **Dossier `logs`** : Contient les journaux d'entraînement générés par TensorBoard, permettant le suivi des pertes et des métriques de performance.
- **Fichier `mask_rcnn_coco.h5`** : Fichier de poids pré-entraînés sur le dataset COCO pour initialiser Mask R-CNN.

---

## Configurations et Hyperparamètres

- **Classes** : Le modèle est entraîné pour reconnaître deux classes : "tulipe" (classe 1) et "non_tulipe" (classe 2) et la classe background.
- **Taux d'apprentissage** : `LEARNING_RATE` est fixé à 0.001 pour assurer une convergence stable et éviter les oscillations dans les pertes.
- **Nombre d'époques** : Plusieurs configurations d'époques ont été testées pour évaluer l'impact sur les performances du modèle et identifier le point optimal avant le surapprentissage.

---

## Expériences et Résultats


## Choix Optimal : Entraînement sur 20 Époques

Après avoir comparé les performances des différentes configurations, l'entraînement sur **20 époques** s'est révélé être la meilleure option. À ce stade, le modèle atteint un équilibre entre précision d'entraînement et généralisation sur l'ensemble de validation. Avec plus de 20 époques, le modèle commence à tomber dans le surapprentissage, apprenant trop de caractéristiques spécifiques aux données d'entraînement.

---

## Test avec epoque 10 
## image1 
![Texte alternatif](./fleur.jpg "Texte au survol")

## voici son resultat 

![Texte alternatif](./output_fleur_10.png "Texte au survol")

## image 2
![Texte alternatif](./fleur2.jpg "Texte au survol")

## voici son resultat 

![Texte alternatif](./output_fleur2_10.png "Texte au survol")


## image 3
![Texte alternatif](./fleur3.jpg "Texte au survol")

## voici son resultat 

![Texte alternatif](./output_fleur3_10.png "Texte au survol")


## image 4
![Texte alternatif](./image4.jpg "Texte au survol")

## voici son resultat 

![Texte alternatif](./output_image4_10.png "Texte au survol")


## Test avec epoque 20 
## image1 
![Texte alternatif](./fleur.jpg "Texte au survol")

## voici son resultat 

![Texte alternatif](./output_fleur_20.png "Texte au survol")

## image 2
![Texte alternatif](./fleur2.jpg "Texte au survol")

## voici son resultat 

![Texte alternatif](./output_fleur2_20.png "Texte au survol")


## image 3
![Texte alternatif](./fleur3.jpg "Texte au survol")

## voici son resultat 

![Texte alternatif](./output_fleur3_20.png "Texte au survol")


## image 4
![Texte alternatif](./image4.jpg "Texte au survol")

## voici son resultat 

![Texte alternatif](./output_image4_20.png "Texte au survol")

## Test avec epoque 30 
## image1 
![Texte alternatif](./fleur.jpg "Texte au survol")

## voici son resultat 

![Texte alternatif](./output_fleur_30.png "Texte au survol")

## image 2
![Texte alternatif](./fleur2.jpg "Texte au survol")

## voici son resultat 

![Texte alternatif](./output_fleur2_30.png "Texte au survol")


## image 3
![Texte alternatif](./fleur3.jpg "Texte au survol")

## voici son resultat 

![Texte alternatif](./output_fleur3_30.png "Texte au survol")


## image 4
![Texte alternatif](./image4.jpg "Texte au survol")

## voici son resultat 

![Texte alternatif](./output_image4_30.png "Texte au survol")


## matrice de confusion du resultat du modele en une seule epoque 
![Texte alternatif](./mat_conf_epch1.png "Texte au survol")


## courbe Roc du resultat du modele en une seule epoque 
![Texte alternatif](./cr_epch01.png "Texte au survol")


## matrice de confusion du resultat du modele en 20 epoques 
![Texte alternatif](./mc_epoch20.png "Texte au survol")


## matrice de confusion du resultat du modele en 20 epoques 
![Texte alternatif](./cr_epch20.png "Texte au survol")


## matrice de confusion du resultat du modele en 30 epoques 
![Texte alternatif](./mc_epoch30.png "Texte au survol")


## matrice de confusion du resultat du modele en 30 epoques 
![Texte alternatif](./cr_epoch30.png "Texte au survol")

