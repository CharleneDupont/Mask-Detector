<h1 align="center">Face mask detector</h1> 

<div align= "center">
  <h4>Depuis la crise sanitaire, de nombreuses mesures ont été prises pour lutter contre le virus notamment le port du masque dans les espaces publiques. <br> 
    Ce projet avait pour but de tenter de reproduire les technologies récemment mises en place pour contrôler le port du masque
</h4>
</div>

## Sommaire
  - [Contexte et Objectifs](#Contexte-et-objectifs)
  - [Dataset](#Dataset)
  - [Modèles utilisés](#Modèles-utilisés)
  - [Video detector set up](#VideoSetUp)
  - [Image detector set up](#ImageSetUp)
  - [Limites et amélioration](#Limites-et-amélioration)
 


## Contexte et objectifs
Ce projet s’articule en deux modules

Création d’une web application permettant de détecter les personnes masquées à partir d’une image importée

![Live Demo](https://github.com/CharlieDpt/Mask-Detector/blob/main/Mask%20Detection%20App.gif)



## Dataset
 Le dossier "dataset" est issus d'un webscrapping sur les images de Bing avec Selenium : 2361 images
 
 Il contient les deux dossiers d'images utilisés pour le projet :
 - "Mask" : 1144 images avec des visages portant un masque
 - "NoMask" : 1217 images avec des visages sans masques.

*N.B : Les deux datasets ont été créés automatiquement via le fichier BingWebScrapping.ipynb. Bien qu'un filtre mannuel ait été fait il peut comporter des irrégularités.*

## Modèles utilisés

### Modèle de détection des visages

Ce modèe a été repris directement depuis Open-CV et est disponnible également ici : https://github.com/opencv/opencv/tree/3.4/data/haarcascades

### Modèle de détection des masques

Une fois le visage identifié via le modèle de détection des visages, celui-ci est envoyé au modèle de détection des masques.

Nous avons entrainé le modèle comme suit : 

1 - Preprocessing : 
Redimension des images et data-augmentation (flip , contrast) 

2 - Transfert Learning : apprentissage sur les 50 dernières couches du modèle CNN pré-entrainé InceptionV3

Le fichier permettant d'entrainter le modèle est : **CovidMaskDetection_V4.ipynb**

## Video detector Set up

Ce projet a été codé en python, et utilise différentes librairies. 

Pour faire fonctionner la détéction de masque via la webcam l'installation suivante est nécessaire :

- Créer un environnement virtuel python
- Télécharger requirements.txt
- Installer les versions des librairies nécessaires présentes dans requirements.txt
- Lancer video_live_detection.py

## Image detector set up
** TO DO **

## Limites et amélioration

1. Limites
Le modèle de détection du visage ne détecte pas tous les visages sur une image 
Il trouve des visages qui n’en sont pas

2. Améliorations :
Détecter en fonction de la probabilité de la classe si le masque est porté en dessous du nez ou du menton
Développer la détection video live dans une application web ou smartphone

