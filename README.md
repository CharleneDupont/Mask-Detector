<h1 align="center">Face mask detector</h1> 

<div align= "center">
  <h4>Depuis la crise sanitaire, de nombreuses mesures ont été prises pour lutter contre le virus notamment le port du masque dans les espaces publiques. <br> 
    Ce projet avait pour but de tenter de reproduire les technologies récemment mises en place pour contrôler le port du masque
</h4>
</div>


## Contexte et objectifs
Ce projet s’articule en deux modules

Création d’une web application permettant de détecter les personnes masquées à partir d’une image importée

![Live Demo](https://github.com/CharlieDpt/Mask-Detector/blob/main/Mask%20Detection%20App.gif)

## Sommaire

  - [Dataset](#Dataset)
  - [Models](#Modèles Utilisés)
  - [Video detector set up](#VideoSetUp)
  - [Image detector set up](#ImageSetUp)


## Dataset
 Le dossier "dataset" est issus d'un webscrapping sur les images de Bing avec Selenium : 2361 images
 
 Il contient les deux dossiers d'images utilisés pour le projet :
 - "Mask" : 1144 images avec des visages portant un masque
 - "NoMask" : 1217 images avec des visages sans masques.

*N.B : Les deux datasets ont été créés automatiquement via le fichier BingWebScrapping.ipynb. Bien qu'un filtre mannuel ait été fait il peut comporter des irrégularités.*

## Video detector Set up

Ce projet a été codé en python, et utilise différentes librairies. 

Pour faire fonctionner la détéction de masque via la webcam l'installation suivante est nécessaire :

- Créer un environnement virtuel python
- Télécharger requirements.txt
- Installer les versions des librairies nécessaires présentes dans requirements.txt
- Lancer video_live_detection.py
  

Preprocessing : Redimension des images et data-augmentation (flip , contrast) 

Transfert Learning : apprentissage sur les 50 dernières couches du modèle CNN pré-entrainé InceptionV3 

## Image detector set up
** TO DO **

## Limites et amélioration

1. Limites
Le modèle de détection du visage ne détecte pas tous les visages sur une image 
Il trouve des visages qui n’en sont pas

2. Améliorations :
Détecter en fonction de la probabilité de la classe si le masque est porté en dessous du nez ou du menton
Développer la détection video live dans une application web ou smartphone

