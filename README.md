---

# 🧠 Image Classifier avec ResNet50

Ce projet utilise le modèle **ResNet50** pré-entraîné sur ImageNet pour effectuer de la **classification d’images**. Il est capable de reconnaître des objets communs tels que **chiens, chats, oiseaux, humains**, etc., à partir d'images fournies dans un dossier.

---

## 📦 Fonctionnalités BY Filamatriniarivo S.M.Nathalie

* Utilise **PyTorch** et **Torchvision** pour le traitement et l’inférence.
* Chargement automatique de la **liste des classes ImageNet**.
* Prétraitement standard des images pour ResNet50.
* Classification personnalisée simplifiée (animaux, humains, etc.).
* Affichage des images avec les prédictions.

---

## 📁 Structure du projet BY Filamatriniarivo S.M.Nathalie

```
.
├── main.py                # Script principal (celui fourni ci-dessus)
├── imagenet_classes.txt  # Fichier texte contenant les classes (téléchargé automatiquement)
├── images/               # Dossier contenant les images à classifier
└── README.md             # Ce fichier
```

---

## ▶️ Utilisation  BY Filamatriniarivo S.M.Nathalie
  
### 1. Pré-requis

Assurez-vous d’avoir **Python 3.7+** et installez les dépendances nécessaires :

```bash
pip install torch torchvision matplotlib pillow
```

### 2. Ajoutez vos images  BY Filamatriniarivo S.M.Nathalie

Placez vos images dans le dossier `images/`. Formats supportés : `.jpg`, `.jpeg`, `.png`.

### 3. Lancez le script

```bash
python main.py
```

Le script va :

* parcourir toutes les images du dossier `images/`,
* les classer avec **ResNet50**,
* afficher la prédiction brute et une **catégorie simplifiée** (chat 🐱, chien 🐶, oiseau 🐦, etc.),
* et montrer l’image avec un titre personnalisé.

---

## 🧠 Exemple de sortie

```
📷 Image : cat.jpg
  Prédiction brute : tabby cat
  Catégorie : C’est un chat 🐱


## 💡 Notes

* Le modèle **ResNet50** est téléchargé automatiquement via `torchvision.models`.
* La détection "simplifiée" repose sur quelques mots-clés présents dans les noms de classes ImageNet.
* Ce projet est un bon point de départ pour de la **classification personnalisée** ou une future application web/mobile.

## 📜 Licence

Ce projet est fourni à des fins éducatives. Aucune licence spécifique n’est attachée. Utilisation libre sous réserve de respect des modèles tiers (PyTorch, ImageNet).

