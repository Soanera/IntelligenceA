import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import urllib.request
import matplotlib.pyplot as plt

# 📥 Télécharger la liste des classes ImageNet si pas déjà présente
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
filename = "imagenet_classes.txt"
if not os.path.exists(filename):
    urllib.request.urlretrieve(url, filename)

# 📚 Charger les classes
with open(filename) as f:
    classes = [line.strip() for line in f.readlines()]

# 🧠 Charger le modèle pré-entraîné
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

# 🔄 Prétraitement des images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ✅ Méthode bonus : classification simplifiée
def classifier_prediction(prediction):
    pred = prediction.lower()

    # Chats
    if "cat" in pred:
        return "C’est un chat 🐱"

    # Chiens (plusieurs races détectées)
    elif any(mot in pred for mot in [
        "dog", "whippet", "kelpie", "pembroke", "labrador", "retriever",
        "chihuahua", "terrier", "shepherd", "poodle", "bulldog", "beagle", "husky"
    ]):
        return "C’est un chien 🐶"

    # Oiseaux
    elif any(mot in pred for mot in [
        "bird", "parrot", "magpie", "sparrow", "pigeon", "canary", "woodpecker"
    ]):
        return "C’est un oiseau 🐦"

    # Femmes
    elif "woman" in pred or "girl" in pred or "bride" in pred:
        return "C’est une femme 👩"

    # Hommes
    elif "man" in pred or "boy" in pred or "groom" in pred:
        return "C’est un homme 👨"

    # Humain générique
    elif "person" in pred or "human" in pred:
        return "C’est un humain 🙍"

    # Sinon
    else:
        return f"Autre chose détectée : {prediction}"

# 🔍 Fonction de prédiction + affichage image
def predire_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            _, index = output.max(1)
            prediction = classes[index]
            categorie = classifier_prediction(prediction)

            print(f"\n📷 Image : {os.path.basename(image_path)}")
            print(f"  Prédiction brute : {prediction}")
            print(f"  Catégorie : {categorie}")

            # 👁️ Affichage image
            plt.imshow(image)
            plt.title(f"{os.path.basename(image_path)}\n{categorie}", fontsize=10)
            plt.axis('off')
            plt.show()

    except Exception as e:
        print(f"⚠️ Erreur avec l’image {image_path} : {e}")

# 📁 Dossier contenant les images
dossier_images = "images"

# 🔁 Traiter toutes les images du dossier
for fichier in os.listdir(dossier_images):
    if fichier.lower().endswith((".jpg", ".jpeg", ".png")):
        chemin_complet = os.path.join(dossier_images, fichier)
        predire_image(chemin_complet)
