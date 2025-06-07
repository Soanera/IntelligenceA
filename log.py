import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Charger le modèle MobileNetV2 pré-entraîné
model = MobileNetV2(weights='imagenet')

# Fonction pour charger et prédire l’image
def load_and_predict():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Charger l’image et l’afficher dans l’interface
    img = Image.open(file_path).resize((224, 224))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Préparer l’image pour MobileNetV2
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Prédire
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]

    # Afficher les résultats dans le label résultat
    result_text = "Top 3 prédictions :\n"
    for i, (imagenet_id, name, prob) in enumerate(decoded):
        result_text += f"{i+1}. {name} ({prob*100:.2f}%)\n"
    result_label.config(text=result_text)

# Création de la fenêtre Tkinter
root = tk.Tk()
root.title("Classement et Prédiction d'Image")

# Bouton pour charger l’image
btn = Button(root, text="Charger une image", command=load_and_predict)
btn.pack(pady=10)

# Label pour afficher l’image
img_label = Label(root)
img_label.pack()

# Label pour afficher le résultat
result_label = Label(root, text="Aucune image chargée", font=("Helvetica", 12))
result_label.pack(pady=10)

root.mainloop()