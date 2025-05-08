import json
import os

# Chemin vers ton fichier d’annotation VGG
vgg_path = './val2/vgg_annotations.json'

# Charger le fichier JSON
with open(vgg_path, 'r') as f:
    data = json.load(f)

# Nouveau dictionnaire pour stocker les annotations modifiées
new_data = {}

for filename, annotation in data.items():
    # Extraire le vrai nom du fichier avant le premier "_jpg"
    if "_jpg" in filename:
        new_name = filename.split('_jpg')[0] + '.jpg'
    else:
        new_name = filename

    # Mettre à jour le nom du fichier dans les annotations
    annotation['filename'] = new_name
    new_data[new_name] = annotation

# Sauvegarder le fichier corrigé
with open('annotations_cleaned.json', 'w') as f:
    json.dump(new_data, f, indent=4)

print("Fichier d'annotation mis à jour avec succès.")
