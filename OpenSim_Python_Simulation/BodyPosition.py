
import opensim as osim
import math
import csv

# Charger le modèle
model = osim.Model('Lumbar_C_210.osim')
state = model.initSystem()

# Accéder à la coordonnée du joint que vous souhaitez fixer
coord = model.getCoordinateSet().get('flex_extension')

# Obtenir le corps dont vous voulez la position
body = model.getBodySet().get('Abdomen')

# Initialiser l'angle de départ
angle = -1.222

# Ouvrir un fichier CSV pour écrire les données
with open('joint_positions.csv', 'w', newline='') as csvfile:
    # Créer un objet writer
    writer = csv.writer(csvfile)
    
    # Écrire l'en-tête du fichier CSV
    writer.writerow(['Angle (degrees)', 'Position X', 'Position Y', 'Position Z'])

    # Utiliser une boucle while pour parcourir les angles
    while angle <= 0.4538:
        # Convertir l'angle en radians
        radian_value = math.radians(math.degrees(angle))
        
        # Fixer la valeur de la coordonnée à cet angle
        coord.setValue(state, radian_value)
        
        # Optionnel : Fixer la vitesse à zéro si nécessaire
        coord.setSpeedValue(state, 0)
        
        # Réaliser l'état pour prendre en compte la nouvelle position
        model.realizePosition(state)
        
        # Récupérer la position du corps par rapport au repère global
        position = body.getPositionInGround(state)

        # Écrire l'angle et la position dans le fichier CSV
        writer.writerow([angle, position[0], position[1], position[2]])
        
        # Imprimer l'angle et la position correspondante
        print(f"Position at {angle} degrees: {position}")
        
        # Incrémenter l'angle de 15 degrés pour la prochaine itération
        angle += 0.01

    radian_value = math.radians(math.degrees(0.4538))
    # Fixer la valeur de la coordonnée au dernier angle
    coord.setValue(state, radian_value)
    # Optionnel : Fixer la vitesse à zéro si nécessaire
    coord.setSpeedValue(state, 0)
    # Réaliser l'état pour prendre en compte la nouvelle position
    model.realizePosition(state)
    # Récupérer la position du corps par rapport au repère global
    position = body.getPositionInGround(state)
    # Écrire l'angle et la position dans le fichier CSV
    writer.writerow([angle, position[0], position[1], position[2]])
    # Imprimer l'angle et la position correspondante
    print(f"Position at {angle} degrees: {position}")

    print("Les données ont été enregistrées dans 'joint_positions.csv'.")


