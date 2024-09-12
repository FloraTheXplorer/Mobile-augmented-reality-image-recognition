import numpy as np
import cv2 as cv2
import os

path = "./ressources/"
  
# Change the directory
os.chdir(path)

sift = cv2.SIFT_create()

data = []
        
for file in os.listdir():
    # Lecture de tout les fichiers de notre dossier
    if file.endswith(".png"):
        file_path = file
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        # calcul des points d'intérêts
        # des est un tableau des descripteurs
        kp, des = sift.detectAndCompute(img, None)
        
        
        data.append({
            "name": file_path,
            "desc": des.tolist()   
        })
 
    
os.chdir("../data/")
# enregistrement de nos données dans un fichier numpy
np.save("descripters.npy", data)