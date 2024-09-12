import numpy as np
import cv2
import time

FLANN_INDEX_KDTREE = 1

def siftMatcher(des1, des2):
    # paramètre du matcher
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 1)
    
    # valeur évaluée
    search_params = dict()
        
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # check le temps
    matches = flann.knnMatch(des1,des2,k=2)
    
    # on vérifie si le match est à une distance acceptable
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)       
        
    return len(good)
    

# création de l'objet SIFT
sift = cv2.SIFT_create()

img_base = cv2.imread('./tests/temoin3.png', cv2.IMREAD_GRAYSCALE)
_, des1 = sift.detectAndCompute(img_base, None)

data = np.load("./data/descripters.npy", allow_pickle=True)

cpt = 0
sume = 0
stop_time = 2 #secondes
looping = True

for j in range(0, 10):
    cpt = 0
    looping = True
    tic = time.process_time()
    while looping:
        for i in data:
            tac = time.process_time()
            if tac - tic > stop_time:
                looping = False
                break
            desc = np.array(i["desc"], dtype=np.float32)
            name = i["name"]
            
            siftMatcher(des1, desc)
            
            cpt += 1
    sume += cpt
    print(str(cpt) + " fichiers lus")
        
        
print("Moyenne : {}".format(sume / 10))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    