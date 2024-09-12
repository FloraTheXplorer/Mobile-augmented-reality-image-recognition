import numpy as np
import cv2
import csv
import time

MATCH_COUNT = 20
CHECKS = 50
FLANN_INDEX_KDTREE = 1
FILE_TO_FIND = 'aff_tableau8-oth.png'

def siftMatcher(des1, des2, check = CHECKS, tree = 1):
    # paramètre du matcher
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = tree)
    
    # valeur évaluée
    search_params = dict(checks = check)
        
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
reco = 0
stop_time = 2 #secondes
looping = True

stats = []

mini = 10
maxi = 100
pas = 10

for check in range(mini, maxi+1, pas):
    for tree in range(1, 5):
        nbMatches = 0
        matchesImages = []
        tic = time.process_time()
        print("Checking for tree = {} and check = {}".format(tree, check))
        for i in data:
            desc = np.array(i["desc"], dtype=np.float32)
            name = i["name"]
            
            matches = siftMatcher(des1, desc, check = check, tree = tree)
            if(name == FILE_TO_FIND):
                nbMatches = matches
            matchesImages.append({
                "image": name, "matches": matches
            })

        tac = time.process_time()
        matchesImages = sorted(matchesImages, key=lambda d : d["matches"], reverse=True)   
        stats.append([
            tree, check, round(tac - tic, 4), any(d["image"] == FILE_TO_FIND for d in matchesImages[:5]), nbMatches
         ])


head = ["Tree", "Checks", "Time", "Recognize", "Number of matches"]
with open('./data/stat_multiple.csv', 'w', newline='') as file:
    # 2. Create a CSV writer
    writer = csv.writer(file)
    # 3. Write data to the file
    writer.writerow(head)
    for row in stats:
        writer.writerow(row)     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    