import numpy as np
import cv2
import time
import csv

MATCH_COUNT = 20
FLANN_INDEX_KDTREE = 1
DEFAULT_CHECK = 50

def siftMatcher(des1, des2, check = DEFAULT_CHECK):
    # paramètre du matcher
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 1)
    
    search_params = dict(checks = check)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)


    # on vérifie si le match est à une distance acceptable
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    return len(good) > MATCH_COUNT
    

sift = cv2.SIFT_create()

img_base = cv2.imread('./tests/temoin2.png', cv2.IMREAD_GRAYSCALE)
_, des1 = sift.detectAndCompute(img_base, None)

# lecture de nos données
data = np.load("./data/descripters.npy", allow_pickle=True)

stats = []

mini = 40
maxi = 100
pas = 5
for match_count in range(mini, maxi+1, pas):
    cpt = 0
    
    tic = time.process_time()
    for i in data:
        desc = np.array(i["desc"], dtype=np.float32)        
        
        if siftMatcher(des1, desc, match_count) is True:
            cpt += 1
            
    toc = time.process_time()
            
    stats.append([
        match_count, cpt, round(toc - tic, 4)
    ])
    
head = ["Valeur de check", "Nombre d'images reconnu", "Temps"]

with open('./data/stat_match_count.csv', 'w', newline='') as file:
    # 2. Create a CSV writer
    writer = csv.writer(file)
    # 3. Write data to the file
    writer.writerow(head)
    for row in stats:
        writer.writerow(row)
        