import numpy as np
import cv2 as cv2
import time
import os

MIN_MATCH_COUNT = 20
FLANN_INDEX_KDTREE = 1

def siftMatcher(img_base, img_courant):
    # création de l'objet SIFT
    sift = cv2.SIFT_create()
    
    # calcul des points d'intérêts
    # des est un tableau des descripteurs
    kp1, des1 = sift.detectAndCompute(img_base, None)
    kp2, des2 = sift.detectAndCompute(img_courant, None)

    # paramètre du matcher
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)


    # on vérifie si le match est à une distance acceptable
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img_base.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img_courant = cv2.polylines(img_courant,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)
        return cv2.drawMatches(img_base,kp1,img_courant,kp2,good,None,**draw_params)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        return None
    
        
    
img_base = cv2.imread('./tests/temoin.png', cv2.IMREAD_GRAYSCALE)

        
# Folder Path
path = "./ressources/"
  
# Change the directory
os.chdir(path)

tempsTotal = 0

maxFile = 10

for file in os.listdir():
    
    # Check whether file is in text format or not
    if file.endswith(".png"):
        file_path = file
        img2 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        # img2 = resize(img2)
        print("\nTest pour " + file_path)
        
        tic = time.process_time()
        imgr = siftMatcher(img_base, img2)
        toc = time.process_time()
        
        if imgr is not None:
            # plt.imshow(imgr, 'gray'),plt.show()
            print("Ok pour " + file_path)
        tempsTotal += (toc - tic)
        print(f"Temps de calcul pour {file_path} {toc - tic:0.4f} seconds")
    
print(f"Temps de calcul total {tempsTotal:0.4f} seconds")