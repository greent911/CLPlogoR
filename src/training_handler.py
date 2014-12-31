import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
import sys

class TrainingHandler():
    def __init__(self):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        self.flann = cv2.FlannBasedMatcher(index_params,search_params)
        self.DISTCOMPAREFACTOR = 0.7
        
    def drawKeyPoints(self, img1, img2, gkp1, gkp2, num=-1):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        nWidth = w1+w2
        nHeight = h1+h2
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
        newimg[:h2, :w2] = img2
        newimg[h2:h2+h1, w2:w2+w1] = img1

        maxlen = min(len(gkp1), len(gkp2))
        if num < 0 or num > maxlen:
            num = maxlen
        for i in range(num):
            pt_a = (int(gkp2[i].pt[0]), int(gkp2[i].pt[1]))
            pt_b = (int(gkp1[i].pt[0]+w2), int(gkp1[i].pt[1]+h2))
            cv2.line(newimg, pt_a, pt_b, (255, 0, 0))
        cv2.imshow('Matches',newimg)

    def feature_matching(self, img1path, img2path):
        """Feature Matching
        """
        img1 = cv2.imread(img1path) # queryImage
        img2 = cv2.imread(img2path) # trainImage

        # Initiate SIFT detector
        sift = cv2.SIFT()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        
        # Need only good matches
        matches = self.flann.knnMatch(des1,des2,k=2)
        goodmatches = []
        for i,(m,n) in enumerate(matches):
            if m.distance < self.DISTCOMPAREFACTOR*n.distance:
                goodmatches.append(m)
                # print i
        # print goodmatches
        # print goodmatches[0].distance
        # print matches[233][0].distance
        
        # Sort goodmatches by distance
        indices = range(len(goodmatches))
        indices.sort(key=lambda i: goodmatches[i].distance)

        gkp1 = []
        for i in indices:
            gkp1.append(kp1[goodmatches[i].queryIdx])
        gkp2 = []
        for i in indices:
            gkp2.append(kp2[goodmatches[i].trainIdx])

        self.drawKeyPoints(img1,img2,gkp1,gkp2)

if __name__ == '__main__':
   trHandler = TrainingHandler()
   trHandler.feature_matching('box.png','box_in_scene.png')
