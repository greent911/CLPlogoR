import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
import sys

class TrainingHandler():
    def drawKeyPoints(self, img, template, skp, tkp, num=-1):
        h1, w1 = img.shape[:2]
        h2, w2 = template.shape[:2]
        nWidth = w1+w2
        nHeight = h1+h2
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
        newimg[:h2, :w2] = template
        newimg[h2:h2+h1, w2:w2+w1] = img
        # cv2.imshow('test',newimg)

        maxlen = min(len(skp), len(tkp))
        if num < 0 or num > maxlen:
            num = maxlen
        for i in range(num):
            pt_a = (int(tkp[i].pt[0]), int(tkp[i].pt[1]))
            pt_b = (int(skp[i].pt[0]+w2), int(skp[i].pt[1]+h2))
            cv2.line(newimg, pt_a, pt_b, (255, 0, 0))
        cv2.imshow('test',newimg)
        # return newimg

    def feature_matching(self, img1path, img2path):
        """Feature Matching
        """
        img1 = cv2.imread(img1path) # queryImage
        img2 = cv2.imread(img2path) # trainImage
        # cv2.imshow('img1', img1)
        # cv2.imshow('img2', img2)

        # Initiate SIFT detector
        sift = cv2.SIFT()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(des1,des2,k=2)
        goodmatches = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                goodmatches.append(m)
                # print i
        # print goodmatches
        # print goodmatches[0].distance
        # print matches[233][0].distance

        indices = range(len(goodmatches))
        indices.sort(key=lambda i: goodmatches[i].distance)
        # for i in indices:
        #     print goodmatches[i].distance
        skp = []
        for i in indices:
            skp.append(kp1[goodmatches[i].queryIdx])
        # print skp
        tkp = []
        for i in indices:
            tkp.append(kp2[goodmatches[i].trainIdx])
        # print tkp
        self.drawKeyPoints(img1,img2,skp,tkp,50)
