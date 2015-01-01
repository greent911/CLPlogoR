import cv2
import numpy as np
from math import atan, degrees, exp

class TrainingHandler():
    def __init__(self):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        self.flann = cv2.FlannBasedMatcher(index_params,search_params)
        self.DISTCOMPAREFACTOR = 0.7
        
    def drawKeyPoints(self, img1, img2, keypoints1, keypoints2, num=-1):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        nWidth = w1+w2
        nHeight = h1+h2
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
        newimg[:h2, :w2] = img2
        newimg[h2:h2+h1, w2:w2+w1] = img1

        maxlen = min(len(keypoints1), len(keypoints2))
        if num < 0 or num > maxlen:
            num = maxlen
        for i in range(num):
            pt_a = (int(keypoints2[i].pt[0]), int(keypoints2[i].pt[1]))
            pt_b = (int(keypoints1[i].pt[0]+w2), int(keypoints1[i].pt[1]+h2))
            cv2.line(newimg, pt_a, pt_b, (255, 0, 0))
        cv2.imshow('Matches',newimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def compute_relative_angle(self, siftangle, vx, vy):
        angle = 0.0
        rangle = 0.0
        if vx > 0 and vy >= 0:
            angle = degrees(atan(vy/vx))        
        elif vx == 0 and vy == 0:
            angle = 0.0
        elif vx == 0 and vy > 0:
            angle = 90.0
        elif vx < 0 and vy > 0:
            angle = 90.0 + degrees(atan(-vx/vy))        
        elif vx < 0 and vy == 0:
            angle = 180.0       
        elif vx < 0 and vy < 0:
            angle = 180.0 + degrees(atan(vy/vx))        
        elif vx == 0 and vy < 0:
            angle = 270.0
        elif vx > 0 and vy < 0:
            angle = 270.0 + degrees(atan(vx/-vy)) 
        else:
            print 'bugs here'
                
        if (360.0 - siftangle) > angle:
            rangle = (360.0 - siftangle) - angle        
        else:
            rangle = 360 - (angle - (360.0 - siftangle))
        return rangle

    def similarity_keypoints(self, keypoints1, keypoints2):
        p1length = len(keypoints1)

        simedge = np.zeros((p1length,p1length),float)
        for i in range(p1length-1):
            for j in range(i+1,p1length):
                ix = keypoints1[i].pt[0]
                iy = -keypoints1[i].pt[1]
                jx = keypoints1[j].pt[0]
                jy = -keypoints1[j].pt[1]
                vix = float(jx - ix)
                viy = float(jy - iy)
                # vjx = -vix
                # vjy = -viy
                alpha = self.compute_relative_angle(keypoints1[i].angle,vix,viy)
                beta = self.compute_relative_angle(keypoints1[j].angle,-vix,-viy)
                ixp = keypoints2[i].pt[0]
                iyp = -keypoints2[i].pt[1]
                jxp = keypoints2[j].pt[0]
                jyp = -keypoints2[j].pt[1]
                vixp = float(jxp - ixp)
                viyp = float(jyp - iyp)
                alphap = self.compute_relative_angle(keypoints2[i].angle,vixp,viyp)
                betap = self.compute_relative_angle(keypoints2[j].angle,-vixp,-viyp)
       
                dalpha = abs(alpha - alphap)
                dbeta = abs(beta - betap)
                simedge[i,j] = exp(-dalpha*dalpha/128) * exp(-dbeta*dbeta/128)
                print simedge[i, j]
                

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

        goodkeypoints1 = []
        for i in indices:
            goodkeypoints1.append(kp1[goodmatches[i].queryIdx])
        goodkeypoints2 = []
        for i in indices:
            goodkeypoints2.append(kp2[goodmatches[i].trainIdx])
            # print kp2[goodmatches[i].trainIdx].angle
            # print kp2[goodmatches[i].trainIdx].pt
        # test = cv2.drawKeypoints(img2,[goodkeypoints2[47]],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # test = cv2.drawKeypoints(img2,[goodkeypoints2[11]],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # print goodkeypoints2[47].angle
        # print goodkeypoints2[11].pt
        # cv2.imshow('test',test)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # self.drawKeyPoints(img1,img2,goodkeypoints1,goodkeypoints2)
        self.similarity_keypoints(goodkeypoints1,goodkeypoints2)

if __name__ == '__main__':
   trHandler = TrainingHandler()
   trHandler.feature_matching('box.png','box_in_scene.png')
