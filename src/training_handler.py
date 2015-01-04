import cv2
import numpy as np
from math import atan, degrees, exp, acos, sqrt
import itertools
from math_formula import dictCode

class TrainingHandler():
    def __init__(self):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        self.flann = cv2.FlannBasedMatcher(index_params,search_params)
        self.DISTCOMPAREFACTOR = 0.7
        self.SIM_THRESHOLD = 0.95
        self.TRIANGLE_CONSTRAINT_DIST = 5.0
        self.TRIANGLE_CONSTRAINT_ANGLE = 15.0
        self.TRIANGLE_CONSTRAINT_ECCENTRICITY_LOWERBOUND = 1.0/3
        self.TRIANGLE_CONSTRAINT_ECCENTRICITY_UPPERBOUND = 3.0
        self.triangleFeaturesSetList = []
        self.edgeIndexCodeDict = {i+1: False for i in xrange(180*180)}
        
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

    def generate_EdgeIndexArray_IndexInEdge(self, keypoints1, keypoints2):
        kpLength = len(keypoints1)

        # keypoints1[i], keypoints1[j]
        # ex:alpha:30 i------j beta:60
        # kpIndexOfEdgeAngleArray:If two points are not matching,it will be 0.
        # [ 0.,  30.,  128.]
        # [ 60.,  0.,  0.]
        # [ 60.,  0.,  0.]
        kpIndexOfEdgeAngleArray = np.zeros((kpLength,kpLength),float)
        # ex:kpIndexOfInEdge={13,23,31,34} i=13,23,31,34 are in edges,keypoints1[i]
        kpIndexOfInEdge = set()
        for i in range(kpLength-1):
            for j in range(i+1,kpLength):
                # Change coordinate to:->x ^y (opencv:->x vy)
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
                simedge = exp(-dalpha*dalpha/128) * exp(-dbeta*dbeta/128)
                if simedge > self.SIM_THRESHOLD:
                    kpIndexOfEdgeAngleArray[i,j] = alpha
                    kpIndexOfEdgeAngleArray[j,i] = beta
                    kpIndexOfInEdge.add(i)
                    kpIndexOfInEdge.add(j)
                    
        return kpIndexOfEdgeAngleArray,list(kpIndexOfInEdge)

    def create_triangles(self,indexOfEdgeAngle,indexInEdge,keypoints,descriptors,imgpath):
        tripleListSetFromIndexInEdge = list(itertools.combinations(indexInEdge,3))
        for keyindexi,keyindexj,keyindexk in tripleListSetFromIndexInEdge:
            ix = keypoints[keyindexi].pt[0]
            iy = -keypoints[keyindexi].pt[1]
            jx = keypoints[keyindexj].pt[0]
            jy = -keypoints[keyindexj].pt[1]
            kx = keypoints[keyindexk].pt[0]
            ky = -keypoints[keyindexk].pt[1]
            vikx = float(kx - ix)
            viky = float(ky - iy)
            vijx = float(jx - ix)
            vijy = float(jy - iy)
            length_vik = sqrt(abs(vikx*vikx+viky*viky))
            length_vij = sqrt(abs(vijx*vijx+vijy*vijy))
            # Add some constraints for generating triangles
            if length_vik < self.TRIANGLE_CONSTRAINT_DIST or length_vij < self.TRIANGLE_CONSTRAINT_DIST or length_vij/length_vik < self.TRIANGLE_CONSTRAINT_ECCENTRICITY_LOWERBOUND or length_vij/length_vik > self.TRIANGLE_CONSTRAINT_ECCENTRICITY_UPPERBOUND:
                continue
            vcos = (vikx*vijx+viky*vijy)/length_vik/length_vij
            delta1 = degrees(acos(round(vcos,13)))
            if delta1 < self.TRIANGLE_CONSTRAINT_ANGLE:
                continue
            vjkx = float(kx - jx)
            vjky = float(ky - jy)
            vjix = -vijx
            vjiy = -vijy
            length_vjk = sqrt(abs(vjkx*vjkx+vjky*vjky))
            length_vji = sqrt(abs(vjix*vjix+vjiy*vjiy))
            if length_vjk < self.TRIANGLE_CONSTRAINT_DIST or length_vji < self.TRIANGLE_CONSTRAINT_DIST or length_vjk/length_vji < self.TRIANGLE_CONSTRAINT_ECCENTRICITY_LOWERBOUND or length_vjk/length_vji > self.TRIANGLE_CONSTRAINT_ECCENTRICITY_UPPERBOUND:
                continue
            vcos = (vjkx*vjix+vjky*vjiy)/length_vjk/length_vji
            delta2 = degrees(acos(round(vcos,13)))
            if delta2 < self.TRIANGLE_CONSTRAINT_ANGLE:
                continue

            edgeij_anglei = indexOfEdgeAngle[keyindexi,keyindexj]
            edgejk_anglej = indexOfEdgeAngle[keyindexj,keyindexk]
            edgeik_anglek = indexOfEdgeAngle[keyindexk,keyindexi]
            if edgeij_anglei == 0.0 or edgejk_anglej == 0.0 or edgeik_anglek == 0.0:
                continue
            edgeij_anglej = indexOfEdgeAngle[keyindexj,keyindexi]
            edgejk_anglek = indexOfEdgeAngle[keyindexk,keyindexi]
            edgeik_anglei = indexOfEdgeAngle[keyindexi,keyindexk]

            self.triangleFeaturesSetList.append([descriptors[keyindexi],descriptors[keyindexj],descriptors[keyindexk],delta1,delta2,edgeij_anglei,edgejk_anglej,edgeik_anglek,keypoints[keyindexi],keypoints[keyindexj],keypoints[keyindexk],imgpath])

            self.edgeIndexCodeDict[dictCode(edgeij_anglei,edgeij_anglej)] = True
            self.edgeIndexCodeDict[dictCode(edgeij_anglej,edgeij_anglei)] = True
            self.edgeIndexCodeDict[dictCode(edgeik_anglei,edgeik_anglek)] = True
            self.edgeIndexCodeDict[dictCode(edgeik_anglek,edgeik_anglei)] = True
            self.edgeIndexCodeDict[dictCode(edgejk_anglej,edgejk_anglek)] = True
            self.edgeIndexCodeDict[dictCode(edgejk_anglek,edgejk_anglej)] = True


    def image_training(self, img1path, img2path):
        img1 = cv2.imread(img1path) # 1:queryImage is going to be trained
        img2 = cv2.imread(img2path) # 2:trainImage trains queryImage

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
        
        indices = range(len(goodmatches))
        
        # indices.sort(key=lambda i: goodmatches[i].distance)

        # The matching SIFT keypoints of queryImage and trainImage are at the same indexes
        # And store the matching SIFT descriptors of queryImage
        goodkeypoints1 = []
        goodkeypoints2 = []
        goodkeydes1 = []
        for i in indices:
            goodkeypoints1.append(kp1[goodmatches[i].queryIdx])
            goodkeypoints2.append(kp2[goodmatches[i].trainIdx])
            goodkeydes1.append(des1[goodmatches[i].queryIdx])

            # print kp2[goodmatches[i].trainIdx].pt
        # test = cv2.drawKeypoints(img2,[goodkeypoints2[47]],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # test = cv2.drawKeypoints(img2,[goodkeypoints2[11]],flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # print goodkeypoints2[47].angle
        # print goodkeypoints2[11].pt
        # cv2.imshow('test',test)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # self.drawKeyPoints(img1,img2,goodkeypoints1,goodkeypoints2)

        edgeIndexArray,indexInEdge = self.generate_EdgeIndexArray_IndexInEdge(goodkeypoints1,goodkeypoints2)
        self.create_triangles(edgeIndexArray,indexInEdge,goodkeypoints1,goodkeydes1,img1path)

if __name__ == '__main__':
   trHandler = TrainingHandler()
   trHandler.image_training('box.png','box_in_scene.png')
   print len(trHandler.triangleFeaturesSetList)
