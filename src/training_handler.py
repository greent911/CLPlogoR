import cv2
import numpy as np
from math import atan, degrees, exp, acos, sqrt
from math_formula import dictCode
from scipy.cluster.vq import vq, kmeans 
from lshash import LSHash
import time

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

        self.trainedDescriptorsList = []
        self.centroidsOfKmean2000 = tuple()
        self.visualWordLabelIDs = []
        self.edgesIndexLSH = LSHash(32, 4)
        self.trianglesIndexLSH = LSHash(32, 8)
        self.triangleVWwith6anglesFeatureList = []
        
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
        indexOfEdgePairs = []
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
                    indexOfEdgePairs.append([i,j])
                    
        return kpIndexOfEdgeAngleArray,list(kpIndexOfInEdge),indexOfEdgePairs

    def create_triangles(self,indexOfEdgeAngle,indexInEdge,indexOfEdgePairs,keypoints,descriptors,imgpath):
        kpIndexOfInTriangle = set()
        key3indexandDegreesofTriangle = list()

        indexHavePairDict = {i: set() for i in indexInEdge}
        for i,j in indexOfEdgePairs:
            indexHavePairDict[i].add(j)
            indexHavePairDict[j].add(i)
        tripleListSet = []
        for i,j in indexOfEdgePairs:
            tempset = indexHavePairDict[i].intersection(indexHavePairDict[j])
            while tempset:
                temps = {i,j,tempset.pop()}
                if temps not in tripleListSet:
                    tripleListSet.append(temps)
        for keyindexi,keyindexj,keyindexk in tripleListSet:
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
            
            kpIndexOfInTriangle.add(keyindexi)
            kpIndexOfInTriangle.add(keyindexj)
            kpIndexOfInTriangle.add(keyindexk)

            self.triangleFeaturesSetList.append([descriptors[keyindexi],descriptors[keyindexj],descriptors[keyindexk],delta1,delta2,edgeij_anglei,edgejk_anglej,edgeik_anglek,keypoints[keyindexi],keypoints[keyindexj],keypoints[keyindexk],imgpath])

            key3indexandDegreesofTriangle.append([keyindexi,keyindexj,keyindexk,delta1,delta2])
            
            self.edgeIndexCodeDict[dictCode(edgeij_anglei,edgeij_anglej)] = True
            self.edgeIndexCodeDict[dictCode(edgeij_anglej,edgeij_anglei)] = True
            self.edgeIndexCodeDict[dictCode(edgeik_anglei,edgeik_anglek)] = True
            self.edgeIndexCodeDict[dictCode(edgeik_anglek,edgeik_anglei)] = True
            self.edgeIndexCodeDict[dictCode(edgejk_anglej,edgejk_anglek)] = True
            self.edgeIndexCodeDict[dictCode(edgejk_anglek,edgejk_anglej)] = True

        return kpIndexOfInTriangle,key3indexandDegreesofTriangle 

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

        edgeIndexArray,indexInEdge,indexOfEdgePairs = self.generate_EdgeIndexArray_IndexInEdge(goodkeypoints1,goodkeypoints2)
        kpIndexOfInTriangle,key3indexandDegreesofTriangle = self.create_triangles(edgeIndexArray,indexInEdge,indexOfEdgePairs,goodkeypoints1,goodkeydes1,img1path)
        indexOfIDstartPosition = len(self.trainedDescriptorsList)
        for keyindex in kpIndexOfInTriangle:
            self.trainedDescriptorsList.append(goodkeydes1[keyindex])
        keyInTriangleLabelIDDict = dict(zip(list(kpIndexOfInTriangle), range(indexOfIDstartPosition,indexOfIDstartPosition+len(kpIndexOfInTriangle))))
        
        for x in range(len(key3indexandDegreesofTriangle)):
            i = key3indexandDegreesofTriangle[x][0]
            j = key3indexandDegreesofTriangle[x][1]
            k = key3indexandDegreesofTriangle[x][2]
            delta1 = key3indexandDegreesofTriangle[x][3]
            delta2 = key3indexandDegreesofTriangle[x][4]
            alpha = edgeIndexArray[i,j]
            beta = edgeIndexArray[j,k]
            gamma = edgeIndexArray[k,i]
            edgeij_anglej = edgeIndexArray[j,i]
            edgejk_anglek = edgeIndexArray[k,j]
            edgeik_anglei = edgeIndexArray[i,k]
            self.triangleVWwith6anglesFeatureList.append([keyInTriangleLabelIDDict[i],keyInTriangleLabelIDDict[j],keyInTriangleLabelIDDict[k],delta1,delta2,alpha,beta,gamma,edgeij_anglej,edgejk_anglek,edgeik_anglei])

    def generate_EdgeandTriangle_LSH(self):
        x=0
        for i,j,k,delta1,delta2,edgeij_anglei,edgejk_anglej,edgeik_anglek,edgeij_anglej,edgejk_anglek,edgeik_anglei in self.triangleVWwith6anglesFeatureList:
            vi = self.visualWordLabelIDs[i]*1000
            vj = self.visualWordLabelIDs[j]*1000
            vk = self.visualWordLabelIDs[k]*1000
            delta3 = 180.0-delta1-delta2
            self.trianglesIndexLSH.index([vi,vj,vk,delta1,delta2,edgeij_anglei,edgejk_anglej,edgeik_anglek],extra_data=x)
            self.trianglesIndexLSH.index([vi,vk,vj,delta1,delta3,edgeik_anglei,edgejk_anglek,edgeij_anglej],extra_data=x)
            self.trianglesIndexLSH.index([vj,vi,vk,delta2,delta1,edgeij_anglej,edgeik_anglei,edgejk_anglek],extra_data=x)
            self.trianglesIndexLSH.index([vj,vk,vi,delta2,delta3,edgejk_anglej,edgeik_anglek,edgeij_anglei],extra_data=x)
            self.trianglesIndexLSH.index([vk,vi,vj,delta3,delta1,edgeik_anglek,edgeij_anglei,edgejk_anglej],extra_data=x)
            self.trianglesIndexLSH.index([vk,vj,vi,delta3,delta2,edgejk_anglek,edgeij_anglej,edgeik_anglei],extra_data=x)
            self.edgesIndexLSH.index([vi,vj,edgeij_anglei,edgeij_anglej])
            self.edgesIndexLSH.index([vj,vi,edgeij_anglej,edgeij_anglei])
            self.edgesIndexLSH.index([vj,vk,edgejk_anglej,edgejk_anglek])
            self.edgesIndexLSH.index([vk,vj,edgejk_anglek,edgejk_anglej])
            self.edgesIndexLSH.index([vi,vk,edgeik_anglei,edgeik_anglek])
            self.edgesIndexLSH.index([vk,vi,edgeik_anglek,edgeik_anglei])

            x=x+1

    def edgeIndexLSH(self):
        lsh = LSHash(32, 4)
        # for i in range(len(self.the2IndexesOfIDandEdge2AnglesList)):
        #     a=self.visualWordLabelIDs[self.the2IndexesOfIDandEdge2AnglesList[i][0]]
        #     b=self.visualWordLabelIDs[self.the2IndexesOfIDandEdge2AnglesList[i][1]]
        #     c=self.the2IndexesOfIDandEdge2AnglesList[i][2]
        #     d=self.the2IndexesOfIDandEdge2AnglesList[i][3]
        #     lsh.index([a,b,c,d])
        #     lsh.index([b,a,d,c])
        for i,j,k,delta1,delta2,edgeij_anglei,edgejk_anglej,edgeik_anglek,edgeij_anglej,edgejk_anglek,edgeik_anglei in self.triangleVWwith6anglesFeatureList:
            vi = self.visualWordLabelIDs[i]
            vj = self.visualWordLabelIDs[j]
            vk = self.visualWordLabelIDs[k]
            lsh.index([vi,vj,edgeij_anglei,edgeij_anglej])
            lsh.index([vj,vi,edgeij_anglej,edgeij_anglei])
            lsh.index([vj,vk,edgejk_anglej,edgejk_anglek])
            lsh.index([vk,vj,edgejk_anglek,edgejk_anglej])
            lsh.index([vi,vk,edgeik_anglei,edgeik_anglek])
            lsh.index([vk,vi,edgeik_anglek,edgeik_anglei])
        return lsh

    def training_imageSet(self,setOfimgPaths):
        imgCount = len(setOfimgPaths)
        # for test, should not use it
        for i in range(imgCount-1):
            for j in range(i+1,imgCount):
        # for i in range(imgCount):
        #     for j in range(imgCount):
                if i != j:
                    self.image_training(setOfimgPaths[i],setOfimgPaths[j])
        desArray = np.asarray(self.trainedDescriptorsList)
        self.centroidsOfKmean2000 = kmeans(desArray, 2000)
        self.visualWordLabelIDs  = list(vq(desArray, self.centroidsOfKmean2000[0])[0])
        self.generate_EdgeandTriangle_LSH()

if __name__ == '__main__':
   trHandler = TrainingHandler()
   tStart = time.time()
   trHandler.training_imageSet(['box.png','box_in_scene.png'])
   tEnd = time.time()
   print "cost %f sec" % (tEnd - tStart)
   # trHandler.edgeIndexLSH()
   print len(trHandler.triangleFeaturesSetList)
