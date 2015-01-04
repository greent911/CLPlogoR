import cv2
import itertools
from training_handler import TrainingHandler
import math_formula
from math import sqrt,degrees,acos
import numpy as np
# from lshash import LSHash
from scipy.cluster.vq import vq

class Recognizer():
    def createAtriangle(self,tripePoint,kp,des,imgpath):
        
        keyindexi,keyindexj,keyindexk = list(tripePoint)
        # print tripePoint
        # print keyindexi,keyindexj,keyindexk
        ix = kp[keyindexi].pt[0]
        iy = -kp[keyindexi].pt[1]
        jx = kp[keyindexj].pt[0]
        jy = -kp[keyindexj].pt[1]
        kx = kp[keyindexk].pt[0]
        ky = -kp[keyindexk].pt[1]
        vikx = float(kx - ix)
        viky = float(ky - iy)
        vijx = float(jx - ix)
        vijy = float(jy - iy)
        length_vik = sqrt(abs(vikx*vikx+viky*viky))
        length_vij = sqrt(abs(vijx*vijx+vijy*vijy))
        if length_vij == 0.0:
            length_vij = 1.0
        if length_vik == 0.0:
            length_vik = 1.0
        vcos = (vikx*vijx+viky*vijy)/length_vik/length_vij
        delta1 = degrees(acos(round(vcos,13)))
        vjkx = float(kx - jx)
        vjky = float(ky - jy)
        vjix = -vijx
        vjiy = -vijy
        length_vjk = sqrt(abs(vjkx*vjkx+vjky*vjky))
        length_vji = sqrt(abs(vjix*vjix+vjiy*vjiy))
        if length_vjk == 0.0:
            length_vjk = 1.0
        if length_vji == 0.0:
            length_vji = 1.0
        vcos = (vjkx*vjix+vjky*vjiy)/length_vjk/length_vji
        delta2 = degrees(acos(round(vcos,13)))
        t_alpha = math_formula.computeRelativeAngle(kp[keyindexi].angle,vijx,vijy)
        t_beta = math_formula.computeRelativeAngle(kp[keyindexj].angle,vjkx,vjky)
        t_gamma = math_formula.computeRelativeAngle(kp[keyindexk].angle,-vikx,-viky)
        return [des[keyindexi],des[keyindexj],des[keyindexk],delta1,delta2,t_alpha,t_beta,t_gamma,kp[keyindexi],kp[keyindexj],kp[keyindexk],imgpath]
    
    def drawTrianglePair(self,triangle1,triangle2):
        img1 = cv2.imread(triangle1[11])
        img2 = cv2.imread(triangle2[11])

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        nWidth = w1+w2
        nHeight = h1+h2
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
        newimg[:h2, :w2] = img2
        newimg[h2:h2+h1, w2:w2+w1] = img1

        pt_i = (int(triangle1[8].pt[0]+w2), int(triangle1[8].pt[1]+h2))
        pt_j = (int(triangle1[9].pt[0]+w2), int(triangle1[9].pt[1]+h2))
        pt_k = (int(triangle1[10].pt[0]+w2), int(triangle1[10].pt[1]+h2))
        pt_ip = (int(triangle2[8].pt[0]), int(triangle2[8].pt[1]))
        pt_jp = (int(triangle2[9].pt[0]), int(triangle2[9].pt[1]))
        pt_kp = (int(triangle2[10].pt[0]), int(triangle2[10].pt[1]))

        cv2.line(newimg, pt_i, pt_j, (255, 0, 0))
        cv2.line(newimg, pt_j, pt_k, (255, 0, 0))
        cv2.line(newimg, pt_i, pt_k, (255, 0, 0))
        cv2.line(newimg, pt_ip, pt_jp, (255, 0, 0))
        cv2.line(newimg, pt_jp, pt_kp, (255, 0, 0))
        cv2.line(newimg, pt_ip, pt_kp, (255, 0, 0))

        cv2.imshow('Matches',newimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def triangleCompare(self,triangle1,triangle2):
        if 180.0 - triangle1[3] - triangle1[4] < 0.0 or 180.0 - triangle1[3] - triangle1[4] > 180.0:
            print "wtf"
        if 180.0 - triangle2[3] - triangle2[4] < 0.0 or 180.0 - triangle2[3] - triangle2[4] > 180.0:
            print "wtf"
        tr1angles = {0: triangle1[3],1: triangle1[4],2: 180.0 - triangle1[3] - triangle1[4]}
        tr1anglesTuples = sorted(tr1angles.items(), key=lambda x: x[1])
        tr2angles = {0: triangle2[3],1: triangle2[4],2: 180.0 - triangle2[3] - triangle2[4]}
        tr2anglesTuples = sorted(tr2angles.items(), key=lambda x: x[1])
        # print tr1anglesTuples,tr2anglesTuples
        if abs(tr1anglesTuples[0][1] - tr2anglesTuples[0][1]) < 2 and abs(tr1anglesTuples[1][1] - tr2anglesTuples[1][1]) < 2 and abs(tr1anglesTuples[2][1] - tr2anglesTuples[2][1]) < 2:
            if abs(triangle1[tr1anglesTuples[0][0]+5]-triangle2[tr2anglesTuples[0][0]+5]) < 2 and abs(triangle1[tr1anglesTuples[1][0]+5]-triangle2[tr2anglesTuples[1][0]+5]) < 2 and abs(triangle1[tr1anglesTuples[2][0]+5]-triangle2[tr2anglesTuples[2][0]+5]) < 2:
                matches = math_formula.flann.knnMatch(np.asarray([triangle1[0],triangle1[1],triangle1[2]]),np.asarray([triangle2[0],triangle2[1],triangle2[2]]),k=1)
                # print matches[0][0].queryIdx,matches[0][0].trainIdx
                # print matches[1][0].queryIdx,matches[1][0].trainIdx
                # print matches[2][0].queryIdx,matches[2][0].trainIdx
                tempset ={0,1,2}
                if not tempset.difference({matches[0][0].trainIdx,matches[1][0].trainIdx,matches[2][0].trainIdx}):
                    self.drawTrianglePair(triangle1,triangle2)
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def recognize(self,imgpath,trHandler):
        img = cv2.imread(imgpath)

        # Initiate SIFT detector
        sift = cv2.SIFT()

        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img,None)

        desArray = np.asarray(des)
        keyIds = list(vq(desArray, trHandler.centroidsOfKmean2000[0])[0])

        kppairs_num = list(itertools.combinations(range(len(kp)),2))
        # print len(kppairs_num)
        # matchEdgePairNum = []
        matchSimpleEdgePairNum = []
        matchPointNum = set()
        lsh = trHandler.edgeIndexLSH()
        for i,j in kppairs_num:
            ix = kp[i].pt[0]
            iy = -kp[i].pt[1]
            jx = kp[j].pt[0]
            jy = -kp[j].pt[1]
            alpha = math_formula.computeRelativeAngle(kp[i].angle,(jx-ix),(jy-iy))
            beta = math_formula.computeRelativeAngle(kp[j].angle,(ix-jx),(iy-jy))
            

            tempIndexNum = math_formula.dictCode(alpha,beta) 
            if trHandler.edgeIndexCodeDict[tempIndexNum]==True or (tempIndexNum < 180*180 and trHandler.edgeIndexCodeDict[tempIndexNum+1]==True ):
                # matchEdgePairNum.append([i,j,alpha,beta])
                temp = lsh.query([keyIds[i],keyIds[j],alpha,beta])
                if temp:
                    for k in range(len(temp)):
                        if keyIds[i]-temp[k][0][0] == 0 and keyIds[j]-temp[k][0][1] == 0:
                            # print keyIds[i],keyIds[j],temp[k][0][0],temp[k][0][1]
                            matchSimpleEdgePairNum.append([i,j])
                            matchPointNum.add(i)
                            matchPointNum.add(j)

        print 'Edge Match Count:',len(matchSimpleEdgePairNum)
        # matchPointNum = {1,2,3}
        # matchSimpleEdgePairNum = [[1,2],[1,3],[2,3]]

        tempDict = {i: set() for i in matchPointNum}
        for i,j in matchSimpleEdgePairNum:
            tempDict[i].add(j)
            tempDict[j].add(i)
        
        tripePointNum = []
        for i,j in matchSimpleEdgePairNum:
            tempset = tempDict[i].intersection(tempDict[j])
            while tempset:
                temps = {i,j,tempset.pop()}
                if temps not in tripePointNum:
                    tripePointNum.append(temps)

        # print tripePointNum
        # print len(tripePointNum)

        queryImgTriangles = []
        while tripePointNum:
            queryImgTriangles.append(self.createAtriangle(tripePointNum.pop(),kp,des,imgpath))
        # print queryImgTriangles
        print imgpath,'Possible Triangles Count:',len(queryImgTriangles)
        
        matchCount = 0
        for i in range(len(queryImgTriangles)):
            for j in range(len(trHandler.triangleFeaturesSetList)):
                if self.triangleCompare(queryImgTriangles[i],trHandler.triangleFeaturesSetList[j]):
                    matchCount = matchCount + 1
                    # print i,j
        print 'Triangle Feature Match Count:',matchCount

if __name__ == '__main__':
   trHandler = TrainingHandler()
   trHandler.image_training('box.png','box_in_scene.png')
   print 'Length of Trained Triangle Set:',len(trHandler.triangleFeaturesSetList)
   recognizer = Recognizer()
   recognizer.recognize('box_in_scene.png',trHandler)
   # recognizer.recognize('box.png',trHandler.edgeIndexCodeDict,trHandler.triangleFeaturesSetList)

