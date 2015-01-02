import cv2
import itertools
from training_handler import TrainingHandler
import math_formula
from math import sqrt,degrees,acos
class Recognizer():
    def createAtriangle(self,tripePoint,kp,des):
        
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
        return [des[keyindexi],des[keyindexj],des[keyindexk],delta1,delta2,t_alpha,t_beta,t_gamma]

    def triangleCompare(self,triangle1,triangle2):
        # delta1
        print abs(triangle1[3]-triangle2[3])

    def recognize(self,imgpath,edgeIndexCodeDict,triangleSet):
        img = cv2.imread(imgpath)
        
        # Initiate SIFT detector
        sift = cv2.SIFT()

        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img,None)

        kppairs_num = list(itertools.combinations(range(len(kp)),2))
        # print len(kppairs_num)
        matchEdgePairNum = []
        matchSimpleEdgePairNum = []
        matchPointNum = set()
        for i,j in kppairs_num:
            ix = kp[i].pt[0]
            iy = -kp[i].pt[1]
            jx = kp[j].pt[0]
            jy = -kp[j].pt[1]
            alpha = math_formula.computeRelativeAngle(kp[i].angle,(jx-ix),(jy-iy))
            beta = math_formula.computeRelativeAngle(kp[j].angle,(ix-jx),(iy-jy))
            # edgeFeatureList.append([des[i],des[j],alpha,beta])
            # edgeFeatureList.append([alpha,beta])
            # print alpha,beta
            # print math_formula.dictCode(alpha,beta)
            if edgeIndexCodeDict[math_formula.dictCode(alpha,beta)]==True:
                matchEdgePairNum.append([i,j,alpha,beta])
                matchSimpleEdgePairNum.append([i,j])
                matchPointNum.add(i)
                matchPointNum.add(j)
        
        # matchPointNum = {1,2,3,4,5}
        # matchSimpleEdgePairNum = [[1,2],[1,3],[2,3],[2,4],[1,4],[3,5]]

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
            queryImgTriangles.append(self.createAtriangle(tripePointNum.pop(),kp,des))
        # print queryImgTriangles
        print len(queryImgTriangles)

        for queryTriangle in queryImgTriangles:
            for trainedTriangle in triangleSet:
                self.triangleCompare(queryTriangle,trainedTriangle)

if __name__ == '__main__':
   trHandler = TrainingHandler()
   trHandler.feature_matching('box.png','box_in_scene.png')
   print len(trHandler.triangleSet)
   recognizer = Recognizer()
   recognizer.recognize('box_in_scene.png',trHandler.edgeIndexCodeDict,trHandler.triangleSet)
   # recognizer.recognize('box.png',trHandler.edgeIndexCodeDict,trHandler.triangleSet)

