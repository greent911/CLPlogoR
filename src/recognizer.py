import cv2
import itertools
from training_handler import TrainingHandler
import math_formula 
class Recognizer():
    def queryingTheEdge(self,imgpath,edgeIndexCodeDict):
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

        print tripePointNum
        print len(tripePointNum)
        # print tempDict
        # print len(matchSimpleEdgePairNum)
        # count = 0
        # for i in tempDict.keys():
        #     if tempDict.has_key(i):
        #         count = count + len(tempDict[i])
        # print count
                    

if __name__ == '__main__':
   trHandler = TrainingHandler()
   trHandler.feature_matching('box.png','box_in_scene.png')
   print len(trHandler.triangleSet)
   recognizer = Recognizer()
   recognizer.queryingTheEdge('box_in_scene.png',trHandler.edgeIndexCodeDict)

