import cv2
import itertools
from training_handler import TrainingHandler
import math_formula
from math import sqrt,degrees,acos
import numpy as np
# from lshash import LSHash
from scipy.cluster.vq import vq
import time
import random
from feature_storage import FeatureStorage

class Recognizer():
    def __init__(self):
        self.img_traingle_counter = {}

    def createAtriangle(self,tripePoint,kp,keyIds,imgpath):

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
        return [keyIds[keyindexi],keyIds[keyindexj],keyIds[keyindexk],delta1,delta2,t_alpha,t_beta,t_gamma,kp[keyindexi],kp[keyindexj],kp[keyindexk],imgpath]

    def drawTrianglePair(self,triangle1,trTriangle):
        img1 = cv2.imread(triangle1[11])
        img2 = cv2.imread(trTriangle[3])

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
        pt_ip = (int(trTriangle[0][0]), int(trTriangle[0][1]))
        pt_jp = (int(trTriangle[1][0]), int(trTriangle[1][1]))
        pt_kp = (int(trTriangle[2][0]), int(trTriangle[2][1]))

        cv2.line(newimg, pt_i, pt_j, (255, 0, 0))
        cv2.line(newimg, pt_j, pt_k, (255, 0, 0))
        cv2.line(newimg, pt_i, pt_k, (255, 0, 0))
        cv2.line(newimg, pt_ip, pt_jp, (255, 0, 0))
        cv2.line(newimg, pt_jp, pt_kp, (255, 0, 0))
        cv2.line(newimg, pt_ip, pt_kp, (255, 0, 0))

        cv2.imshow('Matches',newimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def recognize(self,imgpath,trHandler):

        img = cv2.imread(imgpath)
        # Initiate SIFT detector
        sift = cv2.SIFT()

        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img,None)

        desArray = np.asarray(des)
        Ids = list(vq(desArray, trHandler.centroidsOfKmean2000[0])[0])
        keyIds = [x*1000 for x in Ids]

        kppairs_num = list(itertools.combinations(range(len(kp)),2))
        # print len(kppairs_num)
        matchSimpleEdgePairNum = []
        matchPointNum = set()

        # Time Start
        tStart = time.time()
        maxcount = 0
        random.shuffle(kppairs_num)
        while kppairs_num and maxcount < 100000:
            i,j = kppairs_num.pop()
        # for i,j in kppairs_num:
            ix = kp[i].pt[0]
            iy = -kp[i].pt[1]
            jx = kp[j].pt[0]
            jy = -kp[j].pt[1]
            vix = jx - ix
            viy = jy - iy
            length = sqrt(abs(vix*vix+viy*viy))
            if length < 3.0 or length > 30.0:
                continue
            maxcount = maxcount + 1
            alpha = math_formula.computeRelativeAngle(kp[i].angle,vix,viy)
            beta = math_formula.computeRelativeAngle(kp[j].angle,-vix,-viy)

            if trHandler.dVisualWordIndexCheck[keyIds[i]/1000,keyIds[j]/1000]:
                temp = trHandler.edgesIndexLSH.query([keyIds[i],keyIds[j],alpha,beta],1)
                if temp:
                    if temp[0][1] < 1000 and keyIds[i]-temp[0][0][0] == 0 and keyIds[j]-temp[0][0][1] == 0 and abs(alpha-temp[0][0][2]) < 25 and abs(beta-temp[0][0][3]) < 25:
                        # print keyIds[i],keyIds[j],temp[0][0][0],temp[0][0][1]
                        matchSimpleEdgePairNum.append([i,j])
                        matchPointNum.add(i)
                        matchPointNum.add(j)

        print 'Edge Match Count:',len(matchSimpleEdgePairNum)
        # Time End
        tEnd = time.time()
        print "cost %f sec" % (tEnd - tStart)
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
            queryImgTriangles.append(self.createAtriangle(tripePointNum.pop(),kp,keyIds,imgpath))

        # print queryImgTriangles
        print imgpath,'Possible Triangles Count:',len(queryImgTriangles)
        self.showTraingle(queryImgTriangles, trHandler, imgpath)

    def showTraingle(self, queryImgTriangles, trHandler, imgpath):

        matchCount = 0
        for i in range(len(queryImgTriangles)):
            queryResult = trHandler.trianglesIndexLSH.query([queryImgTriangles[i][0],queryImgTriangles[i][1],queryImgTriangles[i][2],queryImgTriangles[i][3],queryImgTriangles[i][4],queryImgTriangles[i][5],queryImgTriangles[i][6],queryImgTriangles[i][7]],1)
            if queryResult:
                if queryImgTriangles[i][0] == queryResult[0][0][0][0] and queryImgTriangles[i][1] == queryResult[0][0][0][1] and queryImgTriangles[i][2] == queryResult[0][0][0][2] and queryResult[0][1] < 1352:
                    #self.drawTrianglePair(queryImgTriangles[i],trHandler.trianglePositionList[int(queryResult[0][0][1])])
                    matchCount = matchCount + 1
                    # print queryResult[0][0][1]

        print 'Triangle Feature Match Count:',matchCount
        self.img_traingle_counter[imgpath] = matchCount

    def showImgTriangleCounter(self):
        for key in self.img_traingle_counter:
            print key, self.img_traingle_counter[key]

    def writeImgTriangleCounter(self, logo_name):
        f = open('../triangleCounter/'+logo_name,'w')
        for key in self.img_traingle_counter:
            print key, self.img_traingle_counter[key]
            f.write(key+','+str(self.img_traingle_counter[key])+ '\n') # python will convert \n to os.linesep
        f.close()


if __name__ == '__main__':
   trHandler = TrainingHandler('adidas')
   # trHandler.image_training('box.png','box_in_scene.png')
   #trHandler.training_imageSet(['box.png','box_in_scene.png'])
   # print 'Length of Trained Triangle Set:',len(trHandler.trianglePositionList)
   recognizer = Recognizer()
   # Time Start
   tStart = time.time()
   recognizer.recognize('box_query.png',trHandler)
   recognizer.recognize('./22.png', trHandler)
   recognizer.showImgTriangleCounter()
   # Time End
   tEnd = time.time()
   print "cost %f sec" % (tEnd - tStart)

