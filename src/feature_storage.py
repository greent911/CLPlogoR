import cPickle

class FeatureStorage:
    def __init__(self, logo_tile, triangleFeaturesSetList=None,\
                                  edgeIndexCodeDict=None,\
                                  trainedDescriptorsList=None,\
                                  centroidsOfKmean2000=None,\
                                  visualWordLabelIDs=None,\
                                  edgesIndexLSH=None,\
                                  trianglesIndexLSH=None,\
                                  triangleVWwith6anglesFeatureList=None):
        self.logo_tile = logo_tile
        self.triangleFeaturesSetList = triangleVWwith6anglesFeatureList
        self.edgeIndexCodeDict = edgesIndexLSH
        self.trainedDescriptorsList = trainedDescriptorsList
        self.centroidsOfKmean2000 = centroidsOfKmean2000
        self.visualWordLabelIDs = visualWordLabelIDs
        self.edgesIndexLSH = edgesIndexLSH
        self.trianglesIndexLSH = trianglesIndexLSH
        self.triangleVWwith6anglesFeatureList = triangleVWwith6anglesFeatureList

    def save(self):
        data = (self.triangleFeaturesSetList,
                self.edgeIndexCodeDict,
                self.trainedDescriptorsList,
                self.centroidsOfKmean2000,
                self.visualWordLabelIDs,
                self.edgesIndexLSH,
                self.trianglesIndexLSH,
                self.triangleVWwith6anglesFeatureList)
        fh = open(self.logo_tile+'.pkl', 'wb')
        cPickle.dump(data,fh)

        if fh is not None:
            fh.close()

    def load(self, logo_tile = None):
        if logo_tile == None:
            logo_tile = self.logo_tile
        fh = None

        fh = open(logo_tile+'.pkl', 'rb')
        data = cPickle.load(fh)

        (self.triangleFeaturesSetList,
         self.edgeIndexCodeDict,
         self.trainedDescriptorsList,
         self.centroidsOfKmean2000,
         self.visualWordLabelIDs,
         self.edgesIndexLSH,
         self.trianglesIndexLSH,
         self.triangleVWwith6anglesFeatureLis) = data
        fh.close()



if __name__=='__main__':
    FS = FeatureStorage('adidas')
    FS.save()
    FS.load()

