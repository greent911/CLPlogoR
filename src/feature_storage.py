import cPickle
from training_handler import *

class FeatureStorage:
    def __init__(self, logo_tile):
        self.destDirPath = '../model'
        self.logo_tile = logo_tile

    def save(self, trHandler):

        logo_tile = self.logo_tile

        # store whole trHandler into ./model/logo_name.pkl
        fh = open( self.destDirPath + '/' + logo_tile+'.pkl', 'wb')


        data = (
            #trHandler.triangleFeaturesSetList,
                trHandler.edgeIndexCodeDict,
                trHandler.edgeIndexCodeDict,
                trHandler.trainedDescriptorsList ,
                trHandler.centroidsOfKmean2000,
                trHandler.visualWordLabelIDs,
                trHandler.edgesIndexLSH,
                trHandler.trianglesIndexLSH,
                trHandler.triangleVWwith6anglesFeatureList,
                trHandler.dVisualWordIndexCheck )

        cPickle.dump(data,fh)

        if fh is not None:
            fh.close()

    def load(self, logo_tile = None):
        if logo_tile == None:
            # raise error
            logo_tile = self.logo_tile

        fh = open(self.destDirPath + '/' + logo_tile+'.pkl', 'rb')
        data = cPickle.load(fh)
        fh.close()

        trHandler = TrainingHandler()

        #(trHandler.triangleFeaturesSetList,
        (
            trHandler.edgeIndexCodeDict,
            trHandler.edgeIndexCodeDict,
            trHandler.trainedDescriptorsList ,
            trHandler.centroidsOfKmean2000,
            trHandler.visualWordLabelIDs,
            trHandler.edgesIndexLSH,
            trHandler.trianglesIndexLSH,
            trHandler.triangleVWwith6anglesFeatureList,
            trHandler.dVisualWordIndexCheck ) = data

        return trHandler

if __name__=='__main__':

    trHandler = TrainingHandler()
    trHandler.training_imageSet(['box.png', 'box_in_scene.png'])

    FS = FeatureStorage('adidas')
    FS.save(trHandler)

    new_trHandler = FS.load('adidas')
    print new_trHandler

