import cPickle

class FeatureStorage:
    def __init__(self, logo_tile):
        self.destDirPath = '../model'
        self.logo_tile = logo_tile

    def save(self, data):

        logo_tile = self.logo_tile

        # store whole trHandler into ./model/logo_name.pkl
        fh = open( self.destDirPath + '/' + logo_tile+'.pkl', 'wb')


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

        return data

if __name__=='__main__':

    trHandler = TrainingHandler('adidas')
    trHandler.training_imageSet(['box.png', 'box_in_scene.png'])

    FS = FeatureStorage('adidas')
    FS.save(trHandler)

    new_trHandler = FS.load()
    print new_trHandler

