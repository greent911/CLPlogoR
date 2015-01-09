import os
from getImagePath import GetImagePath
from training_handler import TrainingHandler

if __name__=='__main__':

    flickr_db_path = '../../FlickrLogos-v2'

    logo_classes = os.listdir(flickr_db_path + '/classes/jpg')
    logo_classes = [logo_name for logo_name in logo_classes if not '.' in logo_name]

    print logo_classes

    getImagePath = GetImagePath(flickr_db_path)

    # get training dataset and train model for each class
    for logo_name in logo_classes:
        print 'Logo:', logo_name
        trainingPaths = getImagePath.getImagePath(logo_name,1)
        print trainingPaths

        trHandler = TrainingHandler(logo_name)
        trHandler.training_imageSet(trainingPaths)

