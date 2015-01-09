import os
from getImagePath import GetImagePath
from training_handler import TrainingHandler
from recognizer import Recognizer

def train(logo_classes):

    # get training dataset and train model for each class
    for logo_name in logo_classes:
        if 'no-logo' in logo_name:
            continue
        else:
            print 'Logo:', logo_name
            trainingPaths = getImagePath.getImagePath(logo_name,1)
            print trainingPaths

            trHandler = TrainingHandler(logo_name)
            trHandler.training_imageSet(trainingPaths)

def validate(logo_classes):


    # in order to get the threshold of each class
    # we need to check number of match triangle in same logo class images
    # and no-logo images.

    for logo_name in logo_classes[:1]:
        print logo_name
        recognizer = Recognizer()

        if 'no-logo' in logo_name:
            continue
        else:
            trHandler = TrainingHandler(logo_name)
            imgPaths = getImagePath.getImagePath(logo_name,2)

            for imgPath in imgPaths[:1]:
                print imgPath
                recognizer.recognize(imgPath, trHandler)

            nologoImgPaths = getImagePath.getImagePath('no-logo',2)

            for imgPath in nologoImgPaths[:1]:
                print imgPath
                recognizer.recognize(imgPath, trHandler)

            recognizer.writeImgTriangleCounter(logo_name)

if __name__=='__main__':

    flickr_db_path = '../../FlickrLogos-v2'

    logo_classes = os.listdir(flickr_db_path + '/classes/jpg')
    logo_classes = [logo_name for logo_name in logo_classes if not '.' in logo_name]

    print logo_classes

    getImagePath = GetImagePath(flickr_db_path)

    #train(logo_classes)
    validate(logo_classes)

