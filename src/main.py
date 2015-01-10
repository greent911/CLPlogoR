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

    model_list = os.listdir('../model/')
    model_list = [model for model in model_list if 'DS' not in model]
    model_list = [model.replace('.pkl','') for model in model_list]

    for logo_name in logo_classes:
        if logo_name not in model_list:
            continue
        else:
            print logo_name
            recognizer = Recognizer()

            if 'no-logo' in logo_name:
                continue
            else:
                trHandler = TrainingHandler(logo_name)
                imgPaths = getImagePath.getImagePath(logo_name,2)

                for imgPath in imgPaths:
                    print imgPath
                    recognizer.recognize(imgPath, trHandler)

                nologoImgPaths = getImagePath.getImagePath('no-logo',2)

                for imgPath in nologoImgPaths[:30]:
                    print imgPath
                    recognizer.recognize(imgPath, trHandler)

            recognizer.writeImgTriangleCounter(logo_name)

def test(logo_classes):

    f = open('../test_result', 'w')

    model_list = os.listdir('../model/')
    model_list = [model for model in model_list if 'DS' not in model]
    model_list_without_pkl = [model.replace('.pkl','') for model in model_list]

    for model in model_list_without_pkl:

        trHandler = TrainingHandler(model)
        recognizer = Recognizer()

        for logo_name in model_list_without_pkl:
            imgPaths = getImagePath.getImagePath(logo_name, 3)

            for imgPath in imgPaths[:1]:
                recognizer.recognize(imgPath, trHandler)

                f = open('../test_result', 'a')
                print '-------------'
                print 'model: %s' % model
                print recognizer.triangleMatchCount()
                count = recognizer.triangleMatchCount()
                print '-------------'
                f.write(imgPath + ' ' + logo_name + ' ' + model + ' ' + str(count) + '\n')
                f.close()

        imgPaths = getImagePath.getImagePath('no-logo', 3)
        for imgPath in imgPaths[:1]:
            recognizer.recognize(imgPath, trHandler)

            f = open('../test_result', 'a')
            print '-------------'
            print 'model: %s' % model
            print recognizer.triangleMatchCount()
            count = recognizer.triangleMatchCount()
            print '-------------'
            f.write(imgPath + ' ' + logo_name + ' ' + model + ' ' + str(count) + '\n')

#    for logo_name  in logo_classes:
#        print logo_name
#        recognizer = Recognizer()
#
#        if logo_name not in model_list_without_pkl:
#            continue
#
#        if 'no-logo' in logo_name:
#            imgPaths = getImagePath.getImagePath(logo_name,3)
#            imgPaths = imgPaths[:30]
#        else:
#            imgPaths = getImagePath.getImagePath(logo_name,3)
#
#
#        for imgPath in imgPaths[:1]:
#            print imgPath
#            for model in model_list:
#                model_name = model.replace('.pkl', '')
#                trHandler = TrainingHandler(model_name)
#                recognizer = Recognizer()
#                recognizer.recognize(imgPath, trHandler)
#
#                f = open('../test_result', 'a')
#                print '-------------'
#                print 'model: %s' % model
#                print recognizer.triangleMatchCount()
#                count = recognizer.triangleMatchCount()
#                print '-------------'
#                f.write(imgPath + ' ' + logo_name + ' ' + model + ' ' + str(count) + '\n')
#                f.close()

if __name__=='__main__':

    flickr_db_path = '../../FlickrLogos-v2'

    logo_classes = os.listdir(flickr_db_path + '/classes/jpg')
    logo_classes = [logo_name for logo_name in logo_classes if not '.' in logo_name]

    print logo_classes

    getImagePath = GetImagePath(flickr_db_path)

    #train(logo_classes)
    #validate(logo_classes)
    test(logo_classes)
