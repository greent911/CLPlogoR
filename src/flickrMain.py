import os
from getImagePath import GetImagePath
from training_handler import TrainingHandler
from recognizer import Recognizer
import time

def elapsed_time():
    global last_time
    diff = time.time()-last_time
    last_time = time.time()
    return diff

last_time = time.time()

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

def test():

    f = open('../test_starbucksWeb_result', 'w')

    model_list = os.listdir('../model/')
    model_list = [model for model in model_list if 'DS' not in model]
    model_list_without_pkl = [model.replace('.pkl','') for model in model_list]

    for model in model_list_without_pkl:

        if model != "starbucks":
            continue

        print model

        trHandler = TrainingHandler(model)
        recognizer = Recognizer()

        image_list = os.listdir('../starbucksWeb/')
        image_list = ['../starbucksWeb/' + image_name for image_name in image_list if 'DS' not in image_name]
        print image_list

        for imgPath in image_list:
            print 'time:', elapsed_time()
            recognizer.recognize(imgPath, trHandler)

            print '-------------'
            print 'model: %s' % model
            print recognizer.triangleMatchCount()
            count = recognizer.triangleMatchCount()
            print '-------------'
            if count >= 1:
                f = open('../test_starbucksWeb_result', 'a')
                f.write(imgPath + '\n')
                f.close()

if __name__=='__main__':
    test()
