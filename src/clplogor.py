from os.path import exists, isdir, basename, join, splitext
from glob import glob
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
from cPickle import dump
import argparse
from training_handler import TrainingHandler
from recognizer import Recognizer

EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
# DATASETPATH = '../dataset'
DATASETPATH = '/home/yuan/dataset'

def get_categories(datasetpath):
    cat_paths = [files
                for files in glob(datasetpath + "/*")
                if isdir(files)]
    cat_paths.sort()
    cats = [basename(cat_path) for cat_path in cat_paths]
    return cats


def get_imgfiles(path):
    all_files = []
    all_files.extend([join(path, basename(fname))

    for fname in glob(path + "/*")
        if splitext(fname)[-1].lower() in EXTENSIONS])

    return all_files

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--string", help="echo the string", default="Hello World!!")
    args = parser.parse_args()
    print args.string

if __name__ == '__main__':
    parse_arguments()

    # load dataset categories
    cats = get_categories(DATASETPATH)

    # for cat in cats:
    #     print 'category:%s' % cat
    #     print get_imgfiles(DATASETPATH+'/'+cat)
    trHandler = TrainingHandler()
    trHandler.training_imageSet(get_imgfiles(DATASETPATH+'/adidas'))
    recognizer = Recognizer()
    recognizer.recognize('../../2345541572.jpg',trHandler)
