#USEAGE:
#   getImagePath = GetImagePath('./FlickrLogos-v2')
#   print(getImagePath.getImagePath('apple', 1))
class GetImagePath(object):
    """docstring for GetImagePath"""
    def __init__(self, path):
        super(GetImagePath, self).__init__()
        self.path = path
        self.arrayOfSet = [{}, {}, {}]

    def getImagePath(self, className, setNum):
        setNum = setNum - 1
        if len(self.arrayOfSet[setNum]) < 1:
            pathOfSet = [self.path+'/trainset.txt', self.path+'/valset.txt', self.path+'/testset.txt']
            f = open( pathOfSet[setNum], 'r' )
            while True:
                line = f.readline().strip()
                if len(line) == 0:
                    break
                lineSplit = line.split(',')
                filePath = self.path+'/classes/jpg/'+lineSplit[0]+'/'+lineSplit[1]
                if lineSplit[0] in self.arrayOfSet[setNum]:
                    self.arrayOfSet[setNum][lineSplit[0]].append(filePath)
                else:
                    self.arrayOfSet[setNum][lineSplit[0]] = [filePath]
            f.close()

        if setNum >= 0 and setNum < 3:
            if className in self.arrayOfSet[setNum]:
                return self.arrayOfSet[setNum][className]
            else:
                raise Exception("Class not exist!")
                return []
        else:
            raise Exception("Set not exist!")
            return []

if __name__=='__main__':
    path = '../FlickrLogos-v2'

    gi = GetImagePath(path)
    print gi.getImagePath('no-logo',2)
