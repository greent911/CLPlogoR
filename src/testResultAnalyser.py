classes_threshold_dict = {
        'adidas': 0,
        'aldi': 0,
        'apple': 0,
        'becks': 0,
        'bmw': 0,
        'carlsberg': 0,
        'chimay': 0,
        'cocacola': 0,
        'corona': 0,
        'dhl': 0,
        'erdinger': 0,
        'esso': 0,
        'fedex': 0,
        'ferrari': 0,
        'ford': 0,
        'fosters': 0,
        'google': 0,
        'guiness': 0,
        'heineken': 0,
        'HP': 0,
        'milka': 0,
        'no-logo': 0,
        'nvidia': 0,
        'paulaner': 0,
        'pepsi': 0,
        'rittersport': 0,
        'shell': 0,
        'singha': 0,
        'starbucks': 0,
        'stellaartois': 0,
        'texaco': 0,
        'tsingtao': 0,
        'ups': 0
        }
class TestResultAnalyser(object):
    """docstring for TestResultAnalyser"""
    def __init__(self, filepath):
        self.fileLines = []
        with open(filepath) as f:
            for line in f:
                self.fileLines.append(line.rstrip('\n'))

    def analysisBest(self):
        f = open('../result', 'w')
        imgFileDict = dict()
        for line in testResultAnalyser.fileLines:
            lineSplit = line.split(' ')
            imgFileDict[lineSplit[0]] = dict()
        for line in testResultAnalyser.fileLines:
            lineSplit = line.split(' ')
            imgFileDict[lineSplit[0]][lineSplit[2]] = int(lineSplit[3])
        for imgpath in imgFileDict:
            tuplesList = [(k,v) for v,k in sorted([(v,k) for k,v in imgFileDict[imgpath].items()],reverse=True)]
            detectClass = 'no-logo'
            for pair in tuplesList:
                if classes_threshold_dict.has_key(pair[0]) and pair[1] > classes_threshold_dict[pair[0]]:
                    detectClass = pair[0]
                    break

            temps = imgpath.split('/')
            imgfilename = temps[len(temps)-1].rstrip('.jpg')
            f = open('../result', 'a')
            f.write(imgfilename + ' ' + detectClass + '\n')
            f.close()
            print imgfilename,detectClass

if __name__=='__main__':
    filepath = '../test_result'

    testResultAnalyser = TestResultAnalyser(filepath)
    testResultAnalyser.analysisBest()
