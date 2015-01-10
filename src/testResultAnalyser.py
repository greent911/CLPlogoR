class TestResultAnalyser(object):
    """docstring for TestResultAnalyser"""
    def __init__(self, filepath):
        with open(filepath) as f:
            for line in f:
                print line

if __name__=='__main__':
    filepath = '../test_result'

    gi = TestResultAnalyser(filepath)
