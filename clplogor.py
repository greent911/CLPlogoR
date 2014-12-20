import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--string", help="echo the string", default="Hello World!!")
    args = parser.parse_args()
    print args.string
if __name__ == '__main__':
    main()
