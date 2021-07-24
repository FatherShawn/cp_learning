import sys
from chardet.universaldetector import UniversalDetector

def main(argv):
    filename = argv[1]
    detector = UniversalDetector()

    with open(filename, 'rb') as file:
        for line in file:
            detector.feed(line)
    detector.close()
    print(detector.result)


if __name__ == '__main__':
    main(sys.argv)