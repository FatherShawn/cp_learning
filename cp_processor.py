import sys
from cp_flatten import CensoredPlanetFlatten


def main(argv):
    filename = argv[1]
    dataset = CensoredPlanetFlatten(filename, True, True)

    for item in dataset:
        pass

if __name__ == '__main__':
    main(sys.argv)