import sys
from cp_flatten import CensoredPlanetFlatten
from webdataset.writer import ShardWriter


def main(argv):
    filename = argv[1]
    dataset = CensoredPlanetFlatten(filename, True, True)
    writer = ShardWriter('preprocessed/quack-%i.tar')

    for item in dataset:
        writer.write({
        "__key__": f"response-{writer.count:06}",
        "pth": item
        })
    writer.finish()
    writer.close()


if __name__ == '__main__':
    main(sys.argv)