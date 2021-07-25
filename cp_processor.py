from cp_flatten import CensoredPlanetFlatten
from webdataset.writer import ShardWriter


def main():
    urls = [

    ]
    dataset = CensoredPlanetFlatten(urls, True, True)
    writer = ShardWriter('preprocessed/quack-%i.tar')

    for item in dataset:
        writer.write({
        "__key__": f"response-{writer.count:06}",
        "pth": item
        })
    writer.finish()
    writer.close()


if __name__ == '__main__':
    main()