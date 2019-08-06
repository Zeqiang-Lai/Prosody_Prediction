import argparse
import thulac

from inference.api import ProsodyNet

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/biaobei2', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='model2', help="Directory containing params.json")


def _tokenize():
    tokenzier = thulac.thulac()

    def _tokenize(text):
        words = []
        pos = []
        pairs = tokenzier.cut(text)
        for pair in pairs:
            words.append(pair[0])
            pos.append(pair[1])
        return words, pos

    return _tokenize


def run(net, tokenize):
    while True:
        text = input('>> ')
        words, pos = tokenize(text)
        print(words)
        tags = net.inference(words, pos)
        print(tags)


if __name__ == '__main__':
    args = parser.parse_args()

    net = ProsodyNet(args.model_dir, args.data_dir)

    run(net, _tokenize())
