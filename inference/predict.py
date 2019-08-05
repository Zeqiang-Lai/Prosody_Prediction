import argparse
import thulac

from inference.api import ProsodyNet

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/biaobei2', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='model', help="Directory containing params.json")


def _tokenize():
    tokenzier = thulac.thulac(seg_only=True)

    def _tokenize(text):
        words = tokenzier.cut(text, text=True)
        words = words.strip().split()
        return words

    return _tokenize


def run(net, tokenize):
    while True:
        text = input('>> ')
        words = tokenize(text)
        print(words)
        tags = net.inference(words)
        print(tags)


if __name__ == '__main__':
    args = parser.parse_args()

    net = ProsodyNet(args.model_dir, args.data_dir)

    run(net, _tokenize())
