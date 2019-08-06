import argparse
import thulac
import os

from inference.api import ProsodyNet

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
print('Working directory: {0}'.format(dname))

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='model', help="Directory containing params.json")


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


def run(nets, tokenize):
    while True:
        text = input('>> ')
        words, pos = tokenize(text)
        print(words)
        for net, name in nets:
            tags = net.inference(words, pos)
            # print(name, tags)
            print(name, concate(words, tags))


def concate(words, tags):
    assert len(words) == len(tags)

    cat = []
    s = ''
    for i in range(len(tags)):
        if tags[i] == 'B':
            cat.append(s)
            s = '' + words[i]
        else:
            s += words[i]
    cat.append(s)
    return cat[1:]


if __name__ == '__main__':
    args = parser.parse_args()

    net1 = ProsodyNet(args.model_dir, 'pw')
    net2 = ProsodyNet(args.model_dir, 'pph')
    net3 = ProsodyNet(args.model_dir, 'iph')

    run([(net1, 'PW'), (net2, 'PPH'), (net3, 'IPH')], _tokenize())
