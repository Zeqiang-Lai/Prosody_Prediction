import argparse
import os

from core.inference.api import ProsodyNet, concate, _tokenize

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
print('Working directory: {0}'.format(dname))

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='pretrained/biaobei', help="Directory containing pretrained models")


def run(nets, tokenize):
    while True:
        text = input('>> ')
        words, pos = tokenize(text)
        print(words, pos)
        for net, name in nets:
            tags = net.inference(words, pos)
            print(name, tags)
            print(name, concate(words, tags))


if __name__ == '__main__':
    args = parser.parse_args()

    net1 = ProsodyNet(args.model_dir, 'pw')
    net2 = ProsodyNet(args.model_dir, 'pph')
    net3 = ProsodyNet(args.model_dir, 'iph')

    run([(net1, 'PW'), (net2, 'PPH'), (net3, 'IPH')], _tokenize())
