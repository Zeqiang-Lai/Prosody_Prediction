import argparse
import os

from inference.api import ProsodyNet

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='inference/model')
parser.add_argument('--model_name', default='pph')   # pw, pph, iph
parser.add_argument('--data_dir', default='data/biaobei2/train')
parser.add_argument('--out_dir', default='result')

def load_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split() for line in lines]
        return lines

if __name__ == '__main__':
    args = parser.parse_args()

    net1 = ProsodyNet(args.model_dir, args.model_name)

    pos = load_data(os.path.join(args.data_dir, 'pos.txt'))
    labels = load_data(os.path.join(args.data_dir, 'labels.txt'))
    sents = load_data(os.path.join(args.data_dir, 'sentences.txt'))

    assert len(pos) == len(labels) == len(sents)

    with open(os.path.join(args.out_dir, 'predict.txt'), 'w') as f_p:
        with open(os.path.join(args.out_dir, 'truth.txt'), 'w') as f_t:
            with open(os.path.join(args.out_dir, 'sent.txt'), 'w') as f_s:
                for i in range(len(pos)):
                    tags = net1.inference(sents[i], pos[i])
                    if tags != labels[i]:
                        f_p.write(" ".join(tags) + '\n')
                        f_p.write(" ".join(labels[i]) + '\n')
                        f_p.write(" ".join(sents[i]) + '\n')
                        f_p.write(" ".join(pos[i]) + '\n')
                    if i % 100 == 0:
                        print("Processed {0}/{1}".format(i, len(pos)))