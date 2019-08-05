import argparse
import os

import numpy as np

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--emb_dir', default='embedding', help="Directory containing the word embedding")
parser.add_argument('--out_dir', default='embedding', help="Directory containing the processed word embedding")
parser.add_argument('--words_dir', default='data/biaobei2')
parser.add_argument('--dim', default=200)
parser.add_argument('--max_num_words', default=30000)

def load_embedding_matrix(path_emb, word_index, emb_dim, max_num_words):
    num_words = min(max_num_words, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, emb_dim))

    with open(path_emb, 'r', encoding="utf-8") as f:
        count = 0
        cc = 0
        for line in f:
            values = line.split()
            word = values[0]
            if word in word_index.keys():
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embedding_matrix[word_index[word]] = coefs
                    count += 1
                except Exception as e:
                    print(line)
                    input('press to continue')
            if count == len(word_index.keys()):
                break

            cc += 1
            if cc % 50000 == 0:
                print('Parse {0} line, find {1}/{2}'.format(cc, count, len(word_idx)))

    return embedding_matrix


def load_word_idx(path_words):
    with open(path_words, 'r') as f:
        word_idx = {}
        for i, word in enumerate(f.readlines()):
            word = word.strip().split()[0]
            word_idx[word] = i
        return word_idx

if __name__ == '__main__':
    args = parser.parse_args()

    word_idx = load_word_idx(os.path.join(args.words_dir, 'words.txt'))
    path_emb_tecent = os.path.join(args.emb_dir, 'Tencent_AILab_ChineseEmbedding.txt')
    emb = load_embedding_matrix(path_emb_tecent, word_idx, args.dim, args.max_num_words)

    path_out_emb = args.emb_dir
    if not os.path.exists(path_out_emb):
        os.makedirs(path_out_emb)
    np.save(os.path.join(path_out_emb, 'embedding200.npy'), emb)

    # Save embeddings properties in json file
    sizes = {
        'embedding_dim': args.dim,
    }
    utils.save_dict_to_json(sizes, os.path.join(args.out_dir, 'embedding_params.json'))

    # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the embedding:\n{}".format(to_print))