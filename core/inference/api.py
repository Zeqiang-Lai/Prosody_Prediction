import os

import numpy as np
import thulac
import torch
import time

from ..model.net import Net
from ..utils import Params, load_checkpoint

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ProsodyPred")
logger.setLevel(level=logging.DEBUG)


class ProsodyNet:
    def __init__(self, model_dir, weight_dir):
        restore_file = 'best'
        json_path = os.path.join(model_dir, weight_dir, 'params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = Params(json_path)
        params.cuda = torch.cuda.is_available()
        params.embedding_path = os.path.join(model_dir, 'embedding200.npy')

        torch.manual_seed(230)
        if params.cuda:
            torch.cuda.manual_seed(230)

        self.model = Net(params).cuda() if params.cuda else Net(params)
        start = time.time()
        load_checkpoint(os.path.join(model_dir, weight_dir, restore_file + '.pth.tar'), self.model)
        logger.debug("Loaded pretrained model in {0:.4f}s".format(time.time() - start))

        total_params = sum(p.numel() for p in self.model.parameters())
        train_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print(self.model)
        logger.debug('Number of Parameters: {0}'.format(total_params))
        logger.debug('Number of Trainable Parameters: {0}'.format(train_total_params))

        self.params = params
        self._load_dict(model_dir)

    def inference(self, words, pos):
        x = self.words2idx(words)
        pos = self.poss2idx(pos)

        input_batch = torch.tensor([x], dtype=torch.long)
        pos_batch = torch.tensor([pos], dtype=torch.long)
        if self.params.cuda:
            input_batch, pos_batch = input_batch.cuda(), pos_batch.cuda()

        output_batch = self.model(input_batch, pos_batch)
        out = output_batch.cpu().detach().numpy()
        out = np.argmax(out, axis=1)
        out = self.idxs2tag(out)
        return out

    def _load_dict(self, model_dir):
        tag_path = os.path.join(model_dir, 'tags.txt')
        words_path = os.path.join(model_dir, 'words.txt')
        pos_path = os.path.join(model_dir, 'pos.txt')

        self.idx2tag = load_tag_idx(tag_path)
        self.word2idx = load_word_idx(words_path)
        self.pos2idx = load_word_idx(pos_path)

    def words2idx(self, words):
        x = []
        for word in words:
            if word in self.word2idx.keys():
                x.append(self.word2idx[word])
            else:
                x.append(self.word2idx['UNK'])
        return np.array(x)

    def poss2idx(self, pos):
        x = []
        for p in pos:
            if p in self.pos2idx.keys():
                x.append(self.pos2idx[p])
            else:
                raise RuntimeError('error converting pos to idx')
        return np.array(x)

    def idxs2tag(self, idxes):
        tags = []
        for idx in idxes:
            if idx in self.idx2tag.keys():
                tags.append(self.idx2tag[idx])
            else:
                raise RuntimeError('unknown idx {0}'.format(idx))
        return tags


def load_word_idx(path_words):
    with open(path_words, 'r', encoding="utf-8") as f:
        word_idx = {}
        for i, word in enumerate(f.readlines()):
            word = word.strip().split()[0]
            word_idx[word] = i
        return word_idx


def load_tag_idx(path_tags):
    with open(path_tags, 'r', encoding="utf-8") as f:
        tag_idx = {}
        for i, tag in enumerate(f.readlines()):
            tag = tag.strip().split()[0]
            tag_idx[i] = tag
        return tag_idx


def _tokenize(user_dict=None):
    tokenzier = thulac.thulac(user_dict=user_dict)

    def _tokenize(text):
        words = []
        pos = []
        pairs = tokenzier.cut(text)
        for pair in pairs:
            words.append(pair[0])
            pos.append(pair[1])
        return words, pos

    return _tokenize


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
