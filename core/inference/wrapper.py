import os
import re

from .api import ProsodyNet, _tokenize, concate


class ProsodyPred:
    def __init__(self, model_dir):
        self.model_dir = model_dir

        self.net_pw = ProsodyNet(self.model_dir, 'pw')
        self.net_pph = ProsodyNet(self.model_dir, 'pph')
        self.net_iph = ProsodyNet(self.model_dir, 'iph')

        self.tokenzie = _tokenize()

    def predict(self, text, user_dict=None):
        """
        预测给定句子的三级韵律边界。

        :param text:   给定文本。
        :param user_dict:  用户词典，如果给定句子中已有某些可以确定切分的词,应通过该参数传入。
        :return: (韵律词, 韵律短语, 语调短语), 分词结果
                 例: 今天天气真好 ->
                 (["今天", "天气", "真好", "啊"], ["今天天气", "真好啊"], ["今天天气真好"]), ["今天", "天气", "真好", "啊"]

        Note: 如果文本中包含标点, 根据韵律切分后的句子开头或结尾可能包含标点。
        """

        words, poses = self.tokenzie(text)

        pws = concate(words, self.net_pw.inference(words, poses))
        pphs = concate(words, self.net_pph.inference(words, poses))
        iphs = concate(words, self.net_iph.inference(words, poses))

        return (pws, pphs, iphs), words

    def run(self, sentences):
        texts = re.split('\*|&', sentences)
        texts_size = len(texts)
        result = ''
        sentences_size = len(sentences)
        sentences_index = 0
        for index in range(texts_size):
            if texts[index] == '':
                continue
            if len(texts[index]) < 6:
                result += texts[index]
                sentences_index += len(texts[index])
                if sentences_index < sentences_size:
                    result += sentences[sentences_index]
                    sentences_index += 1
                continue
            words, pos = self.tokenzie(texts[index])
            tags = self.net_pph.inference(words, pos)
            result += self.concate(words, tags)
            sentences_index += len(texts[index])
            if sentences_index < sentences_size:
                result += sentences[sentences_index]
                sentences_index += 1
        return result

    def concate(self, words, tags):
        assert len(words) == len(tags)
        s = ''
        for i in range(len(tags)):
            if tags[i] == 'B' and i != 0:
                s += "%"
            s += words[i]
        return s

if __name__ == '__main__':
    prosody = ProsodyPred('pretrained/biaobei')
    print(prosody.predict('“大国工匠”是中央电视台从去年五一期间推出的系列节目，节目关注了很多在国家重要领域中技艺高超的工匠们，'
                          '今年五一前夕，第三季匠心传世中雕刻师的故事一开播就得到了观众的好评。'))
