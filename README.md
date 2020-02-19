# Prosody Prediction 

韵律预测模型：给定一个句子，输出停顿的位置。

例子: 
```
今天天气真好
PW(韵律词) ['今天', '天气', '真好']
PPH(韵律短语) ['今天', '天气真好']
IPH(语调短语) ['今天天气真好']
```

> Note : all scripts must be run in `prosody_prediction`.

## Requirements

推荐使用虚拟环境。

```shell
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quickstart 

1. 下载预训练模型

```shell
sh pretrained/download_biaobei_pretrained.sh
```

1. 运行交互式命令行进行测试

```shell
python demo.py
```

## API

1. 指定模型进行预测, 参考`demo.py`

```python
net1 = ProsodyNet(args.model_dir, 'pw')
words, pos = tokenize(text)
tags = net.inference(words, pos)
```

2. 使用`wrapper.py`同时预测韵律词，韵律短语，语调短语边界。只能用于预训练标贝数据集。
   - `model_dir`: 预训练模型路径，例如`pretrained/biaobei`

```python
from core.inference.wrapper import ProsodyPred
prosody_pred = ProsodyPred(model_dir)
(pws, pphs, iphs), words = self.prosody_pred.predict(text)
```

## Training

如果你要自己训练，评测模型，构建数据集，请查看 [training-guide.md](training-guide.md)


## FAQ

1. 运行交互式命令行进行测试时遇到:

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 in position 0: invalid continuation byte
```

解决方法(暂时): 这个错误通常发生在输入文字后又删除, 重启保证一次输对。

## Reference

- 语法词、韵律词、韵律短语、语调短语 的区别: [Link](https://blog.csdn.net/chenyi0818/article/details/82662600)

### 论文

[1] Guohong Fu和K. K. Luke, 《Integrated approaches to prosodic word prediction for Chinese TTS》, 收入 *International Conference on Natural Language Processing and Knowledge Engineering, 2003. Proceedings. 2003*, Beijing, China, 2003, 页 413–418, doi: [10.1109/NLPKE.2003.1275941](https://doi.org/10.1109/NLPKE.2003.1275941).

[2] F. E. Office, 《High-Quality Prosody Generation in Mandarin Text-to-Speech System》, *FUJITSU Sci. Tech. J.*, 卷 46, 期 1, 页 7, 2010.

[3] Y. Qian, Z. Wu, X. Ma和F. Soong, 《Automatic prosody prediction and detection with Conditional Random Field (CRF) models》, 收入 *2010 7th International Symposium on Chinese Spoken Language Processing*, Tainan, Taiwan, 2010, 页 135–138, doi: [10.1109/ISCSLP.2010.5684835](https://doi.org/10.1109/ISCSLP.2010.5684835).

[4] C. Ding, L. Xie, J. Yan, W. Zhang和Y. Liu, 《Automatic Prosody Prediction for Chinese Speech Synthesis using BLSTM-RNN and Embedding Features》, *arXiv:1511.00360 [cs]*, 11月 2015.

[5] C. Lai, M. Farrús和J. D. Moore, 《Automatic Paragraph Segmentation with Lexical and Prosodic Features》, 发表于 Interspeech 2016, 2016, 页 1034–1038, doi: [10.21437/Interspeech.2016-992](https://doi.org/10.21437/Interspeech.2016-992).

[6] Y. Zheng, J. Tao, Z. Wen和Y. Li, 《BLSTM-CRF Based End-to-End Prosodic Boundary Prediction with Context Sensitive Embeddings in a Text-to-Speech Front-End》, 收入 *Interspeech 2018*, 2018, 页 47–51, doi: [10.21437/Interspeech.2018-1472](https://doi.org/10.21437/Interspeech.2018-1472).

[7] H. Che, J. Tao和Y. Li, 《Improving Mandarin Prosodic Boundary Prediction with Rich Syntactic Features》, 页 5.

[8] R. Fernandez, A. Rendel, B. Ramabhadran和R. Hoory, 《Using Deep Bidirectional Recurrent Neural Networks for Prosodic-Target Prediction in a Unit-Selection Text-to-Speech System》, 页 6.

[9] S. Kafle, C. O. Alm和M. Huenerfauth, 《Modeling Acoustic-Prosodic Cues for Word Importance Prediction in Spoken Dialogues》, 页 8.