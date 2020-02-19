## Training Guide

训练代码基于cs230的[Pytorch模版](https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/nlp)。如果看完这个教程对代码还有不理解的地方，可以参考原模版的Guideline。

你可以先看看代码结构。

## Dataset

预训练模型采用标贝的中文标准女声音库，该数据集共有一万个句子，每个句子都有韵律词，韵律短语，语调短语三级切分。

原始的标贝数据集是`/data/biaobei/000001-010000.txt`这个文件。处理这个数据集主要用到了这些脚本:

```
data/biaobei/preprocess.py    # 初步转换,便于后续处理
build_biaobei_dataset.py      # 转换成训练代码需要的格式
build_embedding.py            # 根据数据集提取embedding
build_vocab.py                # 构建单词,标签与数字的映射
```

处理后的`biaobei1`是韵律词，`biaobei2`是韵律短语，`baiobei3`是语调短语。

### Custom Dataset

**格式转换**： 如果你想用自己的数据集，你必须将你的数据集转换成以下格式:

```
/你的数据集的名字
  /train
  	label.txt
  	pos.txt
  	sentences.txt
  /test
  /val
```

- 包含三个文件夹，名字分别为`train`, `val`, `test` 分别用于训练，验证和测试。
- 每个文件夹包含三个文件，每一行存储一个句子的相关信息, 句子中的词用空格分隔。
  - Labels: 存储标签
  - Pos: 存储词性
  - sentences: 存储句子

**构建字典**： 完成格式转换之后，执行以下脚本:

```shell
python build_vocab.py --data_dir 你的数据集的位置
# 其他参数看代码
```

这个脚本会扫描所有句子，包括训练，验证，测试集，然后提取出不重复的标签，词，词性标准构成三个字典，分别存储在`tags.txt`, `words.txt`, `pos.txt` 中。除此之外，脚本还会存储一些数据集的元信息在`dataset_params.json`中。

**提取embedding**：

我们使用的是腾讯的[Embedding](https://ai.tencent.com/ailab/nlp/embedding.html)，下载下来，里面有一个`Tencent_AILab_ChineseEmbedding.txt`文件，把它放在`embedding`文件夹下。

然后运行以下脚本，得到一个`embedding200.npy`:

```shell
python build_embedding.py --words_dir 你的数据集的位置 --out_dir 你想存储embedding的位置
# 其他参数看代码
```

这个脚本的主要作用是提取embedding，并建立一个和之前构建的字典一个顺序的embedding字典。例如`words.txt` 第一个词是 `我`，则npy数组的第一个元素就是`我`的embedding。

**标贝数据集的embedding** 下载方法

```shell
sh embedding/download_biaobei_embedding.sh
```

## Train

- 从头开始

```shell
# 以训练biaobei2为例
python train.py --data_dir data/biaobei2 --model_dir experiments/base --emb_dir embedding/biaobei 
```

- 继续上一次的训练

```shell
python train.py --data_dir data/biaobei2 --model_dir experiments/base --emb_dir embedding/biaobei --restore_file last
```

## Evaluation

与训练类似

```shell
python evaluate.py --data_dir data/biaobei2 --model_dir experiments/base --emb_dir embedding/biaobei
```

## Predict

运行交互式命令行实时预测。

```shell
python demo.py
```

如果你要批量预测，可以参考demo的代码自己写脚本。

## Code Structure

```bash
.
├── core
│   ├── inference             # inference专用代码, 用于其他代码调用
│   ├── model                 # 模型定义, 训练和inference都会用到
│   ├── __init__.py
│   └── utils.py              # 训练和inference都会用到的一些helper函数
├── data                 			# 训练数据
├── embedding             		# 训练用的embedding
├── experiments               # 保存训练的结果,模型,参数等
├── pretrained                # 预训练模型, 包含embedding。
├── build_biaobei_dataset.py  # 构建biaobei数据集。
├── build_embedding.py        # 提取embedding, 具体见下文
├── build_vocab.py            # 构建字母表     
├── search_hyperparams.py     # 用于搜索最优的超参数
├── synthesize_results.py     # 将experiments文件夹里的实验结果汇总            
├── evaluate.py
├── train.py
├── demo.py
├── training-guide.md
└── README.md
```

 