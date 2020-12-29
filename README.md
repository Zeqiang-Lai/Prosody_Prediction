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

```
链接: https://pan.baidu.com/s/1YWH65bM-NGZYOAoK8C3rBQ 密码: 6r1w
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

[实验记录](docs/log.md)

如果你要自己训练，评测模型，构建数据集，请查看 [training-guide.md](docs/training-guide.md)


## FAQ

1. 运行交互式命令行进行测试时遇到:

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 in position 0: invalid continuation byte
```

解决方法(暂时): 这个错误通常发生在输入文字后又删除, 重启保证一次输对。

## TODO

See [todo.md](docs/todo.md)

## Reference

See [ref.md](docs/ref.md)

