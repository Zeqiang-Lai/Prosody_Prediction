# 实验记录

这个是后补的，以前记录的找不到了。调参的记录太久远了，就不补了。以下是预训练模型的记录。

## 结果

- 训练checkpoint下载地址为[biaobei_dataset.tar.gz](http://dvmvd-4602.kmlltpro.corp.kuaishou.com/prosody_prediction/biaobei_dataset.tar.gz)

下载完成后放在experiments文件夹下，可以用`train.py`接着训（如果你想的话）。

`biaobei_dataset.tar.gz`解压后有三个文件夹

```
biaobei_pw    # 韵律词，用biaobei1训练
biaobei_pph   # 韵律短语，用biaobei2训练
biaobei_iph   # 语调短语，用biaobei3训练
```

- Release的模型：[biaobei_pretrained.tar.gz](http://dvmvd-4602.kmlltpro.corp.kuaishou.com/prosody_prediction/biaobei_pretrained.tar.gz)
- 使用的数据集：[biaobei_dataset.tar.gz](http://dvmvd-4602.kmlltpro.corp.kuaishou.com/prosody_prediction/biaobei_dataset.tar.gz)
- Git Commit: 

| 模型       |   数据集   | Performance                                                  |
| ---------- | ---- | ------------------------------------------------------------ |
| biaobei_pw | biaobei1 | accuracy: 0.939 ; block_acc: 0.887 ; precison: 0.864 ; recall: 0.950 ; loss: 0.166 |
| biaobei_pph | biaobei2 | accuracy: 0.932 ; block_acc: 0.767 ; precison: 0.612 ; recall: 0.889 ; loss: 0.170 |
| biaobei_iph | biaobei3 | accuracy: 0.982 ; block_acc: 0.905 ; precison: 0.875 ; recall: 0.945 ; loss: 0.057 |

Performance得到的方法:

```shell
python evaluate.py --data_dir data/biaobei1 --model_dir experiments/biaobei_pw/
python evaluate.py --data_dir data/biaobei2 --model_dir experiments/biaobei_pph/
python evaluate.py --data_dir data/biaobei3 --model_dir experiments/biaobei_iph/
```

## 指标

| 名称      | 解释                   |
| --------- | ---------------------- |
| accuracy  | BI标签预测正确的比例。 |
| block_acc | 切分段预测正确的比例。 |
| precision | BI标签预测的准确率     |
| recall    | BI标签预测的召回率     |

- 关于block_acc, 例

> 你好同学，假设正确的是 你好/同学(BIBI)
>
> 如果切成你好同学(BIII), accuracy=50%， block_acc=0



