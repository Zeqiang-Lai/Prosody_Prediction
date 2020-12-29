# TODO

- [ ] 可能的改进点: Joint Training

### 可能的改进点: Joint Training

现在是分别训练预测pw，pph和iph的模型。每个模型单词的标签都只有`B`(切分点),`I`(不是切分点)。

我们可以将标签扩展成四个

- `B_PW`: 韵律词的切分点
- `B_PPH`: 韵律短语的切分点
- `B_IPH`: 语调短语的切分点

通过这样，我们也许可以通过pw，pph，iph的关联关系，得到更准确的预测结果。

例如：韵律短语的切分点一般也是韵律词的切分点。
