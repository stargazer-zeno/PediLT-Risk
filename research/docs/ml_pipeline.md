# Machine-learning pipeline

三个模型共享同一训练集和序列预处理：

- XGBoost：静态变量、原始时间步展开、时间 mask 和 sequence length；每个 horizon 独立训练，并使用 `StratifiedGroupKFold` 生成患者级 OOF。
- LSTM：Packed LSTM 时序分支与静态 MLP 分支拼接，按患者划分验证集，使用 early stopping。
- RSF：同一展平特征经过训练集均值填补，使用生存时间/事件标签拟合，在 30、365、1825 天读取死亡概率。

模型输入形状和合成数据见 `machine_learning/examples/input_shapes.json`。训练产生的模型权重、NPZ 数据集和患者级预测不随本仓库提供。
