# UniTrackFormer

UniTrackFormer 是一个基于 Transformer 的端到端模型，专注于解决 TrackML Challenge 中的粒子轨迹重建问题。该项目旨在通过深度学习技术，将粒子探测器中的测量数据（hits）分组为属于同一粒子的轨迹。

## 项目结构
trackML/ ├── code/ │ ├── src/ │ │ ├── dataset.py # 数据集加载与特征提取 │ │ ├── models.py # UniTrackFormer 模型定义 │ │ ├── losses.py # 损失函数模块 │ │ ├── trainer.py # 训练逻辑 │ │ └── main.ipynb # 数据分析与模型训练的 Jupyter Notebook │ └── test/ # 测试代码 ├── data/ # 数据文件夹（需自行下载） │ ├── detectors.csv # 探测器几何信息 │ ├── train_sample/ # 示例训练数据 │ └── test_sample/ # 示例测试数据 ├── trackml-library/ # TrackML 官方工具库 └── README.md # 项目说明文档

## 特性

- **数据加载与特征提取**：通过 `TrackMLDataset` 类加载数据并提取特征，包括几何信息、信号强度等。
- **模型设计**：基于 Transformer 的 UniTrackFormer 模型，包含编码器、解码器和多任务输出头。
- **多任务损失**：结合分类损失、掩码损失和参数回归损失，优化轨迹分类与参数预测。
- **GPU 加速**：支持 CUDA 和 Apple MPS 后端，提升训练效率。

## 安装

1. 克隆项目：
   bash
   git clone https://github.com/Ivan-Tang/UniTrackFormer.git
   cd UniTrackFormer

2. 下载数据：
    - 前往 [TrackML Challenge](https://www.kaggle.com/c/trackml-particle-identification) 页面，下载数据集。
    - 将数据解压到 `data/` 文件夹下。
    确保数据结构类似于：
    data/
    ├── detectors.csv
    ├── truth.csv
    ├── particles.csv
    ├── train_sample/
    │   ├── event000001000-hits.csv
    │   ├── event000001000-cells.csv
    │   ├── event000001000-particles.csv
    │   └── event000001000-truth.csv
    |   ...
    └── test/
        ├── event000001000-hits.csv
        ├── event000001000-cells.csv
        ...

3. 安装依赖
   本项目使用简单的机器学习包，因此只需安装python, pandas, numpy, pytorch
   此外，还需要安装trackml-library，请参考[官方github仓库](https://github.com/LAL/trackml-library)安装。

## 训练

交互式训练脚本位于 `code/src/main.ipynb` 文件中，可直接运行。
也可运行`code/src/trainer.py`进行模型训练



    






