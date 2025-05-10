# UniTrackFormer

**UniTrackFormer** 是一个基于 Transformer 的端到端模型，专注于解决 TrackML Challenge 中的粒子轨迹重建问题。本项目旨在通过深度学习技术，将粒子探测器中的测量数据（hits）划分为属于同一粒子的轨迹，实现高效、可扩展的轨迹重建。

---

## 📁 项目结构

```bash
trackML/
├── code/
│   ├── src/
│   │   ├── dataset.py         # 数据集加载与特征提取
│   │   ├── models.py          # UniTrackFormer 模型定义
│   │   ├── losses.py          # 多任务损失函数实现
│   │   ├── trainer.py         # 模型训练逻辑
│   │   ├── kfold_trainer.py   # kfold模型训练逻辑
│   │   ├── metric.py          # 评价efficiency/fake_rate函数
│   │   └── main.ipynb         # 数据探索与训练 Notebook
│   └── test/                  # 测试脚本（可选）
├── data/                      # 数据目录（需手动下载）
│   ├── detectors.csv          # 探测器几何信息
│   ├── train_sample/          # 示例训练集（多个事件）
│   └── test_sample/           # 示例测试集
├── trackml-library/           # TrackML 官方工具库（需手动克隆）
└── README.md                  # 当前说明文档
```

---

## ✨ 项目特性

- **数据加载与特征提取**：从 hits, cells, detectors 构建输入特征，支持空间、电荷和模块信息。
- **模型设计**：Transformer 编码器 + 解码器结构，使用 query 探测轨迹，支持聚类与回归。
- **多任务损失**：包含轨迹分类、掩码聚类、物理参数回归三项子任务。
- **轻量高效**：模型支持 CUDA 和 Apple MPS 加速，推理时间短，结构易部署。

---

## 📦 安装与准备

### 1. 克隆项目

```bash
git clone https://github.com/Ivan-Tang/UniTrackFormer.git
cd UniTrackFormer
```

### 2. 下载数据集

前往 [TrackML Challenge (Kaggle)](https://www.kaggle.com/c/trackml-particle-identification) 页面下载以下数据：

- `train_sample/`、`test/` 等事件数据
- `detectors.csv`、`particles.csv`、`truth.csv` 等辅助信息

解压到 `data/` 目录，使其结构如下：

```bash
data/
├── detectors.csv
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
```

### 3. 安装依赖

建议使用 Python ≥ 3.8，核心依赖如下：

```bash
pip install numpy pandas torch scikit-learn
```

额外依赖（推荐）：

- [`trackml-library`](https://github.com/LAL/trackml-library)：用于解析和可视化 TrackML 数据（根据仓库中的指引在环境中安装trackML）

---

## 🚀 模型训练与评估

### Jupyter Notebook 交互训练

```bash
打开 code/src/main.ipynb 查看并运行单事件数据训练流程
```

### 脚本方式训练

```bash
python src/trainer.py
python src/kfold_trainer.py
```

你可以在 `model.py` `trainer.py` `kfold_trainer.py` 中修改超参数（如 batch size、学习率等），支持多事件迭代训练与保存模型。
视训练平台的显存大小，可以调整model.py中的self.max_hits，这将决定每个batch送入模型的hits数量。

---

如需进一步改进或贡献代码，请提交 PR 或联系作者。
