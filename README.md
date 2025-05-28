# UniTrackFormer

**UniTrackFormer** 是一个基于 Transformer 的端到端模型，专注于 TrackML Challenge 粒子轨迹重建。

---

## 📦 安装与准备

1. 克隆项目

```bash
git clone https://github.com/Ivan-Tang/UniTrackFormer.git
cd UniTrackFormer
```

2. 下载数据集

前往 [TrackML Challenge (Kaggle)](https://www.kaggle.com/c/trackml-particle-identification) 下载数据，解压到 `data/` 目录：

```
data/
├── detectors.csv
├── sample_submission.csv
├── train_sample/
├── train_1_events/
├── train_10_events/
└── test/
```

3. 安装依赖

```bash
pip install numpy pandas torch scikit-learn
```
推荐：[`trackml-library`](https://github.com/LAL/trackml-library)

---

## 🚀 训练与评估

- 交互式训练：
  - 运行 `src/main.ipynb` 体验单事件训练与可视化
- 脚本训练：
  - `python src/trainer.py` 单事件训练
  - `python src/kfold_trainer.py` K折交叉验证
- 超参数可在 `models.py`、`trainer.py`、`kfold_trainer.py` 调整
- 显存不足可调小 `models.py` 的 `self.max_hits`

---

## 📂 项目结构（2025-05-28 更新）

```
trackML/
├── checkpoints/           # 训练模型权重与loss曲线
├── config/                # 配置文件
├── data/                  # 数据集
├── docs/                  # Sphinx文档
├── results/               # 可视化输出
├── src/                   # 主要代码
├── trackml-library/       # TrackML官方工具库
├── README.md
└── 开题报告.pptx
```

src/ 主要文件：
- dataset.py         数据集加载与特征提取
- kfold_trainer.py   K折训练
- losses.py          多任务损失
- main.ipynb         数据探索与训练
- metric.py          评估指标
- models.py          UniTrackFormer结构
- trainer.py         训练主流程
- visual.py          可视化

results/ 主要输出：
- 3d_hits.png、hits_rz.png、ground_truth.png、predictions.png

checkpoints/ 主要成果：
- best_model.pth、unitrackformer_checkpoint.pth、loss_curve.png

---

## 📈 当前项目进度（2025-05-28）

- **数据准备**：已完成，支持多事件数据集
- **配置管理**：统一配置文件
- **数据处理**：特征提取、标签生成，兼容TrackML格式
- **模型实现**：UniTrackFormer结构，Transformer编码-解码、query聚类、参数回归
- **损失与评估**：多任务损失与主流评估指标（efficiency、fake rate等）
- **训练流程**：支持单事件与K折训练，自动保存最优模型与loss曲线
- **可视化**：3D分布、rz投影、真值轨迹、预测轨迹等多种可视化，输出至results/
- **文档与报告**：Sphinx文档结构完善，含开题PPT
- **模型成果**：checkpoints/ 下已保存多轮训练权重与loss曲线

---

如需详细模块说明或进度表，可进一步细化。欢迎贡献与交流！
