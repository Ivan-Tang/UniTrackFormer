# TrackML 粒子追踪与 TrackFormer 实现

本项目旨在使用 TrackFormer 架构（基于 arXiv:2411.07149v1）为 [TrackML 粒子追踪挑战赛](https://www.kaggle.com/c/trackml-particle-identification) 开发一个端到端的粒子追踪管线。该项目包括数据加载、预处理、模型训练、事件预测、性能评估和结果可视化。

## 特性

- **数据处理**: 使用 `trackml.dataset` 加载和处理事件数据（hits, cells, truth, particles）。
- **特征工程**: 为 TrackFormer 模型提取相关特征。
- **模型训练**: 基于 `src/trackformer.py` 中的 TrackFormer 模型进行训练（通过 `src/trainer.py` 或 `src/demo_complete_pipeline.py`）。
- **预测**: 对新事件进行粒子轨迹预测。
- **后处理**: 将原始模型输出转换为标准的 TrackML 提交格式。
- **评估**: 使用 `src/metric.py` 中的评估指标将预测结果与真实数据进行比较。
- **可视化**:
    - 原始 3D hits 和 R-Z 投影图。
    - 真实径迹和预测径迹的可视化对比。
    - 使用 `src/advanced_visualization.py` 进行高级分析和报告生成。
- **完整管线**: `src/demo_complete_pipeline.py` 脚本演示了从数据探索到最终评估和可视化的完整流程。

## 项目结构

```
trackML/
├── README.md
├── TRACKFORMER_SUMMARY.md
├── checkpoints/            # 存储训练好的模型 (例如 best_model.pth)
├── config/                 # 配置文件 (例如 config.yaml)
├── data/                   # TrackML 数据集
│   ├── detectors.csv
│   ├── sample_event/       # 单个示例事件
│   ├── test/               # 测试数据集
│   └── train_sample/       # 训练数据集样本
├── ref/                    # 参考论文
├── report/                 # 报告和演示文稿
├── results/                # 存储预测、评估和可视化结果
│   ├── exploration/
│   ├── predictions/
│   └── analysis/
├── src/                    # 源代码
│   ├── __init__.py
│   ├── advanced_visualization.py # 高级可视化工具
│   ├── dataset.py          # 数据加载和预处理
│   ├── demo_complete_pipeline.py # 端到端演示脚本
│   ├── losses.py           # 损失函数定义
│   ├── main_train.py       # (可能为早期训练脚本的入口)
│   ├── metric.py           # 评估指标
│   ├── test_and_predict.py # 预测和评估脚本
│   ├── trackformer.py      # TrackFormer 模型定义
│   ├── trainer.py          # 模型训练脚本
│   └── visual.py           # 基础可视化工具
└── trackml-library/        # TrackML 官方库
```

## 安装与设置

1.  **克隆仓库**:
    ```bash
    git clone <repository_url>
    cd trackML
    ```
2.  **环境**:
    建议使用 Python 3.8+。可以使用 `conda` 或 `venv` 创建虚拟环境。
3.  **依赖安装**:
    主要依赖包括 PyTorch, pandas, numpy, matplotlib, trackml。
    ```bash
    pip install torch torchvision torchaudio
    pip install pandas numpy matplotlib scikit-learn tqdm
    # 安装 trackml 库 (如果 trackml-library 是子模块或需要手动安装)
    cd trackml-library
    python setup.py install
    cd ..
    ```
    或者，如果提供了 `requirements.txt` 文件：
    ```bash
    pip install -r requirements.txt
    ```
4.  **数据准备**:
    *   将 `detectors.csv` 文件放置在 `data/` 目录下。
    *   将训练数据（例如 `eventXXXX-hits.csv`, `eventXXXX-cells.csv`, `eventXXXX-truth.csv`, `eventXXXX-particles.csv`）放置在 `data/train_sample/` 或 `data/train_10_events/` 等子目录中。
    *   测试数据放置在 `data/test/` 目录下。
    *   示例事件数据可放置于 `data/sample_event/`。

## 使用方法

### 1. 完整管线演示

`src/demo_complete_pipeline.py` 脚本提供了一个自动化的流程，包括数据探索、模型训练（可选）、预测、评估和高级可视化。

```bash
python src/demo_complete_pipeline.py --help
```
常用命令：
```bash
# 对默认事件 event000001000 运行完整管线 (如果模型不存在则会训练)
python src/demo_complete_pipeline.py

# 强制重新训练模型并处理指定事件
python src/demo_complete_pipeline.py --train --event event000001001

# 跳过训练，使用现有模型进行预测和可视化
python src/demo_complete_pipeline.py --skip_training --event event000001002
```
结果将保存在 `results/` 目录下的 `exploration`, `predictions`, `analysis` 子目录中，并生成 `results/pipeline_summary.txt`。

### 2. 单独训练模型

可以使用 `src/trainer.py` 脚本来训练模型。确保 `config/config.yaml` (如果使用) 或脚本内部参数配置正确。
```bash
python src/trainer.py
```
训练好的模型通常保存在 `checkpoints/best_model.pth`。

### 3. 单独进行预测和评估

使用 `test_and_predict.py` 脚本对单个事件进行预测、评估和可视化。
```bash
python test_and_predict.py --model_path checkpoints/best_model.pth --event_path data/train_sample/event000001000 --output_dir results/predictions
```
该脚本会：
1.  加载指定事件的数据。
2.  使用提供的模型进行预测。
3.  将预测结果保存为 CSV 文件。
4.  如果提供了真实数据，则进行评估。
5.  生成可视化图像。

## 关键脚本说明

-   `src/demo_complete_pipeline.py`: 核心演示脚本，整合所有阶段。
-   `src/dataset.py`: 负责加载数据、进行必要的预处理和特征提取。
-   `src/trackformer.py`: 包含 TrackFormer 模型的 PyTorch 实现 (`create_trackformer_600mev`)。
-   `src/trainer.py`: 包含模型训练的逻辑，包括优化器、损失函数和训练循环。
-   `test_and_predict.py`: 用于对单个或多个事件进行预测，将其输出保存为 TrackML 格式，并进行评估和可视化。
-   `src/losses.py`: 定义了训练过程中使用的损失函数。
-   `src/metric.py`: 实现了 TrackML 评分逻辑或自定义评估指标。
-   `src/visual.py` & `src/advanced_visualization.py`: 提供事件、径迹和性能的可视化工具。

## 模型

本项目使用基于 Transformer 的 TrackFormer 模型。预训练模型和训练过程中生成的最佳模型保存在 `checkpoints/` 目录下，例如 `checkpoints/best_model.pth`。

## 结果

所有输出，包括：
-   预测的径迹 (CSV格式)
-   性能评估报告/指标
-   可视化图像 (hits, 真实径迹, 预测径迹, 性能图表等)
都将保存在 `results/` 目录中，按不同阶段分子目录管理。
