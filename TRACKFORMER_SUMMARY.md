# TrackFormer 项目报告与总结

**项目名称**: 基于 TrackFormer 的 TrackML 粒子追踪管线  
**日期**: 2024年7月26日  
**版本**: 1.0  
**主要贡献者**: GitHub Copilot & IvanTang  
**参考架构**: TrackFormer (arXiv:2411.07149v1)  

## 1. 项目概述

本项目的目标是为 TrackML 粒子追踪挑战赛实现一个完整的机器学习管线。该管线基于 TrackFormer 架构，一个利用 Transformer 网络进行径迹重建的深度学习模型。项目涵盖了从数据加载和预处理，到模型训练、推理、评估和高级可视化的所有阶段。

## 2. 已完成的工作和特性

### 2.1. 数据处理与特征工程 (`src/dataset.py`)
- 实现了使用 `trackml.dataset.load_event` 和 `trackml.dataset.load_dataset` 加载 TrackML 事件数据（hits, cells, truth, particles）的功能。
- 开发了 `TrackMLDataset` 类，用于 PyTorch DataLoader，支持批量加载和特征提取。
- 实现了 `extract_features` 方法，用于从原始 hits 数据中提取 TrackFormer 模型所需的特征，如 `x, y, z, r, phi, eta` 以及可能的 cell 信息。

### 2.2. TrackFormer 模型 (`src/trackformer.py`)
- 集成了 TrackFormer 模型架构 (`create_trackformer_600mev`)。
- 模型接受 hits 特征和坐标作为输入，并输出径迹段和分类分数。

### 2.3. 模型训练 (`src/trainer.py`, `src/main_train.py`)
- 实现了 `Trainer` 类，封装了训练循环、优化器设置 (AdamW)、学习率调度和损失计算。
- 使用了在 `src/losses.py` 中定义的组合损失函数（例如，focal loss, L1 loss）。
- 训练脚本支持从检查点恢复训练，并将最佳模型保存到 `checkpoints/best_model.pth`。
- 解决了在直接运行 `src/trainer.py` 时出现的 `ModuleNotFoundError`。
- 修正了模型导入和前向传播调用中的参数问题。

### 2.4. 预测与后处理 (`test_and_predict.py`)
- 开发了 `TrackMLPredictor` 类，用于加载预训练模型并对新事件进行预测。
- `predict_on_hits`: 执行模型推理，获取原始输出。
- `postprocess_predictions`: 将模型输出转换为 TrackML 提交格式（`event_id`, `hit_id`, `track_id`）。正确处理了 TrackFormer 输出中的 `filtered_indices`，以确保仅使用有效的预测段。
- `save_predictions_csv`: 将后处理的预测结果保存为 CSV 文件。

### 2.5. 评估 (`src/metric.py`, `test_and_predict.py`)
- 实现了 `evaluate_event_predictions` 函数，使用官方 `trackml.score.score_event` 或 `src.metric.evaluate_metrics` 中的自定义评分逻辑来评估预测径迹的准确性。
- 评估指标包括 TrackML score, precision, recall, F1-score 等。

### 2.6. 可视化 (`src/visual.py`, `src/advanced_visualization.py`, `test_and_predict.py`)
- **基础可视化 (`src/visual.py`)**: 提供了绘制 3D hits、R-Z 投影、单个径迹等功能。
- **事件预测可视化 (`test_and_predict.py`)**: 创建了 `visualize_event_predictions` 函数，能够：
    - 显示原始 hits (3D 和 R-Z)。
    - 叠加显示真实径迹。
    - 叠加显示预测径迹。
    - 区分正确预测、错误预测和未找到的径迹。
- **高级可视化 (`src/advanced_visualization.py`)**: 包含用于生成综合报告图表的功能，例如：
    - 径迹长度分布。
    - 不同粒子类型的检测效率。
    - 动量/角度分辨率。
    - 混淆矩阵分析。

### 2.7. 完整管线演示 (`src/demo_complete_pipeline.py`)
- 创建了一个端到端的脚本，整合了上述所有步骤。
- 支持命令行参数配置，例如事件 ID、是否训练、模型路径等。
- 自动化流程：
    1. 数据探索与可视化。
    2. 模型训练（如果需要或模型不存在）。
    3. 使用训练好的模型进行预测。
    4. 对预测结果进行评估。
    5. 生成高级可视化报告和摘要。
- 所有输出（预测、图表、报告）均保存在结构化的 `results/` 目录中。

### 2.8. 项目文档
- 创建了 `README.md`，详细说明了项目特性、结构、安装步骤和使用方法。
- 本文档 (`TRACKFORMER_SUMMARY.md`) 总结了项目的主要成果和实现细节。

## 3. 关键代码模块

- `src/dataset.py`: 数据集处理和特征提取。
- `src/trackformer.py`: TrackFormer 模型定义。
- `src/trainer.py`: 模型训练逻辑。
- `test_and_predict.py`: 独立脚本，用于对单个事件进行预测、评估和可视化。
- `src/metric.py`: 评估指标实现。
- `src/visual.py`, `src/advanced_visualization.py`: 可视化工具。
- `src/demo_complete_pipeline.py`: 端到端管线演示。
- `src/losses.py`: 损失函数。

## 4. 遇到的挑战与解决方案

- **环境与依赖**: 确保所有必要的库（PyTorch, trackml, pandas, etc.）正确安装并兼容。
- **模块导入**: 在 `src/trainer.py` 中通过修改 `sys.path` 解决了相对导入问题。
- **模型接口**: 确保传递给模型 `forward` 方法的参数（如 `coords`）正确无误。
- **后处理逻辑**: 正确解析 TrackFormer 的输出，特别是 `filtered_indices`，以生成有效的径迹 ID。
- **数据路径管理**: 使用 `os.path.join` 和绝对路径来确保脚本在不同环境下都能正确找到数据和模型文件。

## 5. 未来工作与改进方向

- **超参数优化**: 对 TrackFormer 模型和训练过程进行更细致的超参数调整。
- **大规模数据训练**: 在完整的 TrackML 训练集上进行训练，并评估其泛化能力。
- **性能优化**: 优化代码以提高数据加载、训练和推理的速度。
- **模型变体探索**: 尝试不同的 TrackFormer 配置或结合其他先进的图神经网络方法。
- **更复杂的后处理**: 研究更高级的径迹连接和过滤算法。
- **交互式可视化**: 开发基于 Web 的或更具交互性的可视化工具，以便更深入地分析事件和模型性能。

## 6. 结论

本项目成功实现了一个基于 TrackFormer 的粒子追踪管线，能够处理 TrackML 数据，训练模型，进行预测，并对结果进行评估和可视化。通过 `test_and_predict.py` 和 `src/demo_complete_pipeline.py` 等脚本，用户可以方便地运行和测试整个流程。代码结构清晰，模块化程度高，为进一步的研究和开发奠定了坚实的基础。
