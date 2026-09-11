# CRABR 项目文档

CRABR (Cross-level Relation-Aware Boundary Reasoning) 是一个用于医学图像分割的深度学习模型，专注于头颅侧位片等医学影像分析任务。默认主模型为 **CRABR**：共享特征与解码器宽度均为 128，并关闭双向区域—边缘约束；其余边缘、区域、ERGA 和几何先验路径保持启用。

## 项目结构

```
CRABR/
├── ablation/                 # 消融实验包
│   ├── __init__.py
│   ├── groups.py             # 消融实验配置（3模块渐进式）
│   └── runner.py             # 消融实验运行器
├── benchmark/                # 对比实验包
│   ├── __init__.py
│   ├── benchmark_matrix.yaml # 对比实验配置
│   ├── compare_results.py    # 结果对比分析
│   ├── infer_sota.py        # SOTA模型推理
│   ├── models/              # SOTA模型定义
│   │   ├── __init__.py
│   │   └── sota_models.py   # 10个对比模型实现
│   ├── run_benchmark.py     # 对比实验主入口
│   └── train_sota.py        # SOTA模型训练
├── configs/                  # 数据集配置
│   ├── jsrt_scr.yaml        # JSRT-SCR 512配置
│   └── vindr_rib.yaml        # VinDr-Rib 512配置
├── kfold/                    # K折交叉验证包
│   ├── __init__.py
│   ├── splits.py            # K折划分逻辑
│   └── runner.py            # K折实验运行器
├── models/                   # 模型代码
│   ├── __init__.py
│   ├── anatomical_geometry_prior.py
│   ├── disjoint_boundary.py
│   ├── encoder.py
│   ├── erga.py
│   ├── feature_enhancement.py
│   ├── improved_decoder.py
│   ├── inner_boundary.py
│   ├── model.py
│   ├── pretrained_encoder.py
│   ├── pyramid_transformer_encoder.py
│   ├── relation_aware_overlap.py
│   ├── relation_aware_region_to_edge.py
│   ├── scale_router.py
│   ├── seg_head.py
│   └── shared_boundary.py
├── losses/                   # 损失函数
│   ├── __init__.py
│   └── segmentation.py
├── metrics/                  # 评估指标
│   ├── __init__.py
│   └── segmentation.py
├── utils/                    # 工具包
│   ├── __init__.py
│   ├── config.py            # 配置加载
│   ├── dataset.py           # 数据集构建
│   ├── training.py          # 训练辅助（ModelEMA, AverageMeter等）
│   ├── tta.py               # 测试时增强
│   └── visualization.py     # 可视化工具
├── viz/                      # 可视化包
│   ├── __init__.py
│   ├── comparison.py        # 对比可视化
│   ├── palette.py           # 调色板工具
│   └── prediction_matrices.py  # 预测矩阵可视化
├── benchmark/               # 对比实验（与主benchmark目录相同）
├── configs/                  # 数据集配置
├── train.py                 # 训练脚本
├── infer.py                 # 推理脚本
├── validate.py              # 验证脚本
├── count_params.py          # 参数量统计
├── make_comparison_figures.py  # 生成对比图
├── run_ablation.py          # 消融实验入口
├── run_kfold.py            # K折实验入口
└── requirements.txt         # 依赖包
```

## 核心功能

### 1. 消融实验 (ablation/)

3阶段渐进式模块消融：

| Stage | 名称 | 模型配置 |
|-------|------|---------|
| 1 | Baseline | 基础模型，无FE、无双分支 |
| 2 | +FE+HCLF+Branch | +特征增强+层级融合+双分支 |
| 3 | +GeoLoop | +几何循环(coarse SDM+ICDC+curriculum) |
| 4 | Full | 完整ERGA模型 |

**使用方式：**
```bash
# 完整消融实验
python run_ablation.py

# 快速调试（30 epochs，单种子）
python run_ablation.py --quick

# 指定数据集
python run_ablation.py --configs configs/jsrt_scr.yaml

# 自定义随机种子
python run_ablation.py --seeds 42 1337 2025
```

### 2. K折交叉验证 (kfold/)

```bash
# 5折交叉验证
python run_kfold.py --config configs/jsrt_scr.yaml --folds 5 --seed 42

# 指定GPU
python run_kfold.py --config configs/vindr_rib.yaml --gpu-ids 0,1
```

### 3. 对比实验 (benchmark/)

对比模型（共10个SOTA）：

| 模型 | 系列 |
|------|------|
| CRABR (ours) | 自研模型 |
| Swin-UNet | Transformer |
| U-Mamba | Mamba |
| VM-UNet v2 | Mamba |
| MSVM-UNet | Mamba |
| SegMamba | Mamba |
| KM-UNet | CNN+Mamba混合 |
| DCM-Net | CNN+Mamba混合 |
| CFM-UNet | CNN+Mamba混合 |
| I2U-Net | 双路径混合 |
| TBConvL-Net | 双路径混合 |

**使用方式：**
```bash
# 运行全部对比实验
python -m benchmark.run_benchmark --stage all --gpu-ids 0,1,2,3

# 仅训练
python -m benchmark.run_benchmark --stage train

# 仅推理
python -m benchmark.run_benchmark --stage infer --split test

# 生成对比报告
python -m benchmark.run_benchmark --stage summary
```

### 4. 单独训练/推理

默认配置已使用 CRABR 主模型参数，无需额外的模型参数。当前主模型的通道数与旧版 Full CRABR 不同，旧版 checkpoint 不能直接加载；请重新训练。

```bash
# 训练
python train.py --config configs/jsrt_scr.yaml --gpu-ids 0,1,2,3

# 推理
python infer.py --config configs/jsrt_scr.yaml --checkpoint checkpoints/best_model.pth

# 验证
python validate.py --config configs/jsrt_scr.yaml

# 参数量统计
python count_params.py --config configs/jsrt_scr.yaml
```

## 数据集配置

项目支持以下数据集（512尺寸）：

| 数据集 | 配置文件 | 任务类型 |
|--------|---------|---------|
| JSRT-SCR | configs/jsrt_scr.yaml | Multilabel (3类) |
| VinDr-Rib | configs/vindr_rib.yaml | Binary (1类) |

### 配置说明

主要配置项：
- `data`: 数据路径、增强参数
- `model`: 模型架构参数
- `loss`: 损失函数权重
- `train`: 训练策略（epochs、batch_size、lr等）
- `inference`: 推理输出设置

## 输出结构

```
checkpoints/
├── jsrt_scr/
│   ├── best_model.pth
│   └── best_metrics.json
└── vindr_rib/
    ├── best_model.pth
    └── best_metrics.json

predictions/
├── jsrt_scr/
└── vindr_rib/

results/
└── *_metrics.json

ablation_results/
├── summary.json
├── ablation_summary_512.csv
├── ablation_summary_512.md
└── runs.json
```

## 环境依赖

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
Pillow>=9.0.0
scipy>=1.7.0
PyYAML>=6.0
tensorboard>=2.10.0
tqdm>=4.64.0
SimpleITK>=2.2.0
medpy>=0.4.0
scikit-image>=0.19.0
einops>=0.6.0
```

## 注意事项

1. **数据集路径**：配置文件中使用绝对路径，请根据实际数据位置修改
2. **GPU配置**：默认使用多GPU训练，根据实际硬件修改 `gpu_ids`
3. **模型保存**：仅保存最佳权重（`best_model.pth`），不保存最后轮次权重
4. **Early Stopping**：默认基于 `boundary_score` 进行早停监控
