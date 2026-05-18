# 🐾 计算机视觉期中作业：基于迁移学习的宠物识别 (Task 1)

> 课程：计算机视觉 | 组队人数：1人 | 框架：PyTorch

## 📦 1. 环境配置 (Environment Setup)
本项目基于 **Python 3.11** 与 **PyTorch 2.x** 开发。请按以下步骤配置运行环境：

```bash
# 1. 创建并激活虚拟环境（可选）
conda create -n cv python=3.11 -y
conda activate cv

# 2. 安装 PyTorch (请根据本地 CUDA 版本替换 index-url，CPU 用户移除 --index-url 参数)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装实验依赖库
pip install swanlab tqdm pandas matplotlib seaborn scikit-learn jupyter

```


## 📂 2. 数据集准备 (Dataset Preparation)
- **数据集名称**：Oxford-IIIT Pet Dataset (37 类宠物)
- **获取方式**：首次运行代码时，`torchvision.datasets.OxfordIIITPet` 会自动下载数据集至 `./data/pet_dataset` 目录（约 800MB）。
- **数据划分**：采用官方默认划分，`trainval` 作为训练集，`test` 作为验证集。
- **预处理**：
  - 训练集：使用 `RandomResizedCrop(224)` + `RandomHorizontalFlip` 进行数据增强。
  - 验证集：使用 `Resize(256)` + `CenterCrop(224)` 进行确定性变换。
  - 标准化：均使用 ImageNet 统计量进行标准化。

## 🚀 3. 训练与测试 (Training & Testing)
本项目所有实验均集成在 `cv.ipynb` 中。请启动 Jupyter 后按顺序执行：
```bash
jupyter notebook cv.ipynb
```
### 🔹 3.1 训练 Baseline 模型
- **运行位置**：`Cell 1 ~ Cell 5`
- **训练策略**：
  - 加载 ImageNet 预训练的 ResNet-18 权重
  - 替换最后的全连接层为 37 维输出
  - 采用**差异化学习率**：骨干网络 `lr=1e-4`（微调），分类头 `lr=1e-3`（从零训练）
  - 优化器：`Adam`，权重衰减 `1e-4`，`Batch Size = 32`，训练 `10 Epochs`
- **可视化记录**：训练过程自动同步至 SwanLab，包含 `Train/Val Loss` 与 `Accuracy` 曲线。

### 🔹 3.2 超参数分析与消融实验
- **超参数搜索**：运行 `Cell 10 ~ 13`（两阶段加速网格搜索，自动输出最优 LR 组合）
- **预训练消融**：运行 `Cell 14 ~ 17`（对比 `Random Init` 与 `Pretrained` 的性能差异）
- **注意力机制**：运行 `Cell 18 ~ 21`（集成 SE-Block 的 `SE-ResNet18` 对比实验）

### 🔹 3.3 模型测试与评估
- **运行位置**：训练完成后，执行 `Cell 6 ~ 8`（可视化模块）
- **输出结果**：
  - 终端打印验证集最终 `Accuracy` 与 `Loss`
  - 自动生成 `training_curves.png`（训练曲线）
  - 自动生成 `confusion_matrix.png`（类别混淆矩阵）
  - 自动生成 `prediction_samples.png`（随机预测样例可视化）

### 🔹 3.4 引入 Transformer / 注意力机制对比
- **运行位置**：`Cell 18 ~ 21`（插入 SE/CBAM 模块）或 `Cell 22 ~ 25`（替换为 ViT-Tiny/Swin-T）
- **模型架构**：
  - 方案 A：在 ResNet-18 的每个残差块后插入 `SE-Block`（通道注意力）或 `CBAM`（通道+空间注意力）
  - 方案 B：直接替换骨干网络为轻量级 `ViT-Tiny` 或 `Swin-Tiny`，保留 37 维分类头
- **训练策略**：
  - 优化器与学习率策略与 Baseline 保持一致（Adam, `lr_backbone=1e-4`, `lr_head=1e-3`）
  - 针对 Transformer 架构额外引入 `DropPath=0.1` 与 `Label Smoothing=0.1` 防止过拟合
  - 训练 `10 Epochs`，全程同步 SwanLab 记录
- **对比分析**：
  - 重点对比验证集 `Accuracy` 与 Baseline 的差异
  - 分析 Transformer/注意力机制在细粒度特征提取、参数量开销及收敛特性上的表现
- **输出结果**：
  - 终端打印最终精度对比（Baseline vs Transformer/Attention）
  - 自动生成 `transformer_comparison.png`（精度对比柱状图与收敛曲线）
