# 🐾 计算机视觉期中作业：基于迁移学习的宠物识别 (Task 1)

> 课程：计算机视觉 | 组队人数：2人 | 框架：PyTorch

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
