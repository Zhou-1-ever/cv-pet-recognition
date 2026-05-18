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
