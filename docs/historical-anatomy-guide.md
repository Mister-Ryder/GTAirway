# 气道分割框架技术指南
这个是个很旧的版本
本指南提供了关于增强型气道分割框架的详细信息，该框架结合了解剖学分类和先进的分割技术，以改进CT图像中气道树的检测和分割效果。

## 项目概述

本项目实现了一个多阶段气道分割框架，利用解剖学知识来提高分割准确性。该框架由三个主要组件组成：

1. **解剖学分类器**：基于代数类型、结构类型和位置对气道结构进行分类
2. **基础分割模型**：执行初步的气道分割
3. **增强型分割**：将解剖学特征融入分割过程，提升结果质量

## 输入/输出规格

### 输入
- CT图像卷（3D）：NIFTI（.nii.gz）格式
- 二值分割掩膜（用于训练）
- 解析后的气道树元数据（可选，用于高级功能）

### 输出
- 气道的二值分割结果
- 解剖学分类结果（如果使用完整管道）：
  - 代数类型（气管、主支气管、叶支气管等）
  - 结构类型（管状、分叉、复杂分支）
  - 位置类型（上/中/下肺、左/右肺）
- 特征图和注意力图（用于可视化目的）

## 数据结构

### 目录结构
```
/
├── checkpoint/                  # 模型检查点
├── dataset_pre/                 # 生成元数据
│   ├── AirwayMetadataGenerator.py
├── dataset_info/                # 元数据和数据集信息 airway_metadata_train_anatomical.pkl
├── func/                        # 核心功能模块
│   ├── AnatomicalClassifier.py  # 气道分类模型
│   ├── AnatomicalFusionModules.py # 特征融合模块
│   ├── EnhancedAirwaySegmentation.py # 增强型分割模型
│   ├── loss_func.py            # 损失函数
│   ├── load_dataset.py         # 数据集加载工具
│   └── model_arch_model10_plus_tou.py # 基础分割架构
├── runs/                        # 基础模型训练日志
├── runs_anatomical_classifier/  # 分类器训练日志
├── runs_enhanced_airway/        # 增强型模型训练日志
└── test_data/                   # 评估用测试数据
```

### 输入数据格式
该框架使用多种数据格式：

1. **原始数据**：NIFTI（.nii.gz）格式的CT扫描
2. **预处理数据**：裁剪和归一化的图像块（.npy格式）
3. **元数据**：存储在pickle（.pkl）格式的解剖学特征
4. **解析数据**：NIFTI和NumPy格式的气道分支信息

### 数据字典格式
该框架使用Python字典组织数据样本。每个条目包含：

- `image_path`：CT图像路径
- `label_path`：分割掩膜路径
- `position`：图像块在原始卷中的(z, y, x)坐标
- `image_size`：原始图像大小
- `crop_size`：裁剪的图像块大小
- `stride`：裁剪使用的步长
- 解剖学特征：
  - `generation_type`：气道代数类型（0-6）
  - `structure_type`：气道结构类型（0-6）
  - `location_type`：空间位置（0-6）
  - `airway_pixel_num`：气道像素数量
  - `airway_density`：图像块中气道密度
  - `complexity`：结构复杂度指标
  - 其他气道特性指标

## 模型架构

### 解剖学分类器
解剖学分类器（`AirwayAnatomicalClassifier`）是一个3D CNN，预测：
- 代数类型（7类）：无气道、气管、主支气管、叶支气管、段支气管、小支气管、细支气管
- 结构类型（7类）：无结构、直管道、圆形管道、分叉点、复杂分支区域（3级）
- 位置类型（7类）：未定位、上肺、中肺、下肺、左肺、右肺、气管区域

### 基础分割模型
基础模型（`SegAirwayModel`）是一个U-Net风格的3D CNN，具有：
- 4个编码器块，通道维度为[32, 64, 128, 256]
- 3个解码器块，通道维度为[128, 64, 32]
- 对应编码器和解码器层之间的跳跃连接
- 用于增强特征提取的注意力模块和扩张卷积

### 增强型分割模型
增强型模型（`EnhancedAirwaySegmentation`）通过以下方式将解剖学分类器与分割网络集成：
- `GenerationGuidanceModule`：用于深层解码器层，增强细小气道检测
- `StructureGuidanceModule`：用于中层解码器层，增强分叉建模
- `LocationGuidanceModule`：用于编码器层，引导空间感知

## 运行指南

### 环境设置
```bash
# 创建并激活conda环境
conda create -n airway_seg python=3.8
conda activate airway_seg

# 安装所需包
pip install torch torchvision
pip install nibabel SimpleITK scikit-image tqdm tensorboard
pip install pandas numpy scipy matplotlib
```

### 数据准备
1. **生成元数据**：
```bash
# 基于原始数据和解析结果生成气道元数据
python AirwayMetadataGenerator.py
```

2. **裁剪图像**（如需要）：
```bash
# 将原始CT图像裁剪成图像块
python BAS_pre_crop_image.py
```

### 模型训练

#### 项目运行顺序

1. **数据准备阶段**：
   - 首先执行`AirwayMetadataGenerator.py`生成解剖学元数据
   - 然后执行`BAS_pre_crop_image.py`处理原始图像并生成固定大小的图像块

2. **模型训练阶段**：
   - 第一步：训练解剖学分类器`train_anatomical_classifier.py`
   - 第二步：训练基础分割模型`train_skeleton_model10-ronghe.py`
   - 第三步：使用预训练的分类器和分割模型，训练增强型分割模型`training_script_enhanced_airway.py`

3. **推理/评估阶段**：
   - 使用训练好的模型进行气道分割和分类

#### 1. 训练解剖学分类器
```bash
# 训练解剖学分类器
python train_anatomical_classifier.py
```

训练参数：
- 学习率：1e-4
- 批量大小：64
- 输入尺寸：128×128×128
- 训练轮数：100
- 优化器：Adam
- 损失函数：带权重的交叉熵

#### 2. 训练基础分割模型
```bash
# 训练基础分割模型
python train_skeleton_model10-ronghe.py
```

训练参数：
- 学习率：1e-5
- 批量大小：16
- 输入尺寸：128×128×128
- 训练轮数：200
- 优化器：Adam
- 损失函数：带权重的Dice损失

#### 3. 训练增强型分割模型
```bash
# 训练具有解剖学引导的增强型模型
python training_script_enhanced_airway.py
```

训练参数：
- 学习率：5e-5
- 批量大小：4
- 输入尺寸：128×128×128
- 训练轮数：200
- 优化器：AdamW
- 损失函数：带解剖学权重的加权Dice损失

### 推理
对新数据进行推理的示例：

```python
# 推理代码示例
import torch
from func.AnatomicalClassifier import AirwayAnatomicalClassifier
from func.model_arch_model10_plus_tou import SegAirwayModel
from func.EnhancedAirwaySegmentation import EnhancedAirwaySegmentation

# 加载模型
base_model = SegAirwayModel(in_channels=1, out_channels=2)
classifier = AirwayAnatomicalClassifier(in_channels=1, generation_classes=7, 
                                        structure_classes=7, location_classes=7)
                                        
# 加载预训练权重
base_model.load_state_dict(torch.load('path/to/base_model.pkl')['model_state_dict'])
classifier.load_state_dict(torch.load('path/to/classifier.pkl')['model_state_dict'])

# 创建增强型模型
model = EnhancedAirwaySegmentation(base_model, classifier)
model.load_state_dict(torch.load('path/to/enhanced_model.pkl')['model_state_dict'])
model.eval()

# 预处理输入
# ...

# 执行推理
with torch.no_grad():
    segmentation, cls_output = model(input_tensor)
    
# 后处理结果
# ...
```

## 高级功能

### 解剖学特征提取
`AirwayMetadataGenerator`类生成丰富的解剖学特征，包括：
- 气道代数类型
- 结构复杂度指标
- 空间位置信息
- 分支多样性和分叉检测

使用方法：
```python
generator = AirwayMetadataGenerator(
    raw_data_dir="/path/to/data",
    parsing_dir="/path/to/parsing",
    output_pkl_path="/path/to/output.pkl",
    crop_size=(128, 128, 128),
    stride=(64, 64, 64)
)
generator.generate_metadata(mode="train")
```

### 数据集适配
`PKLtoNPYAdapter`类可以在元数据格式之间转换：
```python
adapter = PKLtoNPYAdapter(pkl_path, npy_base_dir)
data_dict = adapter.get_adapted_dict()
```

### 多阶段训练策略
增强型模型使用3阶段训练策略：
1. **第1阶段（1-50轮）**：在分类器冻结的情况下训练
2. **第2阶段（51-100轮）**：解冻深层分类器特征
3. **第3阶段（101+轮）**：全面微调

## 性能指标

该框架使用以下指标评估性能：
- Dice系数（用于分割准确性）
- 分支检测率（用于气道树完整性）
- 解剖学分类准确率（用于结构识别）
- 泄漏检测（用于防止假阳性）

## 故障排除

### 常见问题

1. **内存溢出错误**
   - 减小批量大小
   - 在多个GPU上使用数据并行训练
   - 考虑梯度累积

2. **训练速度慢**
   - 增加DataLoader中的workers和prefetch_factor
   - 检查磁盘缓存的I/O瓶颈
   - 对于一致的输入大小，启用CUDA benchmark模式

3. **泛化性能差**
   - 使用数据增强（旋转、缩放、噪声）
   - 实现mixup/cutmix策略
   - 尝试不同的权重初始化方法

4. **类别不平衡**
   - 调整损失函数中的类别权重
   - 使用聚焦采样来平衡稀有结构
   - 对表示不足的类别应用有针对性的增强

## 参考文献

该实现基于以下研究：
- 用于医学图像分割的U-Net架构
- 用于通道注意力的Squeeze-and-Excitation网络
- 用于医学图像分析的解剖学引导
- 用于联合分割和分类的多任务学习

## 贡献指南

在为该项目做贡献时：
1. 保持一致的编码风格
2. 为所有函数提供详细的文档字符串
3. 为新功能实现单元测试
4. 使用能反映解剖学概念的有意义变量名
5. 考虑计算效率（内存使用和处理时间）


我：请帮我们完成这个项目的技术指南readme文件，包含输入输出，数据结构，运行指南等。由于该项目设计多个阶段步骤，可能需要你全面详细考虑。
claude: # Airway Segmentation Framework Technical Guide

This guide provides detailed information about the enhanced airway segmentation framework, which combines anatomical classification and advanced segmentation techniques for improved airway tree detection and segmentation in CT images.

## Project Overview

This project implements a multi-stage airway segmentation framework that leverages anatomical knowledge to improve segmentation accuracy. The framework consists of three main components:

1. **Anatomical Classification**: Classifies airway structures based on generation type, structure type, and location
2. **Base Segmentation Model**: Performs initial airway segmentation
3. **Enhanced Segmentation**: Integrates anatomical features into the segmentation process for improved results

## Input/Output Specification

### Input
- CT image volumes (3D) in NIFTI (.nii.gz) format
- Binary segmentation masks (for training)
- Parsed airway tree metadata (optional, for advanced features)

### Output
- Binary segmentation of airways
- Anatomical classification results (if using the full pipeline):
  - Generation type (trachea, main bronchi, lobar bronchi, etc.)
  - Structure type (tubular, bifurcation, complex branches)
  - Location type (upper/middle/lower lung, left/right lung)
- Feature maps and attention maps (for visualization purposes)

## Data Structure

### Directory Structure
```
/
├── checkpoint/                  # Model checkpoints
├── dataset_info/                # Metadata and dataset information
├── func/                        # Core functionality modules
│   ├── AnatomicalClassifier.py  # Airway classification model
│   ├── AnatomicalFusionModules.py # Feature fusion modules
│   ├── EnhancedAirwaySegmentation.py # Enhanced segmentation model
│   ├── loss_func.py            # Loss functions
│   ├── load_dataset.py         # Dataset loading utilities
│   └── model_arch_model10_plus_tou.py # Base segmentation architecture
├── runs/                        # Training logs for base models
├── runs_anatomical_classifier/  # Training logs for classifier
├── runs_enhanced_airway/        # Training logs for enhanced models
└── test_data/                   # Test data for evaluation
```

### Input Data Format
The framework uses multiple data formats:

1. **Raw Data**: CT scans in NIFTI (.nii.gz) format
2. **Preprocessed Data**: Cropped and normalized patches (.npy format)
3. **Metadata**: Anatomical features stored in pickle (.pkl) format
4. **Parsing Data**: Airway branch information in NIFTI and NumPy formats

### Data Dictionary Format
The framework uses Python dictionaries to organize data samples. Each entry contains:

- `image_path`: Path to the CT image
- `label_path`: Path to the segmentation mask
- `position`: (z, y, x) coordinates of the patch in the original volume
- `image_size`: Size of the original image
- `crop_size`: Size of the cropped patch
- `stride`: Stride used for cropping
- Anatomical features:
  - `generation_type`: Type of airway generation (0-6)
  - `structure_type`: Type of airway structure (0-6)
  - `location_type`: Spatial location (0-6)
  - `airway_pixel_num`: Number of airway pixels
  - `airway_density`: Density of airways in the patch
  - `complexity`: Structural complexity metric
  - Additional metrics for airway characteristics

## Model Architecture

### Anatomical Classifier
The anatomical classifier (`AirwayAnatomicalClassifier`) is a 3D CNN that predicts:
- Generation type (7 classes): none, trachea, main bronchi, lobar bronchi, segmental bronchi, small bronchi, fine bronchi
- Structure type (7 classes): none, tubular, oval, bifurcation, complex branches (3 levels)
- Location type (7 classes): none, upper lung, middle lung, lower lung, left lung, right lung, tracheal region

### Base Segmentation Model
The base model (`SegAirwayModel`) is a U-Net style 3D CNN with:
- 4 encoder blocks with channel dimensions [32, 64, 128, 256]
- 3 decoder blocks with channel dimensions [128, 64, 32]
- Skip connections between corresponding encoder and decoder layers
- Attention modules and dilated convolutions for enhancing feature extraction

### Enhanced Segmentation Model
The enhanced model (`EnhancedAirwaySegmentation`) integrates the anatomical classifier with the segmentation network using:
- `GenerationGuidanceModule`: For deep decoder layers to enhance fine airway detection
- `StructureGuidanceModule`: For middle decoder layers to enhance bifurcation modeling
- `LocationGuidanceModule`: For encoder layers to guide spatial awareness

## Running Instructions

### Environment Setup
```bash
# Create and activate conda environment
conda create -n airway_seg python=3.8
conda activate airway_seg

# Install required packages
pip install torch torchvision
pip install nibabel SimpleITK scikit-image tqdm tensorboard
pip install pandas numpy scipy matplotlib
```

### Data Preparation
1. **Generate Metadata**:
```bash
# Generate airway metadata based on raw data and parsing results
python AirwayMetadataGenerator.py
```

2. **Crop Images** (if needed):
```bash
# Crop raw CT images into patches
python BAS_pre_crop_image.py
```

### Model Training

#### 1. Train Anatomical Classifier
```bash
# Train the anatomical classifier
python train_anatomical_classifier.py
```

Training parameters:
- Learning rate: 1e-4
- Batch size: 64
- Input size: 128×128×128
- Epochs: 100
- Optimizer: Adam
- Loss: Weighted Cross Entropy

#### 2. Train Base Segmentation Model
```bash
# Train the base segmentation model
python train_skeleton_model10-ronghe.py
```

Training parameters:
- Learning rate: 1e-5
- Batch size: 16
- Input size: 128×128×128
- Epochs: 200
- Optimizer: Adam
- Loss: Weighted Dice Loss

#### 3. Train Enhanced Segmentation Model
```bash
# Train the enhanced model with anatomical guidance
python training_script_enhanced_airway.py
```

Training parameters:
- Learning rate: 5e-5
- Batch size: 4
- Input size: 128×128×128
- Epochs: 200
- Optimizer: AdamW
- Loss: Weighted Dice Loss with anatomical weighting

### Inference
To run inference on new data:

```python
# Example inference code
import torch
from func.AnatomicalClassifier import AirwayAnatomicalClassifier
from func.model_arch_model10_plus_tou import SegAirwayModel
from func.EnhancedAirwaySegmentation import EnhancedAirwaySegmentation

# Load models
base_model = SegAirwayModel(in_channels=1, out_channels=2)
classifier = AirwayAnatomicalClassifier(in_channels=1, generation_classes=7, 
                                        structure_classes=7, location_classes=7)
                                        
# Load pretrained weights
base_model.load_state_dict(torch.load('path/to/base_model.pkl')['model_state_dict'])
classifier.load_state_dict(torch.load('path/to/classifier.pkl')['model_state_dict'])

# Create enhanced model
model = EnhancedAirwaySegmentation(base_model, classifier)
model.load_state_dict(torch.load('path/to/enhanced_model.pkl')['model_state_dict'])
model.eval()

# Preprocess input
# ...

# Run inference
with torch.no_grad():
    segmentation, cls_output = model(input_tensor)
    
# Post-process results
# ...
```

## Advanced Features

### Anatomical Feature Extraction
The `AirwayMetadataGenerator` class generates rich anatomical features including:
- Airway generation types
- Structure complexity metrics
- Spatial location information
- Branch diversity and bifurcation detection

Usage:
```python
generator = AirwayMetadataGenerator(
    raw_data_dir="/path/to/data",
    parsing_dir="/path/to/parsing",
    output_pkl_path="/path/to/output.pkl",
    crop_size=(128, 128, 128),
    stride=(64, 64, 64)
)
generator.generate_metadata(mode="train")
```

### Dataset Adaptation
The `PKLtoNPYAdapter` class converts between metadata formats:
```python
adapter = PKLtoNPYAdapter(pkl_path, npy_base_dir)
data_dict = adapter.get_adapted_dict()
```

### Multi-stage Training Strategy
The enhanced model uses a 3-stage training strategy:
1. **Stage 1 (1-50 epochs)**: Train with frozen classifier
2. **Stage 2 (51-100 epochs)**: Unfreeze deep classifier features
3. **Stage 3 (101+ epochs)**: Full fine-tuning

## Performance Metrics

The framework evaluates performance using:
- Dice coefficient (for segmentation accuracy)
- Branch detection rate (for airway tree completeness)
- Anatomical classification accuracy (for structure identification)
- Leakage detection (for false positive prevention)

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Use data parallel training across multiple GPUs
   - Consider gradient accumulation

2. **Slow Training**
   - Increase workers and prefetch factor in DataLoader
   - Check for I/O bottlenecks with disk caching
   - Enable CUDA benchmark mode for consistent input sizes

3. **Poor Generalization**
   - Use data augmentation (rotation, scaling, noise)
   - Implement mixup/cutmix strategies
   - Try different weight initialization methods

4. **Class Imbalance**
   - Adjust class weights in loss functions
   - Use focused sampling to balance rare structures
   - Apply targeted augmentation to underrepresented classes

## References

The implementation is based on the following research:
- U-Net architecture for medical image segmentation
- Squeeze-and-Excitation networks for channel attention
- Anatomical guidance for medical image analysis
- Multi-task learning for joint segmentation and classification

## Contributing Guidelines

When contributing to this project:
1. Maintain consistent coding style
2. Document all functions with detailed docstrings
3. Implement unit tests for new functionality
4. Use meaningful variable names that reflect anatomical concepts
5. Keep computational efficiency in mind (memory usage and processing time)
