# 气道解剖学解耦注意力可视化工具

这个工具集提供了全面的可视化和分析功能，用于理解气道解剖学分类器（AirwayAnatomicalClassifier）中的注意力机制。它可以帮助研究人员和开发者直观地理解模型如何关注不同的解剖属性（气道代数、结构类型、空间位置）特征，从而验证解耦效果。

## 功能特点

- **3D Grad-CAM实现**：扩展到3D医学影像数据的Grad-CAM技术
- **多平面重建(MPR)可视化**：在轴状、冠状和矢状平面上查看注意力分布
- **最大强度投影(MIP)**：提供压缩的3D视图，便于总览
- **解剖属性对比**：比较不同解剖属性（代数/结构/位置）的注意力模式
- **特征图可视化**：探索网络不同层次的特征表示
- **量化指标**：生成注意力分布的统计指标

## 安装要求

- Python 3.7+
- PyTorch 1.7+
- SimpleITK
- Nibabel
- Matplotlib
- NumPy
- tqdm

## 使用方法

### 基本示例

```bash
python run_anatomical_attention.py \
    --pkl_path ./dataset_info/airway_metadata_train_anatomical.pkl \
    --npy_base_dir /path/to/npy/files \
    --model_path ./runs_anatomical_classifier/airway_anatomical_classifier_best.pkl \
    --output_dir ./airway_attention_results \
    --mode single \
    --case_ids BAStest_case1_0
```

### 批量分析

```bash
python run_anatomical_attention.py \
    --pkl_path ./dataset_info/airway_metadata_train_anatomical.pkl \
    --npy_base_dir /path/to/npy/files \
    --model_path ./runs_anatomical_classifier/airway_anatomical_classifier_best.pkl \
    --output_dir ./airway_attention_results \
    --mode batch \
    --num_samples 10
```

### 比较多个样本

```bash
python run_anatomical_attention.py \
    --pkl_path ./dataset_info/airway_metadata_train_anatomical.pkl \
    --npy_base_dir /path/to/npy/files \
    --model_path ./runs_anatomical_classifier/airway_anatomical_classifier_best.pkl \
    --output_dir ./airway_attention_results \
    --mode compare \
    --case_ids BAStest_case1_0 BAStest_case2_3 BAStest_case3_5
```

## 代码结构

- `AirwayAnatomicalGradCAM.py`: 3D Grad-CAM实现，用于生成注意力图
- `AnatomicalAttentionVisualizer.py`: 基础可视化工具，包括MPR和MIP可视化
- `AnatomicalAttentionVisualizer_Advanced.py`: 高级可视化和分析功能
- `FeatureMapVisualizer.py`: 网络特征图可视化
- `run_anatomical_attention.py`: 主运行脚本

## 工具集详细说明

### 1. AirwayAnatomicalGradCAM 

针对3D医学影像优化的Grad-CAM实现：

```python
from AirwayAnatomicalGradCAM import AirwayAnatomicalGradCAM

# 创建实例
gradcam = AirwayAnatomicalGradCAM(model, target_layer, task='generation')

# 生成注意力图
attention_map = gradcam.generate_cam(input_tensor, target_class=3)

# 获取预测类别和概率
pred_class, probabilities = gradcam.get_task_prediction(input_tensor)

# 计算多个类别的激活图
activation_maps = gradcam.compute_multi_class_activation_maps(input_tensor, top_k=3)
```

### 2. AnatomicalAttentionVisualizer

用于可视化注意力图的多平面、MIP和3D表示：

```python
from AnatomicalAttentionVisualizer import AnatomicalAttentionVisualizer

# 创建可视化器
visualizer = AnatomicalAttentionVisualizer(output_dir='./attention_maps')

# 多平面重建可视化
mpr_path = visualizer.visualize_mpr_slices(
    ct_volume, attention_map, case_id, 'generation', target_class=3
)

# 3D可视化 (MIP + NIfTI导出)
mip_path, volume_path = visualizer.create_3d_visualization(
    ct_volume, attention_map, case_id, 'generation', target_class=3
)
```

### 3. AnatomicalAttentionAdvanced

高级可视化和比较功能：

```python
from AnatomicalAttentionVisualizer import AnatomicalAttentionVisualizer
from AnatomicalAttentionVisualizer_Advanced import AnatomicalAttentionAdvanced

# 创建基础和高级可视化器
visualizer = AnatomicalAttentionVisualizer(output_dir='./attention_maps')
adv_vis = AnatomicalAttentionAdvanced(visualizer)

# 比较不同解剖属性的注意力图
comparison_path = adv_vis.compare_attention_maps(
    ct_volume,
    {'generation': gen_attention, 'structure': struct_attention, 'location': loc_attention},
    case_id
)

# 创建两个注意力图之间的差异图
diff_path = adv_vis.create_attention_difference_map(
    gen_attention, struct_attention, case_id, 'generation', 'structure'
)

# 在原始CT上叠加注意力图与分割标签对比
overlay_path = adv_vis.overlay_on_original_slice(
    ct_volume, attention_map, label_map, case_id, 'generation', 
    target_class=3, axis=0
)

# 可视化类别激活强度
probs_path = adv_vis.visualize_class_activation_strength(
    {'generation': gen_probs, 'structure': struct_probs, 'location': loc_probs},
    case_id
)
```

### 4. FeatureMapVisualizer

用于可视化网络内部特征表示：

```python
from FeatureMapVisualizer import AirwayFeatureMapVisualizer

# 创建特征图可视化器
feature_viz = AirwayFeatureMapVisualizer(model, output_dir='./feature_maps')

# 注册特定层的钩子
layer_names = ['encoder.5', 'attention_blocks.1', 'guidance_head']
feature_viz.register_hooks(layer_names)

# 提取特征
features = feature_viz.extract_features(input_tensor)

# 可视化特征图
feat_path = feature_viz.visualize_feature_map(
    features['encoder.5'], 'encoder.5', case_id, slice_idx=64
)

# 可视化引导图
guidance_path = feature_viz.visualize_guidance_map(
    features['guidance_head'], case_id
)

# 运行完整可视化流程
results = feature_viz.run_complete_visualization(input_tensor, case_id)
```

## 可视化结果示例

### 1. 气道代数注意力图

![气道代数注意力图](example_figures/generation_attention.png)

代数注意力图显示了气道解剖学分类器如何区分不同代数级别的气道。模型通常关注：
- 气管：管径大小和中心位置
- 主支气管：分叉角度和管径变化
- 细支气管：微小的管腔结构和分支模式

### 2. 气道结构注意力图

![气道结构注意力图](example_figures/structure_attention.png)

结构注意力图显示了分类器如何识别不同形态结构：
- 直管道：线性结构和均匀直径
- 分叉点：Y形或T形交叉区域
- 复杂区域：多个分支聚集的复杂网络区域

### 3. 空间位置注意力图

![气道位置注意力图](example_figures/location_attention.png)

位置注意力图显示了分类器如何确定气道在肺部的位置：
- 上/中/下肺：基于纵向位置的注意力分布
- 左/右肺：基于横向位置的注意力分布

### 4. 注意力对比图

![注意力对比图](example_figures/attention_comparison.png)

对比图展示了不同解剖属性之间注意力的区别，帮助理解解耦效果。

### 5. 注意力差异图

![注意力差异图](example_figures/attention_difference.png)

红蓝色差异图突出显示了不同属性之间注意力的差异区域，红色表示第一个属性更强，蓝色表示第二个属性更强。

## 使用场景

本工具适用于以下场景：

1. **模型解释**：深入了解气道解剖学分类器如何关注不同的解剖特征
2. **解耦验证**：验证不同解剖属性分类任务是否真正关注不同特征
3. **模型改进**：发现可能的注意力盲点，指导模型改进
4. **异常分析**：识别分类错误的样本中注意力异常区域
5. **医学教育**：直观展示AI如何识别不同的气道解剖结构

## 高级用法

### 自定义注意力层

您可以通过`--target_layer`参数指定不同的目标层来生成注意力图：

```bash
python run_anatomical_attention.py \
    --target_layer attention_blocks.1 \
    --model_path ./model_weights.pkl \
    --output_dir ./results
```

### 保存NIFTI格式注意力图

添加`--save_nifti`参数以保存完整的3D注意力体积为NIFTI格式：

```bash
python run_anatomical_attention.py \
    --save_nifti \
    --model_path ./model_weights.pkl \
    --output_dir ./results
```

### 生成综合报告

运行批量模式会自动生成包含统计信息、性能指标和可视化示例的报告：

```bash
python run_anatomical_attention.py \
    --mode batch \
    --num_samples 20 \
    --model_path ./model_weights.pkl \
    --output_dir ./results
```

## 贡献与许可

本工具是基于AHSegNet项目开发的配套可视化系统，用于理解和验证气道解剖学解耦编码器的工作原理。欢迎贡献和改进建议。
