import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AirwayAnatomicalGradCAM:
    """
    GradCAM实现，专为气道解剖学分类器设计，
    用于可视化各个解剖属性(generation/structure/location)的注意力图
    """
    def __init__(self, model, target_layer, task='generation'):
        """
        Args:
            model: AirwayAnatomicalClassifier模型
            target_layer: 目标层，如model.encoder[5]或model.attention_blocks[1]
            task: 目标任务 - 'generation', 'structure', 或 'location'
        """
        self.model = model
        self.target_layer = target_layer
        self.task = task
        
        # 存储前向传播的特征和梯度
        self.activations = None
        self.gradients = None
        self.hooks = []
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向钩子来捕获特征和梯度"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # 注册前向钩子
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        # 注册反向钩子
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        
        # 保存钩子句柄以便后续移除
        self.hooks.append(forward_handle)
        self.hooks.append(backward_handle)
    
    def remove_hooks(self):
        """移除所有注册的钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        生成类激活图
        
        Args:
            input_tensor: 输入CT体积 [B, C, D, H, W]
            target_class: 目标类别索引，如果为None则使用预测类别
            
        Returns:
            cam: 归一化的注意力图 [B, 1, D, H, W]
        """
        # 设置模型为评估模式
        self.model.eval()
        
        # 启用梯度计算
        with torch.set_grad_enabled(True):
            # 前向传播
            outputs = self.model(input_tensor)
            
            # 获取任务对应的输出
            if self.task == 'generation':
                logits = outputs['generation']
            elif self.task == 'structure':
                logits = outputs['structure']
            elif self.task == 'location':
                logits = outputs['location']
            else:
                raise ValueError(f"未知任务: {self.task}")
            
            # 如果未指定目标类别，使用预测类别
            if target_class is None:
                target_class = torch.argmax(logits, dim=1).item()
            
            # 创建one-hot编码向量
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_class] = 1
            
            # 反向传播
            self.model.zero_grad()
            logits.backward(gradient=one_hot, retain_graph=True)
            
            # 使用全局平均池化获取权重
            weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
            
            # 生成加权激活图
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            
            # 应用ReLU激活
            cam = F.relu(cam)
            
            # 归一化CAM
            if torch.max(cam) > 0:
                cam = cam - torch.min(cam)
                cam = cam / (torch.max(cam) + 1e-8)
            
            return cam
    
    def get_feature_maps(self):
        """获取目标层的特征图"""
        return self.activations
    
    def get_task_prediction(self, input_tensor):
        """
        获取指定任务的预测结果
        
        Args:
            input_tensor: 输入CT体积
            
        Returns:
            tuple: (predicted_class, class_probabilities)
        """
        # 设置模型为评估模式
        self.model.eval()
        
        # 前向传播，不计算梯度
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # 获取任务对应的输出
            if self.task == 'generation':
                logits = outputs['generation']
            elif self.task == 'structure':
                logits = outputs['structure']
            elif self.task == 'location':
                logits = outputs['location']
            else:
                raise ValueError(f"未知任务: {self.task}")
            
            # 计算概率
            probabilities = F.softmax(logits, dim=1)
            
            # 获取预测类别
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            return predicted_class, probabilities
    
    def compute_multi_class_activation_maps(self, input_tensor, top_k=3):
        """
        计算多个类别的激活图
        
        Args:
            input_tensor: 输入CT体积
            top_k: 返回的顶部类别数量
            
        Returns:
            dict: 类别索引到激活图的映射
        """
        # 获取预测概率
        _, probabilities = self.get_task_prediction(input_tensor)
        
        # 获取概率最高的top_k类别
        probs = probabilities[0].cpu().numpy()
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        # 为每个类别计算激活图
        activation_maps = {}
        for class_idx in top_indices:
            activation_maps[int(class_idx)] = self.generate_cam(input_tensor, target_class=int(class_idx))
            
        return activation_maps
