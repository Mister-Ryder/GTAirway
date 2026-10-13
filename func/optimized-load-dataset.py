# dataset load and transform
import numpy as np
import skimage.io as io
import random
import h5py
import os
import SimpleITK as sitk
import time
import threading
import queue
from functools import lru_cache
from collections import OrderedDict
import gc

from torch.utils.data import Dataset
from torch import from_numpy as from_numpy
from torchvision import transforms
import torch
import torchio as tio

class Random3DCrop_np(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)), 'Attention: random 3D crop output size: an int or a tuple (length:3)'
        if isinstance(output_size, int):
            self.output_size=(output_size, output_size, output_size)
        else:
            assert len(output_size)==3, 'Attention: random 3D crop output size: a tuple (length:3)'
            self.output_size=output_size
        
    def random_crop_start_point(self, input_size):
        assert len(input_size)==3, 'Attention: random 3D crop output size: a tuple (length:3)'
        d, h, w=input_size
        d_new, h_new, w_new=self.output_size
        
        d_new = min(d, d_new)
        h_new = min(h, h_new)
        w_new = min(w, w_new)
        
        assert (d>=d_new and h>=h_new and w>=w_new), "Attention: input size should >= crop size; now, input_size is "+str((d,h,w))+", while output_size is "+str((d_new, h_new, w_new))
        
        d_start=np.random.randint(0, d-d_new+1)
        h_start=np.random.randint(0, h-h_new+1)
        w_start=np.random.randint(0, w-w_new+1)
        
        return d_start, h_start, w_start
    
    def __call__(self, img_3d, start_points=None):
        img_3d=np.array(img_3d)
        
        d, h, w=img_3d.shape
        d_new, h_new, w_new=self.output_size
        
        if start_points == None:
            start_points = self.random_crop_start_point(img_3d.shape)
        
        d_start, h_start, w_start = start_points
        d_end = min(d_start+d_new, d)
        h_end = min(h_start+h_new, h)
        w_end = min(w_start+w_new, w)
        
        crop=img_3d[d_start:d_end, h_start:h_end, w_start:w_end]
        
        return crop

class Normalization_np(object):
    def __init__(self, windowMin, windowMax):
        self.name = 'ManualNormalization'
        assert isinstance(windowMax, (int,float))
        assert isinstance(windowMin, (int,float))
        self.windowMax = windowMax
        self.windowMin = windowMin
    
    def __call__(self, img_3d):
        img_3d_norm = np.clip(img_3d, self.windowMin, self.windowMax)
        img_3d_norm-=np.min(img_3d_norm)
        max_99_val=np.percentile(img_3d_norm, 99)
        if max_99_val>0:
            img_3d_norm = img_3d_norm/max_99_val*255
        return img_3d_norm

class TwoLevelCache:
    """高效的两级缓存系统，使用内存缓存和磁盘缓存"""
    def __init__(self, memory_cache_size=10, disk_cache_dir=None):
        self.memory_cache = OrderedDict()
        self.memory_cache_size = memory_cache_size
        self.disk_cache_dir = disk_cache_dir
        
        # 创建磁盘缓存目录
        if disk_cache_dir and not os.path.exists(disk_cache_dir):
            os.makedirs(disk_cache_dir)
        
        # 用于预取的队列和线程
        self.prefetch_queue = queue.Queue(maxsize=20)
        self.running = True
        
        # 启动预取线程
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def _get_disk_cache_path(self, path):
        """根据原始路径创建唯一的磁盘缓存路径"""
        if not self.disk_cache_dir:
            return None
        
        # 创建基于原始路径的唯一文件名
        filename = os.path.basename(path)
        cache_name = f"{hash(path)}_{filename}.npy"
        return os.path.join(self.disk_cache_dir, cache_name)
    
    def _read_nifti(self, path):
        """使用优化设置读取NIFTI文件"""
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(path)
            reader.SetLoadPrivateTags(False)  # 不加载私有标签
            reader.SetLoadSequences(False)    # 不加载序列
            reader.ReadImageInformation()     # 先读取图像信息
            image = reader.Execute()
            return sitk.GetArrayFromImage(image)
        except Exception as e:
            print(f"Error reading NIFTI file {path}: {e}")
            # 返回一个空数组作为后备
            return np.array([])
    
    def get(self, path):
        """获取数据，优先从缓存获取"""
        # 检查内存缓存
        if path in self.memory_cache:
            # 更新缓存顺序（将其移到最新）
            self.memory_cache.move_to_end(path)
            return self.memory_cache[path]
        
        # 检查磁盘缓存
        disk_path = self._get_disk_cache_path(path)
        if disk_path and os.path.exists(disk_path):
            try:
                # 使用内存映射读取大型数组
                array = np.load(disk_path, mmap_mode='r+')
                
                # 添加到内存缓存
                if len(self.memory_cache) >= self.memory_cache_size:
                    # 删除最早使用的项目
                    self.memory_cache.popitem(last=False)
                
                self.memory_cache[path] = array
                return array
            except Exception as e:
                print(f"Error loading from disk cache {disk_path}: {e}")
                # 如果磁盘缓存出错，删除可能损坏的文件
                if os.path.exists(disk_path):
                    try:
                        os.remove(disk_path)
                    except:
                        pass
        
        # 直接从NIFTI加载
        array = self._read_nifti(path)
        
        # 保存到磁盘缓存
        if disk_path:
            try:
                np.save(disk_path, array)
            except Exception as e:
                print(f"Error saving to disk cache {disk_path}: {e}")
        
        # 添加到内存缓存
        if len(self.memory_cache) >= self.memory_cache_size:
            self.memory_cache.popitem(last=False)
        
        self.memory_cache[path] = array
        return array
    
    def prefetch(self, path):
        """将路径加入预取队列"""
        if path not in self.memory_cache:
            try:
                self.prefetch_queue.put(path, block=False)
            except queue.Full:
                pass  # 队列已满，忽略此预取请求
    
    def _prefetch_worker(self):
        """预取工作线程"""
        while self.running:
            try:
                path = self.prefetch_queue.get(timeout=1)
                if path not in self.memory_cache:
                    self.get(path)  # 加载到缓存
                self.prefetch_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in prefetch worker: {e}")
    
    def clear(self):
        """清空缓存"""
        self.memory_cache.clear()

class SmartCrop:
    """更智能的裁剪策略，偏向于包含气道区域的裁剪"""
    def __init__(self, output_size, airway_bias=0.8):
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size, output_size)
        self.airway_bias = airway_bias  # 偏向气道区域的概率
        self.random_crop = Random3DCrop_np(output_size)
    
    def __call__(self, raw_img, label_img=None):
        """执行智能裁剪，偏向于选择包含气道的区域"""
        if label_img is None or random.random() > self.airway_bias:
            # 使用完全随机裁剪
            return self.random_crop.random_crop_start_point(raw_img.shape)
        
        # 查找包含气道的区域
        airway_positions = np.argwhere(label_img > 0)
        
        if len(airway_positions) == 0:
            # 没有气道，使用随机裁剪
            return self.random_crop.random_crop_start_point(raw_img.shape)
        
        # 随机选择一个气道点作为中心
        center_idx = random.randint(0, len(airway_positions) - 1)
        center = airway_positions[center_idx]
        
        # 计算裁剪起始位置，使选中的气道像素位于裁剪区域的随机位置
        d_offset = random.randint(0, self.output_size[0] - 1)
        h_offset = random.randint(0, self.output_size[1] - 1)
        w_offset = random.randint(0, self.output_size[2] - 1)
        
        d_start = max(0, center[0] - d_offset)
        h_start = max(0, center[1] - h_offset)
        w_start = max(0, center[2] - w_offset)
        
        # 确保不会超出边界
        d, h, w = raw_img.shape
        d_start = min(d_start, d - self.output_size[0])
        h_start = min(h_start, h - self.output_size[1])
        w_start = min(w_start, w - self.output_size[2])
        
        return (d_start, h_start, w_start)

class airway_dataset(Dataset):
    def __init__(self, data_dict, num_of_samples=None, cache_dir=None):
        # 初始化数据字典
        self.data_dict = data_dict
        
        # 选择样本
        if num_of_samples is not None:
            num_of_samples = min(len(data_dict), num_of_samples)
            chosen_names = np.random.choice(np.array(list(data_dict)), num_of_samples, replace=False)
        else:
            chosen_names = np.array(list(data_dict))
        self.name_list = chosen_names
        
        # 初始化参数
        self.para = {}
        self.set_para()
        
        # 设置缓存目录
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".airway_cache")
        self.cache_dir = cache_dir
        
        # 初始化两级缓存
        self.nifti_cache = TwoLevelCache(
            memory_cache_size=20,  # 内存中保留20个体积
            disk_cache_dir=cache_dir  # 磁盘缓存目录
        )
        
        # 初始化智能裁剪器
        self.smart_crop = SmartCrop(output_size=(128, 128, 128), airway_bias=0.8)
        
        # 预取一些数据
        self._prefetch_initial_data()
    
    def _prefetch_initial_data(self):
        """预取前几个数据的体积"""
        num_to_prefetch = min(5, len(self.name_list))
        for i in range(num_to_prefetch):
            name = self.name_list[i]
            item_dict = self.data_dict[name]
            
            # 启动预取
            self.nifti_cache.prefetch(item_dict["image_path"])
            self.nifti_cache.prefetch(item_dict["label_path"])
            if "skeleton_path" in item_dict and item_dict["skeleton_path"]:
                self.nifti_cache.prefetch(item_dict["skeleton_path"])
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        # 预取下一个可能的索引
        next_idx = (idx + 1) % len(self.name_list)
        self._prefetch_next_volume(next_idx)
        
        # 获取当前样本
        return self.get(
            idx,
            file_format=self.para["file_format"],
            crop_size=self.para["crop_size"],
            windowMin=self.para["windowMin"],
            windowMax=self.para["windowMax"],
            need_tensor_output=self.para["need_tensor_output"],
            need_transform=self.para["need_transform"]
        )
    
    def _prefetch_next_volume(self, idx):
        """预取下一个索引的体积数据"""
        name = self.name_list[idx]
        item_dict = self.data_dict[name]
        
        # 将路径加入预取队列
        threading.Thread(
            target=self.nifti_cache.prefetch,
            args=(item_dict["image_path"],),
            daemon=True
        ).start()
        
        threading.Thread(
            target=self.nifti_cache.prefetch,
            args=(item_dict["label_path"],),
            daemon=True
        ).start()
        
        if "skeleton_path" in item_dict and item_dict["skeleton_path"]:
            threading.Thread(
                target=self.nifti_cache.prefetch,
                args=(item_dict["skeleton_path"],),
                daemon=True
            ).start()

    def set_para(self, file_format='.nii.gz', crop_size=32, windowMin=-1000, windowMax=150,
                need_tensor_output=True, need_transform=True):
        """设置数据加载参数"""
        self.para["file_format"] = file_format
        self.para["crop_size"] = crop_size
        self.para["windowMin"] = windowMin
        self.para["windowMax"] = windowMax
        self.para["need_tensor_output"] = need_tensor_output
        self.para["need_transform"] = need_transform
        
        # 更新智能裁剪器的输出大小
        self.smart_crop = SmartCrop(output_size=crop_size, airway_bias=0.8)
    
    def get_volume_crop(self, path, position, crop_size):
        """从完整体积获取特定位置的裁剪"""
        if not path:
            return np.zeros(crop_size)
            
        # 从缓存获取完整体积
        try:
            start_time = time.time()
            volume = self.nifti_cache.get(path)
            load_time = time.time() - start_time
            
            if load_time > 0.5:  # 如果加载时间超过0.5秒，输出警告
                print(f"Warning: Slow loading for {path}: {load_time:.2f}s")
            
            # 检查体积是否为空
            if volume.size == 0:
                print(f"Warning: Empty volume from {path}")
                return np.zeros(crop_size)
            
            # 提取裁剪区域
            z_start, y_start, x_start = position
            z_end = min(z_start + crop_size[0], volume.shape[0])
            y_end = min(y_start + crop_size[1], volume.shape[1])
            x_end = min(x_start + crop_size[2], volume.shape[2])
            
            # 返回裁剪区域
            return volume[z_start:z_end, y_start:y_end, x_start:x_end]
        except Exception as e:
            print(f"Error in get_volume_crop for {path}: {e}")
            return np.zeros(crop_size)

    def get(self, idx, file_format='.nii.gz', crop_size=32, windowMin=-1000, windowMax=150,
           need_tensor_output=True, need_transform=True):
        """获取指定索引的样本"""
        start_time = time.time()
        
        random3dcrop = Random3DCrop_np(crop_size)
        normalization = Normalization_np(windowMin, windowMax)

        name = self.name_list[idx]
        item_dict = self.data_dict[name]
        
        try:
            # 根据文件格式选择不同的加载方式
            if file_format == '.npy':
                # 支持旧的行为 - 直接从.npy文件加载
                raw_img = np.load(item_dict["image"])
                label_img = np.load(item_dict["label"])
                has_skeleton = "skeleton" in item_dict and item_dict["skeleton"]
                if has_skeleton:
                    skeleton_img = np.load(item_dict["skeleton"])
                    gaussian_img = np.load(item_dict["gaussian"])
                    gaussian_img2 = np.load(item_dict["gaussian"].replace('gaussian/', 'gaussian2/'))
            else:
                # 新的行为 - 从NIFTI文件加载并裁剪
                position = item_dict.get('position', (0, 0, 0))
                crop_size_initial = item_dict.get('crop_size', (128, 128, 128))
                
                # 从缓存获取预裁剪
                raw_img = self.get_volume_crop(item_dict["image_path"], position, crop_size_initial)
                label_img = self.get_volume_crop(item_dict["label_path"], position, crop_size_initial)
                
                # 检查是否有骨架和高斯模糊数据
                has_skeleton = "skeleton_path" in item_dict and item_dict["skeleton_path"]
                if has_skeleton:
                    skeleton_img = self.get_volume_crop(item_dict["skeleton_path"], position, crop_size_initial)
                    
                    # 处理高斯路径
                    gaussian_path = item_dict.get("gaussian_path", 
                                                 item_dict["skeleton_path"].replace('skeleton', 'gaussian'))
                    gaussian_img = self.get_volume_crop(gaussian_path, position, crop_size_initial)
                    
                    gaussian_path2 = item_dict.get("gaussian_path2", 
                                                 gaussian_path.replace('gaussian/', 'gaussian2/'))
                    gaussian_img2 = self.get_volume_crop(gaussian_path2, position, crop_size_initial)
            
            # 使用智能裁剪选择起始点
            start_points = self.smart_crop(raw_img, label_img)
            
            # 应用随机裁剪
            raw_img_crop = random3dcrop(np.array(raw_img, float), start_points=start_points)
            label_img_crop = random3dcrop(np.array(label_img, float), start_points=start_points)
            
            if has_skeleton:
                skeleton_img_crop = random3dcrop(np.array(skeleton_img, float), start_points=start_points)
                gaussian_img_crop = random3dcrop(np.array(gaussian_img, float), start_points=start_points)
                gaussian_img_crop2 = random3dcrop(np.array(gaussian_img2, float), start_points=start_points)
            
            # 应用归一化
            raw_img_crop = normalization(raw_img_crop)
            
            # 扩展维度
            raw_img_crop = np.expand_dims(raw_img_crop, axis=0)
            label_img_crop = np.expand_dims(label_img_crop, axis=0)
            
            # 创建输出字典
            if has_skeleton:
                skeleton_img_crop = np.expand_dims(skeleton_img_crop, axis=0)
                gaussian_img_crop = np.expand_dims(gaussian_img_crop, axis=0)
                gaussian_img_crop2 = np.expand_dims(gaussian_img_crop2, axis=0)
                
                output = {
                    "image": raw_img_crop, 
                    "label": label_img_crop, 
                    "skeleton": skeleton_img_crop,
                    "gaussian": gaussian_img_crop, 
                    "gaussian2": gaussian_img_crop2
                }
            else:
                output = {"image": raw_img_crop, "label": label_img_crop}
            
            # 转换为张量并应用变换
            if need_tensor_output:
                output = self.to_tensor(output)
                if need_transform:
                    output = self.transform_the_tensor(output, prob=0.5)
            
            # 记录耗时
            load_time = time.time() - start_time
            if load_time > 1.0:  # 超过1秒输出警告
                print(f"Warning: Sample {idx} loading took {load_time:.2f}s")
            
            return output
            
        except Exception as e:
            print(f"Error processing sample {name} (index {idx}): {e}")
            # 返回一个空占位符
            empty_size = (1, *crop_size) if isinstance(crop_size, tuple) else (1, crop_size, crop_size, crop_size)
            empty_output = {
                "image": np.zeros(empty_size),
                "label": np.zeros(empty_size)
            }
            if need_tensor_output:
                empty_output = self.to_tensor(empty_output)
            return empty_output

    def to_tensor(self, images):
        """将numpy数组转换为pytorch张量"""
        for item in images.keys():
            images[item] = from_numpy(images[item]).float()
        return images
    
    def transform_the_tensor(self, image_tensors, prob=0.5):
        """应用数据增强变换"""
        dict_imgs_tio = {}
        
        for item in image_tensors.keys():
            dict_imgs_tio[item] = tio.ScalarImage(tensor=image_tensors[item])
        
        subject_all_imgs = tio.Subject(dict_imgs_tio)
        
        # 应用几何变换
        transform_shape = tio.Compose([
            tio.RandomFlip(axes=int(np.random.randint(3, size=1)[0]), p=prob),
            tio.RandomAffine(p=prob)
        ])
        transformed_subject_all_imgs = transform_shape(subject_all_imgs)
        
        # 应用图像增强（仅对原始图像）
        transform_val = tio.Compose([
            tio.RandomBlur(p=prob),
            tio.RandomNoise(p=prob),
            tio.RandomMotion(p=prob),
            tio.RandomBiasField(p=prob),
            tio.RandomSpike(p=prob),
            tio.RandomGhosting(p=prob)
        ])
        transformed_subject_all_imgs['image'] = transform_val(transformed_subject_all_imgs['image'])
        
        # 更新张量
        for item in subject_all_imgs.keys():
            image_tensors[item] = transformed_subject_all_imgs[item].data
        
        return image_tensors
    
    def clear_cache(self):
        """清空缓存"""
        if hasattr(self, 'nifti_cache'):
            self.nifti_cache.clear()
        
        # 触发垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 测试代码
if __name__=="__main__":
    import pickle
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    
    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    
    # 加载数据并创建数据集
    data_dict = load_obj("../dataset_info/3dataset_info_BAStrain_merged")
    dataset = airway_dataset(data_dict)
    
    # 测试单个样本
    output = dataset.get(0)
    for name in output.keys():
        print(name, output[name].shape)
    
    # 测试DataLoader
    dataset.set_para(file_format='.nii.gz', crop_size=64, windowMin=-1000, windowMax=150,
                    need_tensor_output=True, need_transform=True)
    
    Dataset_loader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # 测试批量加载
    batch = next(iter(Dataset_loader))
    for name in batch.keys():
        print(name, batch[name].shape)
    
    # 清理缓存
    dataset.clear_cache()
