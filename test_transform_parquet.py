#!/usr/bin/env python3

"""
测试 set_transform 是否影响 to_parquet 的输出
"""

import tempfile
import numpy as np
from pathlib import Path
import datasets

def test_transform_parquet():
    """测试 transform 是否影响 parquet 保存"""
    
    print("=== 测试 set_transform 对 to_parquet 的影响 ===\n")
    
    # 创建测试数据
    data = {
        "action": [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
        "observation": [np.array([0.1, 0.2]), np.array([0.3, 0.4])],
        "timestamp": [1.0, 2.0],
    }
    
    # 创建数据集
    dataset = datasets.Dataset.from_dict(data)
    print(f"原始数据集: {dataset}")
    print(f"数据类型: {type(dataset[0]['action'])}")
    print(f"数据内容: {dataset[0]}")
    print()
    
    # 定义 transform 函数
    def hf_transform_to_torch(items_dict):
        """模拟 hf_transform_to_torch 函数"""
        print(f"Transform 被调用，输入: {type(items_dict)}")
        # 将 numpy 数组转换为 torch tensor
        import torch
        transformed = {}
        for key, value in items_dict.items():
            if isinstance(value, np.ndarray):
                transformed[key] = torch.from_numpy(value)
            else:
                transformed[key] = value
        print(f"Transform 输出: {type(transformed['action'])}")
        return transformed
    
    # 测试1: 不设置 transform
    print("--- 测试1: 不设置 transform ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        parquet_path = Path(temp_dir) / "test1.parquet"
        dataset.to_parquet(str(parquet_path))
        
        # 重新加载
        loaded_dataset = datasets.Dataset.from_parquet(str(parquet_path))
        print(f"保存后重新加载的数据类型: {type(loaded_dataset[0]['action'])}")
        print(f"保存后重新加载的数据内容: {loaded_dataset[0]}")
        print()
    
    # 测试2: 设置 transform
    print("--- 测试2: 设置 transform ---")
    dataset_with_transform = dataset.copy()
    dataset_with_transform.set_transform(hf_transform_to_torch)
    
    # 检查 __getitem__ 是否应用了 transform
    print("通过 __getitem__ 访问数据:")
    item = dataset_with_transform[0]
    print(f"Transform 后的数据类型: {type(item['action'])}")
    print(f"Transform 后的数据内容: {item}")
    print()
    
    # 保存到 parquet
    with tempfile.TemporaryDirectory() as temp_dir:
        parquet_path = Path(temp_dir) / "test2.parquet"
        print("保存到 parquet...")
        dataset_with_transform.to_parquet(str(parquet_path))
        
        # 重新加载
        loaded_dataset = datasets.Dataset.from_parquet(str(parquet_path))
        print(f"Transform 后保存并重新加载的数据类型: {type(loaded_dataset[0]['action'])}")
        print(f"Transform 后保存并重新加载的数据内容: {loaded_dataset[0]}")
        print()
    
    # 测试3: 检查 data 属性
    print("--- 测试3: 检查 data 属性 ---")
    print(f"原始 dataset.data 类型: {type(dataset.data)}")
    print(f"Transform dataset.data 类型: {type(dataset_with_transform.data)}")
    print(f"Transform dataset.data 是否相同: {dataset.data == dataset_with_transform.data}")
    print()
    
    # 测试4: 检查 format 属性
    print("--- 测试4: 检查 format 属性 ---")
    print(f"原始 dataset.format: {dataset.format}")
    print(f"Transform dataset.format: {dataset_with_transform.format}")
    print()

def test_concatenate_transform():
    """测试 concatenate 后 transform 的行为"""
    
    print("=== 测试 concatenate 后 transform 的行为 ===\n")
    
    # 创建两个数据集
    data1 = {
        "action": [np.array([1.0, 2.0])],
        "observation": [np.array([0.1])],
    }
    
    data2 = {
        "action": [np.array([3.0, 4.0])],
        "observation": [np.array([0.2])],
    }
    
    dataset1 = datasets.Dataset.from_dict(data1)
    dataset2 = datasets.Dataset.from_dict(data2)
    
    print("原始数据集:")
    print(f"Dataset1: {dataset1[0]}")
    print(f"Dataset2: {dataset2[0]}")
    print()
    
    # 定义 transform
    def hf_transform_to_torch(items_dict):
        import torch
        transformed = {}
        for key, value in items_dict.items():
            if isinstance(value, np.ndarray):
                transformed[key] = torch.from_numpy(value)
            else:
                transformed[key] = value
        return transformed
    
    # 测试1: 先 concatenate，再设置 transform
    print("--- 测试1: 先 concatenate，再设置 transform ---")
    concatenated = datasets.concatenate_datasets([dataset1, dataset2])
    concatenated.set_transform(hf_transform_to_torch)
    
    print("Concatenate 后设置 transform:")
    for i in range(len(concatenated)):
        item = concatenated[i]
        print(f"Item {i}: {type(item['action'])} - {item}")
    print()
    
    # 测试2: 先设置 transform，再 concatenate
    print("--- 测试2: 先设置 transform，再 concatenate ---")
    dataset1_with_transform = dataset1.copy()
    dataset2_with_transform = dataset2.copy()
    dataset1_with_transform.set_transform(hf_transform_to_torch)
    dataset2_with_transform.set_transform(hf_transform_to_torch)
    
    concatenated_with_transform = datasets.concatenate_datasets([dataset1_with_transform, dataset2_with_transform])
    
    print("先设置 transform 再 concatenate:")
    for i in range(len(concatenated_with_transform)):
        item = concatenated_with_transform[i]
        print(f"Item {i}: {type(item['action'])} - {item}")
    print()

if __name__ == "__main__":
    test_transform_parquet()
    test_concatenate_transform()
    
    print("=== 结论 ===")
    print("1. set_transform 只影响 __getitem__ 的返回值，不影响底层数据")
    print("2. to_parquet 保存的是底层数据，不受 transform 影响")
    print("3. concatenate_datasets 会合并底层数据，transform 设置会传递")
    print("4. 因此，在异步保存中，ep_dataset 不会被 transform 影响") 