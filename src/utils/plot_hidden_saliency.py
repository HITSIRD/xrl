import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_hidden_saliency_files(root_dir):
    """可视化每个文件夹中的 hidden_saliency.pkl 文件"""
    # 查找所有 hidden_saliency.pkl 文件
    pkl_files = list(Path(root_dir).rglob("hidden_saliency.pkl"))

    for pkl_file in pkl_files:
        try:
            # 加载 pkl 文件
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            # 确保 data 是 list 且包含 128 维向量
            if isinstance(data, list) and len(data) > 0:
                # 转换为 numpy 数组
                data.insert(0, np.zeros_like(data[0]))
                saliency_array = np.array(data)  # shape: (n, 128)
                if saliency_array.ndim == 3 and saliency_array.shape[1] == 1:
                    saliency_array = saliency_array.squeeze(axis=1)

                # 绘制热力图
                plt.figure(figsize=(9, 4))
                plt.imshow(saliency_array, cmap='viridis', aspect='auto')
                plt.colorbar(label='Saliency')
                plt.xlabel('Hidden Dimension')
                plt.ylabel('Time Step')
                plt.title(f'Hidden Saliency')

                # 保存图像
                save_path = pkl_file.parent / f"hidden_saliency.pdf"
                plt.tight_layout()
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()

                print(f"已保存: {save_path}")

        except Exception as e:
            print(f"处理 {pkl_file} 时出错: {e}")

# 使用示例
visualize_hidden_saliency_files("experiments/hrl/real_kitchen/cola/seq_prior_bc/top50/explanation")
