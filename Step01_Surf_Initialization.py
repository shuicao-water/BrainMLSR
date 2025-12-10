import nibabel as nib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import mne
from mne.io.constants import FIFF
from mne.transforms import apply_trans
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os
import math
import argparse
from multiprocessing import Pool, cpu_count
import nibabel as nib
import numpy as np

def xyz_to_vox_coord_float(Torig, xyz):
    vox_coord = apply_trans(np.linalg.inv(Torig), xyz)
    return vox_coord


def bm_to_Torig_data(brainmask_file):
    """
    读取NIfTI格式的脑部掩模文件，并返回仿射变换矩阵和图像数据。
    
    参数:
        brainmask_file (str): 脑部掩模文件路径。
        
    返回:
        tuple: 包含仿射变换矩阵和图像数据的元组。
    """
    # 加载NIfTI文件
    brainmask = nib.load(brainmask_file)
    # 获取图像数据
    brainmask_data = brainmask.get_fdata()
    # 检查是否为 .mgz（不区分大小写）
    if brainmask_file.lower().endswith('.mgz'):
        Torig = brainmask.header.get_vox2ras_tkr()
    else:
        Torig = brainmask.affine.copy()
    
    return Torig, brainmask_data

def ras_sample(start_ras, end_ras, num_samples):
    """
        输入 start_ras, end_ras, num_sample
        输出 一个列表是20个sample的ras坐标
    """
    vector = end_ras - start_ras  
    # 计算步长  
    step = vector / (num_samples - 1)  
    # 初始化体素值列表  
    ras_sample_points = []  
    # 遍历步长并计算每个点的体素值  
    # 在切片上取平均值
    for i in range(num_samples):  
        ras_point = start_ras + i * step  
        ras_sample_points.append(ras_point)  
    
    return ras_sample_points

# 从体素坐标得到体素的值
def get_vox_value(brainmask_data, vox_coord):  
    return brainmask_data[vox_coord[0], vox_coord[1], vox_coord[2]]

def trilinear_interpolation(brainmask_data, vox_coord):  
    x = vox_coord[0]
    y = vox_coord[1]
    z = vox_coord[2]
    # 确定整数坐标  
    x_floor, y_floor, z_floor = int(x), int(y), int(z)  
      
    # 确定周围的8个整数坐标点  
    coords = [  
        (x_floor,     y_floor,     z_floor),  
        (x_floor + 1, y_floor,     z_floor),  
        (x_floor,     y_floor + 1, z_floor),  
        (x_floor + 1, y_floor + 1, z_floor),  
        (x_floor,     y_floor,     z_floor + 1),  
        (x_floor + 1, y_floor,     z_floor + 1),  
        (x_floor,     y_floor + 1, z_floor + 1),  
        (x_floor + 1, y_floor + 1, z_floor + 1)  
    ]  
      
    # 获取这些点的体素值  
    values = [get_vox_value(brainmask_data, coord) for coord in coords] 

    # 插值计算  
    # 首先在z=z_floor平面上进行两次二维线性插值，得到两个中间值  
    # 然后在z=z_floor+1平面上进行两次二维线性插值，得到另外两个中间值  
    # 最后在这四个中间值上进行一维线性插值，得到目标点的体素值  
      
    # 在z=z_floor平面上的插值  
    val00 = values[0] + (x - x_floor) * (values[1] - values[0]) / 1  
    val01 = values[2] + (x - x_floor) * (values[3] - values[2]) / 1  
    val_z0 = val00 + (y - y_floor) * (val01 - val00) / 1  
      
    # 在z=z_floor+1平面上的插值  
    val10 = values[4] + (x - x_floor) * (values[5] - values[4]) / 1  
    val11 = values[6] + (x - x_floor) * (values[7] - values[6]) / 1  
    val_z1 = val10 + (y - y_floor) * (val11 - val10) / 1  
      
    # 在z方向上进行最终插值  
    result = val_z0 + (z - z_floor) * (val_z1 - val_z0) / 1  
      
    return result  

def vox_value_sample(T2_Torig, brainmask_data, ras_sample_points):
    vox_values = []
    for point in ras_sample_points:
        vox_coord = xyz_to_vox_coord_float(T2_Torig, point)
        vox_value = trilinear_interpolation(brainmask_data, vox_coord)
        vox_values.append(vox_value)
    # num_samples = len(vox_values)

    return vox_values

def find_key_points(data):
    # caoshui 11/26
    """
    输入:
        data (list or numpy array): 包含一系列数值的列表或数组。
    返回:
        list of int: 分割点的索引列表，第一个是最大正值（上升最快），第二个是起始到该点中最小的负值（下降最快）。
        注意：索引是基于原始数据的索引，且只在剔除首尾 5 个点后寻找。
    """
    data = np.array(data)
    
    max_positive_idx = None
    min_negative_idx = None

    # 至少需要 11 个点才能去掉首尾 5 个点并计算导数
    if len(data) < 11:
        return [None, None]

    # 去掉首尾 5 个点
    trim = 5
    middle_data = data[trim:-trim]   # 原本是 data[1:-1]

    # 计算导数
    derivative = np.diff(middle_data)

    # ---- 找最大正导数对应的 index ----
    if len(derivative) > 0:
        max_deriv_value = np.max(derivative)
        if max_deriv_value > 0:
            max_positive_idx_in_middle = np.argmax(derivative)
            # 转换回原始数据索引
            max_positive_idx = max_positive_idx_in_middle + trim

    # ---- 找最小负导数（仅在 0~max_positive_idx 之间）----
    if max_positive_idx is not None and max_positive_idx > trim:
        # 在 data[trim : max_positive_idx] 范围内找下降最快
        segment = np.diff(data[trim:max_positive_idx])
        if len(segment) > 0:
            min_deriv_value = np.min(segment)
            if min_deriv_value < 0:
                min_negative_idx_in_segment = np.argmin(segment)
                # 转换回原始数据索引
                min_negative_idx = min_negative_idx_in_segment + trim

    return min_negative_idx, max_positive_idx


# def find_key_points(data):
#     """
#     输入:
#         data (list or numpy array): 包含一系列数值的列表或数组。
#     返回:
#         list of int: 分割点的索引列表，第一个是最大正值（上升最快），第二个是起始到该点中最小的负值（下降最快）。
#         注意：索引是基于原始数据的索引，且只在第二个点到倒数第二个点之间寻找。
#     """
#     data = np.array(data)
    
#     # 默认分割点
#     max_positive_idx = None
#     min_negative_idx = None

#     # 至少需要3个点才能去掉首尾并计算导数 
#     if len(data) < 3:
#         return [None, None]

#     # 提取中间部分的数据（第二个点到倒数第二个点） 防止曲面之间的顶点有重合的情况
#     middle_data = data[1:-1]
    
#     # 计算中间部分的一阶导数
#     derivative = np.diff(middle_data)

#     # 找最大正值的位置（在中间部分的导数中）
#     if len(derivative) > 0:
#         max_deriv_value = np.max(derivative)
#         if max_deriv_value > 0:
#             max_positive_idx_in_middle = np.argmax(derivative)
#             # 转换为原始数据的索引（+1是因为middle_data从原始数据的索引1开始）
#             max_positive_idx = max_positive_idx_in_middle + 1
#         # else:
#         #     # 没有正值，使用中间位置
#         #     max_positive_idx = int(0.6 * len(middle_data)) + 1  # 转换为原始索引

#     # 找最小负值（在起始到max_positive_idx之间，基于原始数据索引）
#     if max_positive_idx is not None and max_positive_idx > 1:  # 确保有足够的点
#         # 计算原始数据中从索引1到max_positive_idx的导数
#         segment = np.diff(data[1:max_positive_idx])
#         if len(segment) > 0:
#             min_deriv_value = np.min(segment)
#             if min_deriv_value < 0:
#                 min_negative_idx_in_segment = np.argmin(segment)
#                 # 转换为原始数据的索引（+1是因为segment从原始数据的索引1开始）
#                 min_negative_idx = min_negative_idx_in_segment + 1

#     # # 如果没有找到有效的关键点，设置默认值
#     # if max_positive_idx is None:
#     #     max_positive_idx = int(0.6 * (len(data) - 2)) + 1  # 中间位置，忽略首尾
    
#     # if min_negative_idx is None:
#     #     min_negative_idx = int(0.5 * max_positive_idx) if max_positive_idx > 0 else 1

#     return min_negative_idx, max_positive_idx



# def process_index(index, lh_w_v1, lh_p_v1, T2_Torig, T2_brainmask_data, num_samples, png_save_path_fold):
#     """
#     处理单个索引的逻辑。
#     """
#     vox_coords = np.arange(num_samples)
    
#     # 空间坐标采样 --> 体素坐标采样
#     T2_white_ras = lh_w_v1[index]
#     T2_pial_ras = lh_p_v1[index]
#     ras_sample_points = ras_sample(T2_white_ras, T2_pial_ras, num_samples)
#     vox_value_sample_points = vox_value_sample(T2_Torig, T2_brainmask_data, ras_sample_points, png_save_path)

#     data = vox_value_sample_points
#     split_points_index = find_key_points(data)
#     hypointense_layer_inner = ras_sample_points[split_points_index[0]]
#     hypointense_layer_outer = ras_sample_points[split_points_index[1]]

#     return hypointense_layer_inner, hypointense_layer_outer



# def find_mesh_point_ras_coord(lh_w_v1, lh_p_v1, T2_Torig, T2_brainmask_data, 
#                               num_samples=100):
#     """
#         输入：T1 mesh, t1 t2的torig data
#         输出：max的点，按照顺序

#         方法：
#             1.先按照顺序依次读取
#             2.T1空间坐标-T2空间空间坐标-T2体素坐标-T2体素坐标变化-离散取值- -空间坐标
#     """
#     length = len(lh_w_v1)
#     indices = range(length)

#     hypointense_layer_inner_list = []
#     hypointense_layer_outer_list = []

#     # 按顺序处理每个索引
#     for index in indices:
#         hypointense_layer_inner, hypointense_layer_outer = process_index(
#             index, lh_w_v1, lh_p_v1, T2_Torig, T2_brainmask_data, num_samples
#         )
#         hypointense_layer_inner_list.append(hypointense_layer_inner)
#         hypointense_layer_outer_list.append(hypointense_layer_outer)
    
#     # 转换为 NumPy 数组后返回
#     return (
#         np.array(hypointense_layer_inner_list),
#         np.array(hypointense_layer_outer_list)
#     )


def process_all_vertices_refined(lh_w_v1, lh_p_v1, T2_Torig, T2_brainmask_data, num_samples):
    n_vertices = len(lh_w_v1)
    inner_coords = np.full((n_vertices, 3), np.nan)
    outer_coords = np.full((n_vertices, 3), np.nan)

    # 缓存
    all_ras_profiles = []
    valid_outer_rel = []          # max_idx / (N-1)
    valid_inner_local_rel = []    # min_idx / max_idx （仅当两者都有效）

    # 第一轮：检测 + 收集统计量
    for i in range(n_vertices):
        ras_profile = ras_sample(lh_w_v1[i], lh_p_v1[i], num_samples)
        all_ras_profiles.append(ras_profile)

        vox_values = vox_value_sample(T2_Torig, T2_brainmask_data, ras_profile)
        min_idx, max_idx = find_key_points(vox_values)  # 你当前的函数（无默认值）

        # 记录 outer 相对位置（全径向）
        if max_idx is not None:
            outer_rel = max_idx / (num_samples - 1)
            valid_outer_rel.append(outer_rel)
            outer_coords[i] = ras_profile[max_idx]

        # 记录 inner 局部相对位置（相对于 max_idx）
        if min_idx is not None and max_idx is not None and max_idx > 0:
            inner_local_rel = min_idx / max_idx
            valid_inner_local_rel.append(inner_local_rel)
            inner_coords[i] = ras_profile[min_idx]

    # === 第二轮：填充缺失值 ===

    # 1. 填充缺失的 outer (max_positive_idx)
    if valid_outer_rel:
        median_outer_rel = np.median(valid_outer_rel)
    else:
        median_outer_rel = 0.7  # fallback（但应警告）
        print("⚠️ Warning: No valid outer points! Using fallback=0.7")

    for i in range(n_vertices):
        if np.isnan(outer_coords[i, 0]):
            fill_outer_idx = int(round(median_outer_rel * (num_samples - 1)))
            fill_outer_idx = np.clip(fill_outer_idx, 1, num_samples - 2)
            outer_coords[i] = all_ras_profiles[i][fill_outer_idx]

    # 2. 填充缺失的 inner (min_negative_idx)
    if valid_inner_local_rel:
        median_inner_local_rel = np.median(valid_inner_local_rel)
    else:
        median_inner_local_rel = 0.5  # fallback: 在 outer 的一半位置
        print("⚠️ Warning: No valid inner points! Using local fallback=0.5")

    for i in range(n_vertices):
        if np.isnan(inner_coords[i, 0]):
            # 获取当前顶点的 outer_idx（已填充）
            outer_ras = outer_coords[i]
            # 找到 outer_ras 在 ras_profile 中的索引（因为可能有浮点误差，用最近邻）
            ras_profile = all_ras_profiles[i]
            distances = np.linalg.norm(ras_profile - outer_ras, axis=1)
            outer_idx = np.argmin(distances)

            # 计算 inner 填充位置：在 [0, outer_idx] 区间内
            fill_inner_idx = int(round(median_inner_local_rel * outer_idx))
            fill_inner_idx = np.clip(fill_inner_idx, 1, outer_idx-2)  # 确保 ≤ outer_idx
            inner_coords[i] = ras_profile[fill_inner_idx]

    return inner_coords, outer_coords


def extract_signal_surfaces(white_file, pial_file, T2_file, num_samples, \
                                init_hypo_inner, init_hypo_outer):
    """
        输入：T1的mesh, T1的图像, T2的图像
        输出：新组织的mesh

        方法：
            1.遍历所有的一组点
            2.T1空间坐标-T2空间空间坐标-T2体素坐标-T2体素坐标变化-离散-拟合-最大拟合点-最大离散点-最大体素坐标
            3.能得到最大体素坐标的就有 没有的 根据已知的最大点在两点间的相对距离，取出一个点
            4.可以得到max的mesh
        备注：
            1.先做lh的
        
    """
    # # 读取T1的mesh
    # lh_w_v1, lh_w_f1 = nib.freesurfer.io.read_geometry(white_file)
    # lh_p_v1, lh_p_f1 = nib.freesurfer.io.read_geometry(pial_file)
    # 读取T1的mesh 如果是freesurfer的结果
    lh_w_v1, lh_w_f1, volume_info, create_stamp = nib.freesurfer.io.read_geometry(white_file,read_metadata=True, read_stamp=True)
    lh_p_v1, lh_p_f1, _, _ = nib.freesurfer.io.read_geometry(pial_file,read_metadata=True, read_stamp=True)

    # 读取T2 Flair的数据
    T2_Torig, T2_brainmask_data = bm_to_Torig_data(T2_file)

    # 获取中间层坐标点（两个层）
    hypointense_layer_inner_list, hypointense_layer_outer_list = process_all_vertices_refined(
        lh_w_v1, lh_p_v1, T2_Torig, T2_brainmask_data, num_samples
    )

    # # 创建输出目录
    # os.makedirs(output_file_path, exist_ok=True)

    # # 保存为 freesurfer 格式的几何文件
    # nib.freesurfer.io.write_geometry(init_hypo_inner, hypointense_layer_inner_list, lh_w_f1)
    # nib.freesurfer.io.write_geometry(init_hypo_outer, hypointense_layer_outer_list, lh_w_f1)
    # 保存为 freesurfer 格式的几何文件
    nib.freesurfer.io.write_geometry(init_hypo_inner, hypointense_layer_inner_list, lh_w_f1, create_stamp, volume_info)
    nib.freesurfer.io.write_geometry(init_hypo_outer, hypointense_layer_outer_list, lh_w_f1, create_stamp, volume_info)

    print(f"已保存inner到：{init_hypo_inner}")
    print(f"已保存outer到：{init_hypo_outer}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some freesurfer data.")
    parser.add_argument('--white', required=True, help='Path to the left hemisphere white matter surface file.')
    parser.add_argument('--pial', required=True, help='Path to the left hemisphere pial surface file.')
    parser.add_argument('--T2flair', required=True, help='Path to the T2 MRI file.')
    parser.add_argument('--init_hypo_inner', required=True, help='Output inner directory path.')
    parser.add_argument('--init_hypo_outer', required=True, help='Output outer directory path.')
    parser.add_argument('--num_samples', default=100, help='number of the sample points along the line')
    args = parser.parse_args()

    # 处理半球
    extract_signal_surfaces(args.white, args.pial, args.T2flair, args.num_samples, \
        args.init_hypo_inner, args.init_hypo_outer)
