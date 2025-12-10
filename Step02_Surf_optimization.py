import numpy as np
from scipy.spatial import KDTree
from nibabel.freesurfer.io import read_geometry, write_geometry
from mne.transforms import apply_trans
import nibabel as nib
from scipy.ndimage import sobel
import argparse
import os



# def compute_laplacian(vertices, adjacency):
#     """计算拉普拉斯平滑项及其梯度"""
#     laplacian_energy = 0.0
#     gradient = np.zeros_like(vertices)
    
#     for i in range(len(vertices)):
#         neighbor_sum = np.sum(vertices[adjacency[i]], axis=0)
#         n_neighbors = len(adjacency[i])
        
#         # 均匀权重
#         delta_i = vertices[i] - neighbor_sum / n_neighbors
        
#         # 累加拉普拉斯能量
#         laplacian_energy += np.dot(delta_i, delta_i)
        
#         # 梯度
#         # gradient[i] = 2 * delta_i
#         gradient[i] = delta_i
    
#     return laplacian_energy, gradient

def compute_laplacian(vertices, adjacency):
    laplacian_energy = 0.0
    gradient = np.zeros_like(vertices)
    
    for i in range(len(vertices)):
        # 检查索引范围
        valid_indices = [idx for idx in adjacency[i] if 0 <= idx < len(vertices)]
        if valid_indices:
            neighbor_sum = np.sum(vertices[valid_indices], axis=0)
            n_neighbors = len(valid_indices)
            
            delta_i = vertices[i] - neighbor_sum / n_neighbors
            
            laplacian_energy += np.dot(delta_i, delta_i)
            gradient[i] = delta_i
        else:
            print(f"Vertex {i} has no valid neighbors.")
    
    return laplacian_energy, gradient

def compute_distance_error(vertices, target_coords, adjacency):
    # vertices是要被形变的顶点
    # target_coords是参考目标的顶点 是不变的
    """计算距离误差及其梯度"""
    distance_error = 0.0
    gradient = np.zeros_like(vertices)
    
    for i in range(len(vertices)):
        current_distance = np.linalg.norm(vertices[i] - target_coords[i])
        
        # 计算期望距离 d_0 (基于相邻顶点到目标表面的距离平均值)
        if len(adjacency[i]) > 0:
            d_0 = np.mean([np.linalg.norm(vertices[j] - target_coords[j]) for j in adjacency[i]])
        else:
            d_0 = current_distance
        # 梯度方向
        direction = (vertices[i] - target_coords[i]) / current_distance if current_distance > 1e-6 else np.zeros(3)
        # 累加距离误差
        distance_error += (current_distance - d_0) ** 2 / 2
        
        # 梯度
        gradient[i] = (current_distance - d_0) * direction
    
    return distance_error, gradient

# 坐标未回到物理空间
# def compute_image_gradient_error(vertices, Torig, image_data, grad_x, grad_y, grad_z):
#     """计算图像梯度"""
#     image_gradient_energy = 0.0
#     gradient = np.zeros_like(vertices)
#     for i in range(len(vertices)):
#         point_vox_coord = xyz_to_vox_coord_float(Torig, vertices[i])
#         first_gradient_vox_value = trilinear_interpolation(image_data, point_vox_coord)
#         # 改-
#         image_gradient_energy -= first_gradient_vox_value
#         second_gradient_x = trilinear_interpolation(grad_x, point_vox_coord)
#         second_gradient_y = trilinear_interpolation(grad_y, point_vox_coord)
#         second_gradient_z = trilinear_interpolation(grad_z, point_vox_coord)

#         # 梯度
#         second_gradient_vox = np.array([second_gradient_x, second_gradient_y, second_gradient_z])
#         # gradient[i] = second_gradient_vox
#         gradient[i] = vox_to_xyz_coord(Torig, second_gradient_vox)
#         # print(f"second_gradient_vox, gradient={second_gradient_vox}", flush=True)
#         # print(f"gradient, gradient={gradient}", flush=True)
#         # if i>10:
#         #     break
    
#     # print(f"compute_image_gradient_error, gradient={gradient}", flush=True)
    
#     return image_gradient_energy, gradient

def compute_image_gradient_error(vertices, Torig, image_data, grad_x, grad_y, grad_z):
    """计算图像梯度"""
    image_gradient_energy = 0.0
    gradient = np.zeros_like(vertices)
    
    # 提取仿射矩阵的线性部分，并计算其逆矩阵的转置
    R = Torig[:3, :3]  # 线性部分
    try:
        R_inv_T = np.linalg.inv(R).T  # (R^{-1})^T
    except np.linalg.LinAlgError:
        print("Warning: Singular affine matrix. Using identity.")
        R_inv_T = np.eye(3)

    for i in range(len(vertices)):
        point_vox_coord = xyz_to_vox_coord_float(Torig, vertices[i])
        
        # 插值得到梯度幅值（用于能量）
        first_gradient_vox_value = trilinear_interpolation(image_data, point_vox_coord)
        image_gradient_energy -= first_gradient_vox_value  # 负号！最小化负的能量
        
        # 插值得到体素空间梯度分量
        second_gradient_x = trilinear_interpolation(grad_x, point_vox_coord)
        second_gradient_y = trilinear_interpolation(grad_y, point_vox_coord)
        second_gradient_z = trilinear_interpolation(grad_z, point_vox_coord)
        second_gradient_vox = np.array([second_gradient_x, second_gradient_y, second_gradient_z])
        
        # # 将梯度从体素空间转换到世界空间
        # gradient_world = R_inv_T @ second_gradient_vox
        # gradient[i] = -gradient_world
        gradient[i] = -second_gradient_vox
    
    return image_gradient_energy, gradient
    
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

# 从体素坐标得到体素的值
def get_vox_value(brainmask_data, vox_coord):  
    return brainmask_data[vox_coord[0], vox_coord[1], vox_coord[2]]

def xyz_to_vox_coord_float(Torig, xyz):
    vox_coord = apply_trans(np.linalg.inv(Torig), xyz)
    return vox_coord

# 体素到XYZ
def vox_to_xyz_coord(Torig, vox):
    xyz = apply_trans(Torig, vox)
    return xyz


# def compute_gradient_vector_xyz(input_image_path):
#     # 加载MRI图像
#     mri_img = nib.load(input_image_path)
#     Torig = mri_img.header.get_vox2ras_tkr()  # 获取从体素坐标到物理坐标的转换矩阵
#     image_data = mri_img.get_fdata()
    
#     # 计算x, y, z方向上的梯度
#     grad_x = sobel(image_data, axis=0)  # x方向上的梯度
#     grad_y = sobel(image_data, axis=1)  # y方向上的梯度
#     grad_z = sobel(image_data, axis=2)  # z方向上的梯度
    
#     # 对每个点的方向向量进行归一化
#     grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
#     # 防止除以零，设置一个很小的值作为默认值
#     grad_magnitude[grad_magnitude == 0] = 1
    
#     grad_x /= grad_magnitude
#     grad_y /= grad_magnitude
#     grad_z /= grad_magnitude
    
#     return Torig, image_data, grad_x, grad_y, grad_z


# sobel＋归一化
# def compute_gradient_vector_xyz(input_image_path):
#     # 加载MRI图像
#     mri_img = nib.load(input_image_path)
    
#     # 使用 affine 替代 get_vox2ras_tkr
#     Torig = mri_img.affine.copy()
#     # Torig = mri_img.header.get_vox2ras_tkr()

#     image_data = mri_img.get_fdata()
    
#     # 计算x, y, z方向上的梯度
#     grad_x = sobel(image_data, axis=0)  # x方向上的梯度
#     grad_y = sobel(image_data, axis=1)  # y方向上的梯度
#     grad_z = sobel(image_data, axis=2)  # z方向上的梯度
    
#     # 对每个点的方向向量进行归一化
#     grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)  # 避免除以零
    
#     grad_x /= grad_magnitude
#     grad_y /= grad_magnitude
#     grad_z /= grad_magnitude
    
#     return Torig, image_data, grad_x, grad_y, grad_z


import numpy as np
import nibabel as nib

# def compute_gradient_vector_xyz(input_image_path):
#     # 加载MRI图像
#     mri_img = nib.load(input_image_path)
    
#     # 判断是否为 .mgz 文件（不区分大小写）
#     if input_image_path.lower().endswith('.mgz'):
#         Torig = mri_img.header.get_vox2ras_tkr()
#     else:
#         Torig = mri_img.affine.copy()
        
#     # 获取梯度幅值图像数据
#     gradient_magnitude = mri_img.get_fdata()  # 这里已经是 |∇I_original|
    
#     # 计算x, y, z方向上的梯度（对梯度幅值求导）
#     grad_x, grad_y, grad_z = np.gradient(gradient_magnitude)
    
#     # 不需要归一化，因为我们希望保留导数的实际大小
    
#     return Torig, gradient_magnitude, grad_x, grad_y, grad_z
def compute_gradient_magnitude_first_order(image_data):
    """
    使用前向/后向差分计算一阶梯度幅值（非 Sobel）。
    """
    depth, height, width = image_data.shape

    grad_x = np.zeros_like(image_data, dtype=np.float32)
    grad_y = np.zeros_like(image_data, dtype=np.float32)
    grad_z = np.zeros_like(image_data, dtype=np.float32)

    # x: depth (axis=0)
    grad_x[:-1, :, :] = image_data[1:, :, :] - image_data[:-1, :, :]
    grad_x[-1, :, :] = image_data[-1, :, :] - image_data[-2, :, :]

    # y: height (axis=1)
    grad_y[:, :-1, :] = image_data[:, 1:, :] - image_data[:, :-1, :]
    grad_y[:, -1, :] = image_data[:, -1, :] - image_data[:, -2, :]

    # z: width (axis=2)
    grad_z[:, :, :-1] = image_data[:, :, 1:] - image_data[:, :, :-1]
    grad_z[:, :, -1] = image_data[:, :, -1] - image_data[:, :, -2]

    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    return gradient_magnitude

def compute_gradient_vector_xyz(input_image_path):
    """
    端到端：从原始 MRI 图像 → 一阶梯度幅值 → 归一化的二阶梯度方向向量。
    
    Args:
        input_image_path (str): 原始 MRI 文件路径 (.nii, .nii.gz, .mgz)
    
    Returns:
        Torig (np.ndarray): 4x4 RAS 变换矩阵（FreeSurfer TKReg 或标准 affine）
        grad_mag (np.ndarray): 一阶梯度幅值 (D, H, W)
        grad_x_norm (np.ndarray): 二阶单位梯度 x 分量 (D, H, W)
        grad_y_norm (np.ndarray): 二阶单位梯度 y 分量 (D, H, W)
        grad_z_norm (np.ndarray): 二阶单位梯度 z 分量 (D, H, W)
    """
    # --- Step 1: Load original image ---
    img = nib.load(input_image_path)
    data = img.get_fdata()  # shape: (D, H, W)

    # Determine Torig
    if input_image_path.lower().endswith('.mgz'):
        Torig = img.header.get_vox2ras_tkr()
    else:
        Torig = img.affine.copy()

    # --- Step 2: Compute first-order gradient magnitude ---
    grad_mag = compute_gradient_magnitude_first_order(data)  # (D, H, W)

    # --- Step 3: Compute second-order gradient on grad_mag ---
    # np.gradient 默认按 axis=0,1,2 对应 (D, H, W)
    grad_x2, grad_y2, grad_z2 = np.gradient(grad_mag)  # each: (D, H, W)

    # Stack into vector field: shape (D, H, W, 3)
    grad_vec = np.stack([grad_x2, grad_y2, grad_z2], axis=-1)

    # Normalize to unit vectors
    norm = np.linalg.norm(grad_vec, axis=-1, keepdims=True)  # (D, H, W, 1)
    norm_safe = np.where(norm == 0, 1.0, norm)
    grad_unit = grad_vec / norm_safe  # (D, H, W, 3)

    # Split back to components
    grad_x_norm = grad_unit[..., 0]
    grad_y_norm = grad_unit[..., 1]
    grad_z_norm = grad_unit[..., 2]

    return Torig, grad_mag, grad_x_norm, grad_y_norm, grad_z_norm

# def compute_gradient_vector_xyz(input_image_path):
#     # 加载MRI图像
#     mri_img = nib.load(input_image_path)
    
#     # 判断是否为 .mgz 文件（不区分大小写）
#     if input_image_path.lower().endswith('.mgz'):
#         Torig = mri_img.header.get_vox2ras_tkr()
#     else:
#         Torig = mri_img.affine.copy()
        
#     # 获取梯度幅值图像数据（|∇I_original|）
#     gradient_magnitude = mri_img.get_fdata()
    
#     # 计算 x, y, z 方向上的梯度（对梯度幅值求导）
#     grad_x, grad_y, grad_z = np.gradient(gradient_magnitude)
    
#     # 堆叠成 (H, W, D, 3) 的向量场
#     grad_vec = np.stack([grad_x, grad_y, grad_z], axis=-1)  # shape: (H, W, D, 3)
    
#     # 计算每个体素的 L2 范数（梯度幅值的梯度大小）
#     norm = np.linalg.norm(grad_vec, axis=-1, keepdims=True)  # shape: (H, W, D, 1)
    
#     # 避免除零：将 norm 为 0 的地方设为 1（这样归一化后仍为 0 向量）
#     norm_safe = np.where(norm == 0, 1.0, norm)
    
#     # 归一化得到单位方向向量
#     grad_unit = grad_vec / norm_safe  # shape: (H, W, D, 3)
    
#     # 拆分回 x, y, z 分量（可选，保持接口一致）
#     grad_x_norm = grad_unit[..., 0]
#     grad_y_norm = grad_unit[..., 1]
#     grad_z_norm = grad_unit[..., 2]
    
#     return Torig, gradient_magnitude, grad_x_norm, grad_y_norm, grad_z_norm

# def bm_to_Torig_data(brainmask_file):
#     # brainmask到Torig和data
#     brainmask = nib.load(brainmask_file)
#     brainmask_data = brainmask.get_fdata()  
#     Torig = brainmask.header.get_vox2ras_tkr()
#     return Torig, brainmask_data

# def build_adjacency(num_vertices, faces):
#     """构建邻接矩阵"""
#     # num_vertices = np.max(faces) + 1
#     adjacency = [[] for _ in range(num_vertices)]
#     for face in faces:
#         # 确保face是一个长度为3的列表或元组
#         if len(face) != 3:
#             print(f"Invalid face: {face}", flush=True)
#             continue
#         for i in range(3):
#             if face[(i+1)%3] not in adjacency[face[i]]:
#                 adjacency[face[i]].append(face[(i+1)%3])
#             if face[(i+2)%3] not in adjacency[face[i]]:
#                 adjacency[face[i]].append(face[(i+2)%3])
#     return adjacency

def build_adjacency(num_vertices, faces):
    """构建邻接列表"""
    adjacency = [[] for _ in range(num_vertices)]
    
    for face in faces:
        # 确保face是一个长度为3的列表或元组
        if len(face) != 3:
            print(f"Invalid face: {face}")
            continue
        
        # 遍历face的每个顶点，添加邻居关系
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            v3 = face[(i + 2) % 3]
            
            # 检查索引范围
            if v1 < 0 or v1 >= num_vertices or v2 < 0 or v2 >= num_vertices or v3 < 0 or v3 >= num_vertices:
                print(f"Index out of range in face: {face}")
                continue
            
            # 添加边（无重复）
            if v2 not in adjacency[v1]:
                adjacency[v1].append(v2)
            if v3 not in adjacency[v1]:
                adjacency[v1].append(v3)
            
            if v1 not in adjacency[v2]:
                adjacency[v2].append(v1)
            if v3 not in adjacency[v2]:
                adjacency[v2].append(v3)
            
            if v1 not in adjacency[v3]:
                adjacency[v3].append(v1)
            if v2 not in adjacency[v3]:
                adjacency[v3].append(v2)
    
    return adjacency

def project_gradient_to_line_vectorized(grads, directions):
    """
    将一组梯度向量批量投影到对应的指定方向上。
    
    Parameters:
    - grads: 形状为 (N, 3) 的梯度向量数组，N 是顶点数量。
    - directions: 形状为 (N, 3) 的方向向量数组。
    
    Returns:
    - 投影后的梯度向量数组，形状为 (N, 3)。
    """
    # 计算单位方向向量
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    # 避免除以0，将接近0的值设置为1（这样不会改变原始梯度）
    norms[norms < 1e-6] = 1
    unit_directions = directions / norms
    
    # 计算投影长度
    projection_lengths = np.sum(grads * unit_directions, axis=1, keepdims=True)
    
    # 计算投影后的梯度
    projected_grads = projection_lengths * unit_directions
    return projected_grads


def read_abnormal_csv(csv_file):
    """从 CSV 文件读取 abnormal 标记"""
    df = pd.read_csv(csv_file)
    return df['is_abnormal'].values  # 返回异常标记的数组

# 原始的参数
def gradient_descent(inner_coords, outer_coords, v_inner, v_outer, faces, input_image_path, \
                        alpha_inner, alpha_outer,  beta_inner, beta_middle, beta_outer, gamma_inner, gamma_outer,  \
                        learning_rate, iterations, tol):
    """使用梯度下降优化顶点位置"""
    num_vertices = v_inner.shape[0]
    Torig, image_data, grad_x, grad_y, grad_z = compute_gradient_vector_xyz(input_image_path)
    # 构建邻接矩阵
    adjacency = build_adjacency(num_vertices, faces)
    
    print(f"alpha_inner={alpha_inner}, alpha_outer={alpha_outer},  beta_inner={beta_inner}, beta_middle={beta_middle}, beta_outer={beta_outer}\n, \
          gamma_inner={gamma_inner}, gamma_outer={gamma_outer}, learning_rate={learning_rate}, iterations={iterations}, tol={tol}")

    for iter in range(iterations):
        # 计算能量和梯度
        laplacian_energy_inner, laplacian_gradient_inner = compute_laplacian(v_inner, adjacency)
        laplacian_energy_outer, laplacian_gradient_outer = compute_laplacian(v_outer, adjacency)
        
        distance_error_outer, distance_gradient_outer = compute_distance_error(v_outer, outer_coords, adjacency)
        distance_error_middle, distance_gradient_middle_outer = compute_distance_error(v_outer, v_inner, adjacency)
        distance_error_middle, distance_gradient_middle_inner = compute_distance_error(v_inner, v_outer, adjacency)
        distance_error_inner, distance_gradient_inner = compute_distance_error(v_inner, inner_coords, adjacency)

        image_gradient_energy_outer, image_gradient_gradient_outer = compute_image_gradient_error(v_outer, Torig, image_data, grad_x, grad_y, grad_z)
        image_gradient_energy_inner, image_gradient_gradient_inner = compute_image_gradient_error(v_inner, Torig, image_data, grad_x, grad_y, grad_z)

        # 改成 Imgae gradinet +
        total_energy = alpha_inner * laplacian_energy_inner + alpha_outer * laplacian_energy_outer + \
                       beta_outer * distance_error_outer + beta_middle * distance_error_middle + beta_inner * distance_error_inner + \
                       gamma_outer * image_gradient_energy_outer + gamma_inner * image_gradient_energy_inner
        
        # 改成 Imgae gradinet +
        gradient_inner = alpha_inner * laplacian_gradient_inner + beta_inner * distance_gradient_inner + beta_middle * distance_gradient_middle_inner + gamma_inner * image_gradient_gradient_inner
        gradient_outer = alpha_outer * laplacian_gradient_outer + beta_outer * distance_gradient_outer + beta_middle * distance_gradient_middle_outer + gamma_outer * image_gradient_gradient_outer
        

        # 对梯度进行投影处理
        deform_direction = outer_coords - inner_coords
        gradient_inner = project_gradient_to_line_vectorized(gradient_inner, deform_direction)
        gradient_outer = project_gradient_to_line_vectorized(gradient_outer, deform_direction)

        # 更新顶点位置
        new_v_inner = v_inner - learning_rate * gradient_inner
        new_v_outer = v_outer - learning_rate * gradient_outer
        
        # 通过check进行判断
        # 验证新位置的顺序性和共线性
        for i in range(len(new_v_inner)):
            if not check_collinearity_and_order_single(inner_coords[i], new_v_inner[i], new_v_outer[i], outer_coords[i]):
                # 如果不满足条件，恢复原来的坐标
                new_v_inner[i] = v_inner[i]
                new_v_outer[i] = v_outer[i]


        # 检查停止条件
        if np.linalg.norm(new_v_inner - v_inner) < tol and np.linalg.norm(new_v_outer - v_outer) < tol:
            print("Converged due to small change in vertices")
            break
            
        v_inner = new_v_inner
        v_outer = new_v_outer
    
    return v_inner, v_outer


def check_collinearity_and_order_single(inner_gm, inner_gr, outer_gr, outer_gm, tolerance=1e-3):
    """
    Check if a single set of points are collinear and correctly ordered.
    
    Parameters:
    - inner_gm, inner_gr, outer_gr, outer_gm: Coordinates of the point on each surface.
    - tolerance: Allowed deviation for considering vectors collinear.
    
    Returns:
    A boolean indicating whether the point meets the conditions.
    """
    # 计算向量
    vec1 = inner_gr - inner_gm
    vec2 = outer_gr - inner_gm
    vec3 = outer_gm - inner_gm
    
    # 检查共线性: 使用向量叉乘接近0来判断
    cross1 = np.linalg.norm(np.cross(vec1, vec2))
    cross2 = np.linalg.norm(np.cross(vec2, vec3))
    
    if cross1 > tolerance or cross2 > tolerance:
        return False

    # 检查方向一致性: 通过内积判断
    dot1 = np.dot(vec1, vec2)
    dot2 = np.dot(vec2, vec3)
    
    if not (dot2 >= dot1 and dot1 >= 0):
        return False
        
    return True

def main(white_surf, pial_surf, init_hypo_inner, init_hypo_outer, T2_image, final_hypo_inner, final_hypo_outer, \
                alpha_inner, alpha_outer,  beta_inner, beta_middle, beta_outer, gamma_inner, gamma_outer,                \
                learning_rate, iterations, tol):
    # 加载数据
    inner_coords, faces, volume_info, create_stamp  = read_geometry(white_surf,read_metadata=True, read_stamp=True)
    outer_coords, _ = read_geometry(pial_surf)
    initial_v_inner, _,  = read_geometry(init_hypo_inner)
    initial_v_outer, _,  = read_geometry(init_hypo_outer)



    # 执行梯度下降优化
    optimized_v_inner, optimized_v_outer = gradient_descent(inner_coords, outer_coords, initial_v_inner, initial_v_outer, faces, T2_image,       \
                                                                alpha_inner, alpha_outer,  beta_inner, beta_middle, beta_outer, gamma_inner, gamma_outer, \
                                                                learning_rate, iterations, tol)

    # 保存结果
    write_geometry(final_hypo_inner, optimized_v_inner, faces, create_stamp, volume_info)
    write_geometry(final_hypo_outer, optimized_v_outer, faces, create_stamp, volume_info)
    print(f"final_hypo_inner={final_hypo_inner}", flush=True)
    print(f"final_hypo_outer={final_hypo_outer}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process surface geometries and optimize using gradient descent.")
    parser.add_argument('--white_surf', required=True, type=str, help='Path to the inner white matter surface file.')
    parser.add_argument('--pial_surf', required=True, type=str, help='Path to the pial surface file.')
    parser.add_argument('--init_hypo_inner', required=True, type=str, help='Path to the initial inner hypointense layer surf file.')
    parser.add_argument('--init_hypo_outer', required=True, type=str, help='Path to the initial outer hypointense layer surf file.')
    parser.add_argument('--T2_image', required=True, type=str, help='Path to the T2 image file.')
    parser.add_argument('--final_hypo_inner', required=True, type=str, help='Path to save the optimized inner final surf file.')
    parser.add_argument('--final_hypo_outer', required=True, type=str, help='Path to save the optimized outer final surf file.')

    # 可选的优化参数（带默认值）
    parser.add_argument('--alpha_inner', type=float, default=3.0, help='Weight for inner layer smoothness (default: 3.0)')
    parser.add_argument('--alpha_outer', type=float, default=3.0, help='Weight for outer layer smoothness (default: 3.0)')
    parser.add_argument('--beta_inner', type=float, default=1.0, help='Weight for inner layer distance to white (default: 1.0)')
    parser.add_argument('--beta_middle', type=float, default=1.0, help='Weight for middle layer constraint (default: 1.0)')
    parser.add_argument('--beta_outer', type=float, default=1.0, help='Weight for outer layer distance to pial (default: 1.0)')
    parser.add_argument('--gamma_inner', type=float, default=0.5, help='Weight for inner layer gradient alignment (default: 0.5)')
    parser.add_argument('--gamma_outer', type=float, default=0.5, help='Weight for outer layer gradient alignment (default: 0.5)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for gradient descent (default: 0.01)')
    parser.add_argument('--iterations', type=int, default=80, help='Maximum number of iterations (default: 80)')
    parser.add_argument('--tol', type=float, default=1e-6, help='Convergence tolerance (default: 1e-6)')

    args = parser.parse_args()

    if not all(map(os.path.exists, [args.white_surf, args.pial_surf, args.init_hypo_inner, args.init_hypo_outer, args.T2_image])):
        missing_files = [f for f in ['white_surf', 'pial_surf', 'init_hypo_inner', 'init_hypo_outer', 'T2_image'] if not os.path.exists(getattr(args, f))]
        raise FileNotFoundError(f"The following files do not exist: {', '.join(missing_files)}")

    # main(args.white_surf, args.pial_surf, args.init_hypo_inner, args.init_hypo_outer, args.T2_image, \
    #                             args.final_hypo_inner, args.final_hypo_outer)

    # 调用 main 函数，传入所有参数
    main(
        white_surf=args.white_surf,
        pial_surf=args.pial_surf,
        init_hypo_inner=args.init_hypo_inner,
        init_hypo_outer=args.init_hypo_outer,
        T2_image=args.T2_image,
        final_hypo_inner=args.final_hypo_inner,
        final_hypo_outer=args.final_hypo_outer,

        alpha_inner=args.alpha_inner,
        alpha_outer=args.alpha_outer,
        beta_inner=args.beta_inner,
        beta_middle=args.beta_middle,
        beta_outer=args.beta_outer,
        gamma_inner=args.gamma_inner,
        gamma_outer=args.gamma_outer,
        learning_rate=args.learning_rate,
        iterations=args.iterations,
        tol=args.tol
    )