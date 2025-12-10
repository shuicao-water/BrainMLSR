import os
import ants
import argparse
import shutil
import numpy as np
import pandas as pd

def copy_transform_files(transform_paths, output_dir):
    """
    Copies transformation files to a specified directory.
    
    Parameters:
    transform_paths (list of str): List of file paths to be copied.
    output_dir (str): Directory to copy files to.
    """
    for i, transform_path in enumerate(transform_paths):
        # Generate output filename
        transform_filename = os.path.basename(transform_path)
        output_path = os.path.join(output_dir, transform_filename)
        
        # Copy the file
        if os.path.exists(transform_path):
            shutil.copy(transform_path, output_path)
            print(f"Copied {transform_path} to {output_path}")
        else:
            print(f"File does not exist: {transform_path}")


def ensure_output_dir_exists(output_dir):
    """
    检查输出目录是否存在，如果不存在则创建它。
    
    参数:
        output_dir (str): 输出目录的路径。
    
    返回:
        None
    """
    # 检查路径是否存在
    if not os.path.exists(output_dir):
        print(f"路径 '{output_dir}' 不存在，正在创建...")
        try:
            # 创建路径（包括可能的多级目录）
            os.makedirs(output_dir, exist_ok=True)
            print(f"路径 '{output_dir}' 已成功创建。")
        except Exception as e:
            print(f"创建路径 '{output_dir}' 失败: {e}")
    else:
        print(f"路径 '{output_dir}' 已存在。")


def main():
    parser = argparse.ArgumentParser(description="Perform image registration using ANTs.")
    parser.add_argument('--fixed', type=str, required=True, help='Path to the fixed image.')
    parser.add_argument('--moving', type=str, required=True, help='Path to the moving image.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory.')
    # parser.add_argument('--fwd_dir', type=str, required=True, help='Path to the forward transforms directory.')
    # parser.add_argument('--inv_dir', type=str, required=True, help='Path to the inverse transforms directory.')

    args = parser.parse_args()

    # 读取图像
    fixed_image = ants.image_read(args.fixed)
    moving_image = ants.image_read(args.moving)

    # 进行配准
    registration_result = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyN')
    
    # 获取移动图像的基名（不包含扩展名）
    moving_basename = os.path.splitext(os.path.basename(args.moving))[0]
    # 获取固定图像的基名（不包含s扩展名）
    fixed_basename = os.path.splitext(os.path.basename(args.fixed))[0]

    # # 调用函数确保路径存在
    # ensure_output_dir_exists(args.output_dir)

    # # 保存配准结果
    # # warpedmovout warpedmovout: Moving image warped to space of fixed image.
    # if 'warpedmovout' in registration_result:
    #     ants.image_write(registration_result['warpedmovout'], os.path.join(args.output_dir, f'{moving_basename}_to_{fixed_basename}_registered.nii'))
    # print("success write warpedmovout")

     # warpedmovout warpedmovout: Moving image warped to space of fixed image.
    if 'warpedmovout' in registration_result:
        ants.image_write(registration_result['warpedmovout'], args.output_dir)
    print("success write warpedmovout")




    # # warpedfixout: Fixed image warped to space of moving image
    # if 'warpedfixout' in registration_result:
    #     ants.image_write(registration_result['warpedfixout'], os.path.join(args.output_dir, f'{fixed_basename}_to_{moving_basename}_registered.nii'))
    # print("success write warpedfixout")


    # # 处理和保存变换文件
    # if 'fwdtransforms' in registration_result:
    #     copy_transform_files(registration_result['fwdtransforms'], args.fwd_dir)

    # if 'invtransforms' in registration_result:
    #     copy_transform_files(registration_result['invtransforms'], args.inv_dir)



if __name__ == "__main__":
    main()
