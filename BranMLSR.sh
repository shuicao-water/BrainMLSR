#!/bin/bash
#SBATCH -J _BrainMLSR # 作业名
#SBATCH -o _BrainMLSR_%j.o   # 标准输出文件名，包括作业ID
#SBATCH -e _BrainMLSR_%j.e   # 标准错误文件名，包括作业ID
#SBATCH -n 8                       # 指定core数量（例如8个）
#SBATCH -p bme_cpu                 # 指定分区
#SBATCH -N 1                       # 指定node数量（保持为1，如果任务可以在单个节点上完成）
#SBATCH --mem=64gb                 # 指定内存大小
#SBATCH --time=20:00:00             # 最大wallclock时间

# 参数接收
# SUBJECT_DIR 是存储原始图像nii的路径
# Result_Dir是输出路径
# Code_Dir是本项目的代码路径
SUBJECT_DIR=$1
Result_Dir=$2
Code_Dir=$3



# 加载freesurfer
module load apps/freesurfer7.3.2
source /public/software/apps/freesurfer_infant/freesurfer7.3.2/freesurfer/7.3.2-1/SetUpFreeSurfer.sh

# 加载必要的模块
module load tools/conda/anaconda.2023.09
# 激活Conda环境 这里 "mri" 改成自己的conda环境
source activate mri
# 强制将 mri 环境的 bin 目录放到 PATH 最前面
export PATH="/home_data/home/caoshui2024/.conda/envs/mri/bin:$PATH"


# 总体流程
# 1. 得到要处理的nii 一个t1 两个t2flair（甚至有的是三个）
# 2. t1 配准到 t2flair(reg)
# 3. t2flair 和 t1 改成 0.5mm
# 4. freesurfer t1(0.5mm）
# 5. initial surface
# 6. refine surface
# 7. thickness metric
# 改用mgz以及head.get_函数

# ========================================================== 识别对应的nii t1 t2flair =================================================================
# # 设置t1 和 t2flair 原始图像路径
# T1Image="/public_bme2/bme-dgshen/caoshui2024/5TMRI_Fazekas分级/Fazekas0级/冯新明_0000426325_085522/T1_Gre_fsp3D_Sag_iso0.8mm_ACS_091312_501.nii"
# T2flair="/public_bme2/bme-dgshen/caoshui2024/5TMRI_Fazekas分级/Fazekas0级/冯新明_0000426325_085522/T2_mx3D_Flair_Sag_fs_iso0.8_ACS_090719_401.nii"

# Result_Dir="/public_bme2/bme-dgshen/caoshui2024/5TMRI_Fazekas_Results/Fazekas0/Subject_6_0000426325_085522"

# 自动识别 T1 和 T2-flair 文件，不区分大小写
T1Image=$(find "$SUBJECT_DIR" -iname "t1*.nii" -type f | head -n 1)
T2flair=$(find "$SUBJECT_DIR" -iname "t2*.nii" -type f | head -n 1)

echo "Subject Dir: $SUBJECT_DIR"
echo "Found T1 image: $T1Image"
echo "Found T2 Flair image: $T2flair"
echo "Result Dir: $Result_Dir"

# 检查是否存在
if [ ! -f "$T1Image" ] || [ ! -f "$T2flair" ]; then
    echo "Missing T1 or T2 file in $SUBJECT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$Result_Dir"

# ========================================================== 1. 得到要处理的nii t1 t2flair =================================================================

# 记录开始时间
START_TIME_total=$(date +%s)  # 获取当前时间戳（秒级）
echo "Job started at: $(date)"  # 打印开始时间

Image_Dir="$Result_Dir/mri" 
# 确保输出目录存在
mkdir -p "$Image_Dir"

# 检查源文件是否存在
if [ ! -f "$T1Image" ]; then
    echo "错误：T1 源文件不存在！路径：$T1Image"
    exit 1
fi

if [ ! -f "$T2flair" ]; then
    echo "错误：T2 FLAIR 源文件不存在！路径：$T2flair"
    exit 1
fi

T1Image_original="$Image_Dir/t1.nii.gz"
T2flair_original="$Image_Dir/t2_flair.nii.gz"

# 开始复制
echo "正在复制 T1 图像..."
cp -v "$T1Image" $T1Image_original

echo "正在复制 T2 FLAIR 图像 ..."
cp -v "$T2flair" $T2flair_original

# 检查复制是否成功
if [ $? -eq 0 ]; then
    echo "图像复制完成，保存至：$Image_Dir"
else
    echo "复制失败，请检查权限或磁盘空间。"
    exit 1
fi

T1Image_original="$Image_Dir/t1.nii.gz"
T2flair_original="$Image_Dir/t2_flair.nii.gz"

# # ========================================================== 2. 去骨 (选做) ==========================================================================================

# # mri_synthstrip -i input.nii.gz -o stripped.nii.gz
# T1Image_brain="$Image_Dir/t1_brain.nii.gz"
# T2flair_brain="$Image_Dir/t2_flair_brain.nii.gz"

# mri_synthstrip -i "$T1Image" -o $T1Image_brain
# mri_synthstrip -i "$T2flair" -o $T2flair_brain

# # 检查复制是否成功
# if [ $? -eq 0 ]; then
#     echo "图像去骨完成"
# else
#     echo "去骨失败"
#     exit 1
# fi

# 如果不做去颅骨，就用下面这个代码。否则就注释下面两行 用上面一行。或者在T2 convert到0.5mm后进行去颅骨
T1Image_brain=$T1Image_original
T2flair_brain=$T2flair_original

# ========================================================== 3. t2flair 和 t1 改成 0.5mm ==========================================================================================
echo "开始执行 t2flair 和 t1 改成 0.5mm..."
# 定义输入和输出路径
CONVERT_T1_input_file=$T1Image_brain
CONVERT_T1_output_file="$Image_Dir/T1_05.mgz"
# 确保输出目录存在
output_dir=$(dirname "$CONVERT_T1_output_file")
mkdir -p "$output_dir"

# 执行 mri_convert 命令
echo "T1 开始执行 mri_convert..."
mri_convert "$CONVERT_T1_input_file" "$CONVERT_T1_output_file" -cs 0.5

# 定义输入和输出路径
CONVERT_T2_input_file=$T2flair_brain
CONVERT_T2_output_file="$Image_Dir/T2_05.mgz"
# 确保输出目录存在
output_dir=$(dirname "$CONVERT_T2_output_file")
mkdir -p "$output_dir"

# 执行 mri_convert 命令
echo "T2FLAIR 开始执行 mri_convert..."
mri_convert "$CONVERT_T2_input_file" "$CONVERT_T2_output_file" -cs 0.5

# 检查命令是否成功执行
if [ $? -eq 0 ]; then
    echo "mri_convert 成功完成！"
else
    echo "mri_convert 执行失败！"
    exit 1
fi

echo "完成执行 t2flair 和 t1 改成 0.5mm..."


# ========================================================== 2. t1 配准到 t2flair  ==========================================================================================
# 执行
echo "开始执行 t1 配准到 t2flair..."
# 设置Python脚本和图像文件的路径
REGISTER_SCRIPT_PATH="$Code_Dir/Step00_Register.py"
REGISTER_FIXED_IMAGE=$CONVERT_T2_output_file
REGISTER_MOVING_IMAGE=$CONVERT_T1_output_file
t1flair_register="$Image_Dir/T1_to_T2flair_registered.mgz"
# 确保输出目录存在
output_dir=$(dirname "$t1flair_register")
mkdir -p "$output_dir"

python $REGISTER_SCRIPT_PATH --fixed $REGISTER_FIXED_IMAGE --moving $REGISTER_MOVING_IMAGE --output_dir $t1flair_register 

# 检查命令是否成功执行
if [ $? -eq 0 ]; then
    echo "完成执行  t1 配准到 t2flair..."
else
    echo "执行失败  t1 配准到 t2flair..."
    exit 1
fi

# 改变t1 t2路径
T1_use=$t1flair_register
T2flair_use=$CONVERT_T2_output_file

# ========================================================== 4. freesurfer t1 ==========================================================================================
# 记录开始时间
START_TIME_freesurfer=$(date +%s)  # 获取当前时间戳（秒级）
echo "freesurfer Job started at: $(date)"  # 打印开始时间

echo "开始执行 freesurfer t1 ..."
export SUBJECTS_DIR="$Result_Dir/FreeSurfer" # 运⾏完成后⽂件的存储路径 
 
# 确保输出目录存在
mkdir -p "$SUBJECTS_DIR"

# 定义输入文件和输出主题名称
FREESURFER_INPUT_FILE=$T1_use
SUBJECT_NAME="Subject"

# 执行FreeSurfer的recon-all命令
recon-all -all -i "$FREESURFER_INPUT_FILE" -s "$SUBJECT_NAME" -openmp 8

# 检查命令是否成功执行
if [ $? -eq 0 ]; then
    echo "完成执行 freesurfer t1 ..."
else
    echo "执行失败 freesurfer t1 ..."
    exit 1
fi

# 记录结束时间
END_TIME_freesurfer=$(date +%s)  # 获取当前时间戳
ELAPSED_TIME=$((END_TIME_freesurfer - START_TIME_freesurfer))  # 计算耗时（秒）

# 输出运行时间
echo "freesurfer Job ended at: $(date)"
echo "freesurfer elapsed time: $ELAPSED_TIME seconds"


# ========================================================== 5. initial hypo layer surface  ==========================================================================================
# 记录开始时间
START_TIME_initial_surface=$(date +%s)  # 获取当前时间戳（秒级）
echo "initial surface Job started at: $(date)"  # 打印开始时间

LH_WHITE="$SUBJECTS_DIR/$SUBJECT_NAME/surf/lh.white"
LH_PIAL="$SUBJECTS_DIR/$SUBJECT_NAME/surf/lh.pial"
RH_WHITE="$SUBJECTS_DIR/$SUBJECT_NAME/surf/rh.white"
RH_PIAL="$SUBJECTS_DIR/$SUBJECT_NAME/surf/rh.pial"

# 记录开始时间
START_TIME_granular=$(date +%s)  # 获取当前时间戳（秒级）
echo "granular Job started at: $(date)"  # 打印开始时间

echo "开始执行 initial surface  ..."
# 在Slurm脚本中添加以下内容以检查并创建输出目录
Surf_Dir="$Result_Dir/surf"

# 检查并创建输出目录
mkdir -p $Surf_Dir

# 输出信息确认
echo "Output directories ensured at $Surf_Dir"

echo "开始执行 rh  ..."
# 继续运行Python脚本...
python $Code_Dir/Step01_Surf_Initialization.py \
    --white  $RH_WHITE \
    --pial $RH_PIAL \
    --T2flair $T2flair_use \
    --init_hypo_inner $Surf_Dir/rh_init_hypo_layer.inner \
    --init_hypo_outer $Surf_Dir/rh_init_hypo_layer.outer \

echo "完成执行 rh  ..."

echo "开始执行 lh  ..."
# 继续运行Python脚本...
python $Code_Dir/Step01_Surf_Initialization.py \
    --white  $LH_WHITE \
    --pial $LH_PIAL \
    --T2flair $T2flair_use \
    --init_hypo_inner $Surf_Dir/lh_init_hypo_layer.inner \
    --init_hypo_outer $Surf_Dir/lh_init_hypo_layer.outer \

echo "完成执行 lh  ..."

# 检查命令是否成功执行
if [ $? -eq 0 ]; then
    echo "完成执行 initial surface  ..."
else
    echo "执行失败 initial surface  ..."
    exit 1
fi

# 记录结束时间
END_TIME_initial_surface=$(date +%s)  # 获取当前时间戳
ELAPSED_TIME=$((END_TIME_initial_surface - START_TIME_initial_surface))  # 计算耗时（秒）

# 输出运行时间
echo "initial surface Job ended at: $(date)"
echo "initial surface elapsed time: $ELAPSED_TIME seconds"


# ========================================================== 6. final hypo layer surface  ==========================================================================================
# 记录开始时间
START_TIME_refine_surface=$(date +%s)  # 获取当前时间戳（秒级）
echo "refine surface Job started at: $(date)"  # 打印开始时间

echo "开始执行 refine surface  ..."


# 确保输出目录存在
output_dir=$(dirname "$lh_granular_inner")
mkdir -p "$output_dir"

echo "开始执行 lh refine surface  ..."

python $Code_Dir/Step02_Surf_optimization.py \
    --white_surf $RH_WHITE --init_hypo_inner $Surf_Dir/rh_init_hypo_layer.inner --init_hypo_outer $Surf_Dir/rh_init_hypo_layer.outer --pial_surf $RH_PIAL \
    --T2_image $T2flair_use \
    --final_hypo_inner $Surf_Dir/rh_hypo_layer.inner --final_hypo_outer $Surf_Dir/rh_hypo_layer.outer

echo "完成执行 lh refine surface  ..."


# 确保输出目录存在
output_dir=$(dirname "$rh_granular_inner")
mkdir -p "$output_dir"

echo "开始执行 rh refine surface  ..."

python $Code_Dir/Step02_Surf_optimization.py \
    --white_surf $LH_WHITE --init_hypo_inner $Surf_Dir/lh_init_hypo_layer.inner --init_hypo_outer $Surf_Dir/lh_init_hypo_layer.outer --pial_surf $LH_PIAL \
    --T2_image $T2flair_use \
    --final_hypo_inner $Surf_Dir/lh_hypo_layer.inner --final_hypo_outer $Surf_Dir/lh_hypo_layer.outer

echo "完成执行 rh refine surface  ..."

echo "完成执行 refine surface  ..."


# 检查命令是否成功执行
if [ $? -eq 0 ]; then
    echo "完成执行 refine surface granular ..."
else
    echo "执行失败 refine surface granular ..."
    exit 1
fi


# # ========================================================== END  ==========================================================================================
# 记录结束时间
END_TIME_total=$(date +%s)  # 获取当前时间戳
ELAPSED_TIME=$((END_TIME_total - START_TIME_total))  # 计算耗时（秒）

# 输出运行时间
echo "total ended at: $(date)"
echo "Total elapsed time: $ELAPSED_TIME seconds"
