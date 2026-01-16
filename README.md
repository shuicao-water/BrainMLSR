# BrainMLSR
## 1. The pipeline of BrainMLSR

<p align="center">
  <img src="figures/pipeline.png" width="500" />
  <br>
  <em>Figure 1: pipeline of BrainMLSR.</em>
</p>

Main steps:
Reconstruct the inner and outer cortical surfaces.
Perform multi-signal layer reconstruction using BrainMLSR.
First, install the required environment dependencies.
```
pip install -r requirements.txt
```
## One-Click Execution Integration

SUBJECT_DIR is the path containing the raw images T1.nii.gz and T2FLAIR.nii.gz.
RESULT_DIR is the output directory.
CODE_DIR is the path to this project's source code.

Note: Ensure that a Python environment satisfying the requirements.txt is activated, and FreeSurfer is properly loaded.

```
sbatch BrainMLSR.sh "$subject_path" "$result_dir" "#code_path"
```

### Input Image Structure:
```
Subject/
  ├── T1.nii.gz
  └── T2FLAIR.nii.gz
```

### Final Images:
```
 Result/
  ├── mri/
      ├── T1_to_T2FLAIR_registered.mgz
      ├── T2_05.mgz
```

### Final Hypointense Layer Surfaces:
```
 Result/
  ├── surf/
      ├── lh_hypo_layer.inner
      ├── lh_hypo_layer.outer
      ├── rh_hypo_layer.inner
      ├── rh_hypo_layer.outer
```

### Cortical Surfaces from FreeSurfer (stored under standard FreeSurfer output):
```
 Result/
  ├── Freesurfer/
      ├──  ... (standard FreeSurfer outputs: lh.white, lh.pial, rh.white, rh.pial)
```

### Final Overall Directory Structure:
```
BrainMLSR/
├── Subject/
│   ├── T1.nii.gz
│   └── T2FLAIR.nii.gz
│
├── Result/
│   ├── Freesurfer/
│   │   └── ... (standard FreeSurfer outputs: lh.white, lh.pial, rh.white, rh.pial)
│   │
│   ├── mri/
│   │   ├── T1_05.mgz
│   │   ├── T1_to_T2FLAIR_registered.mgz
│   │   ├── T2_05.mgz
│   │   ├── t1.nii.gz
│   │   ├── t2_flair.nii.gz
│   │   └── run_recon.py
│   │
│   └── surf/
│       ├── lh_hypo_layer.inner
│       ├── lh_hypo_layer.outer
│       ├── lh_init_hypo_layer.inner
│       ├── lh_init_hypo_layer.outer
│       ├── rh_hypo_layer.inner
│       ├── rh_hypo_layer.outer
│       ├── rh_init_hypo_layer.inner
│       ├── rh_init_hypo_layer.outer
```

## Detailed Processing Workflow
## 1. Cortical Inner and Outer Surface Reconstruction
To facilitate broad usability, we use FreeSurfer as an example for preprocessing and generating the inner (white matter) and outer (pial) cortical surfaces. Alternatively, users may employ their own surface reconstruction methods, provided that:
The inner and outer surfaces have vertex-wise correspondence, and
They share the same triangular mesh topology.
First, convert DICOM images of T1 and T2-FLAIR to NIfTI format using dcm2niix. The resulting files are: T1.nii.gz and T2FLAIR.nii.gz.

### 1.1 Image Preprocessing
To ensure compatibility with FreeSurfer, resample both images to isotropic 0.5×0.5×0.5 mm resolution:
```
mri_convert T1.nii.gz T1_05.mgz -cs 0.5
mri_convert T2.nii.gz T2FLAIR_05.mgz -cs 0.5
```
### 1.2  Image Registration: T1 to T2-FLAIR
We register the T1 image to the T2-FLAIR space to align the reconstructed cortical surfaces with the T2-FLAIR image, since the intracortical signal layers are most visible in T2-FLAIR, we keep T2-FLAIR fixed and warp T1 accordingly. Use the provided registration script:
```
python Step00_Register.py --fixed T2FLAIR_05.mgz --moving T1_05.nii.gz --output_dir T1_05_reg.nii.gz
```
### 1.3 Cortical Surface Reconstruction
Run FreeSurfer on the registered T1 image to reconstruct the white and pial surfaces. Replace "UII_5T" with your desired subject ID or output directory name.
```
recon-all -all -i T1_05_reg.nii.gz -s "UII_5T" -openmp 8
```

## 2. Multi-Signal Intracortical Layer Reconstruction Using BrainMLSR
### 2.1 Initial Layer Surface Extraction
Use a gradient-based method to generate initial estimates of the low-signal intracortical layer boundaries. 
For the left hemisphere (lh), replace all rh prefixes with lh.
```
python Step01_Surf_Initialization.py \
    --white  rh.white  \
    --pial rh.pial \
    --T2flair T2FLAIR_05.mgz \
    --init_hypo_inner rh_init_hypo_layer.inner \
    --init_hypo_outer rh_init_hypo_layer.outer \
```

### 2.2 Multi-Signal Layer Surface Optimization
Refine the initial surfaces by minimizing an energy function that incorporates image intensity, geometric smoothness, and topological constraints. Again, replace rh with lh for the left hemisphere.
```
python Step02_Surf_optimization.py \
    --white_surf rh.white --init_hypo_inner rh_init_hypo_layer.inner --init_hypo_outer rh_init_hypo_layer.outer --pial_surf rh.pial \
    --T2_image T2FLAIR_05.mgz \
    --final_hypo_inner rh_hypo_layer.inner --final_hypo_outer rh_hypo_layer.outer
```
All parameters can be used with default values, or customized as needed:
```
python Step03_Surf_optimization.py \
    --white_surf rh.white --init_hypo_inner rh_init_hypo_layer.inner --init_hypo_outer rh_init_hypo_layer.outer --pial_surf rh.pial \
    --T2_image T2FLAIR_05.mgz \
    --final_hypo_inner rh_hypo_layer.inner --final_hypo_outer rh_hypo_layer.outer \
    --alpha_inner 3.0 --alpha_outer 3.0 --beta_inner 1.0 --beta_middle 1.0 --beta_outer 1.0 --gamma_inner 0.5 --gamma_outer 0.5 \
    --learning_rate 0.01 --iterations 80 --tol 1e-6

```
