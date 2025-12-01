# Observer Mask Simulation

Simulation of inter-observer variability for brain tumor segmentation using morphological perturbations and directional (COM-based) boundary shifts.

This tool generates synthetic observer masks from ground-truth segmentations, approximating realistic variability seen between human experts.

---

## How It Works

For each ground-truth segmentation, the pipeline:

1. Loads tumor segmentation  
2. Computes voxel volume and ground-truth volume  
3. Samples an observer-specific target volume ratio  
4. Chooses a patient profile with associated directional COM shift  
5. Generates an anisotropic ellipsoid kernel  
6. Applies iterative dilation or erosion to reach the target volume  
7. Ensures mask stays within the brain region  
8. Saves observer mask as: <CASEID>observer<N>.nii.gz


---

## Running the Simulation

Basic usage:

```bash
python simulate_observers.py \
 --input /path/to/BraTS-MEN-Train \
 --output /path/to/output/observer_masks

python simulate_observers.py \
    --input /home/user/BraTS-MEN-Train \
    --output /home/user/BraTS-MEN-Train/observer_masks \
    --workers 8

Example Dataset Compatibility
This tool works with any dataset following the structure below:
CASEID/
    CASEID-t1c.nii.gz
    CASEID-seg.nii.gz
Ground-truth masks must end with:
-seg.nii.gz
T1c images must end with:
-t1c.nii.gz

Cite This Work
If you use this simulation framework in your research, please cite:
Gillett, H., Stanley, E.A.M., Souza, R., Wilms, M., Forkert, N.D. (2026).
Simulating Inter-observer Variability Across Clinical Experience Levels for Brain Tumour Segmentation.
In: Guo, X., et al. Human-AI Collaboration. HAIC 2025. Lecture Notes in Computer Science, vol 16214.
Springer, Cham. https://doi.org/10.1007/978-3-032-08970-0_8