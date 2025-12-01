#!/usr/bin/env python3
"""
Observer Mask Simulation Script
--------------------------------

This script generates simulated observer segmentation masks from
ground-truth labels using morphological volume perturbations and
directional (COM-based) boundary shifts. The goal is to approximate
inter-observer variability in medical image segmentation.

The tool:
    • Loads tumor ground-truth masks (BraTS-MEN format)
    • Uses observer-specific volume ratios and directional biases
    • Applies ellipsoidal dilation/erosion kernels

Author: Haley Gillett
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_erosion
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


NUM_OBSERVERS = 3

# Observer-specific volume ratios (across patient profiles)
OBSERVER_RATIOS = [
    [1.46, 0.88, 0.92],  
    [0.88, 1.17, 0.97],  
    [1.25, 0.85, 1.12],  
]

# COM directional shift means for patient profiles
COM_MEAN = np.array([
    [2.0, 4.9, 4.0],
    [0.6, 2.2, 2.3],
    [11.0, 7.9, 3.3],
])

COM_OFFSET = 0.5
COM_STD = 0.5

MIN_VOLUME_RATIO = 0.2  # Minimum 20% of original size before stopping erosion


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def voxel_volume(img: sitk.Image) -> float:
    return np.prod(img.GetSpacing())


def volume(mask: np.ndarray, voxel_vol: float) -> float:
    return np.sum(mask > 0) * voxel_vol


def create_ellipsoid(rx, ry, rz) -> np.ndarray:
    """Creates a 3D ellipsoid structuring element."""
    rx, ry, rz = map(lambda r: max(1, int(round(r))), (rx, ry, rz))
    Z, Y, X = np.ogrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]
    return ((X/rx)**2 + (Y/ry)**2 + (Z/rz)**2 <= 1).astype(np.uint8)


def apply_brain_constraint(mask_np: np.ndarray, brain_np: np.ndarray) -> np.ndarray:
    return mask_np & brain_np


def adjust_volume(
    mask_np: np.ndarray,
    target_vol: float,
    initial_vol: float,
    lr: float,
    ap: float,
    cc: float,
    brain_mask: np.ndarray,
    operation: str,
    max_iters: int = 100
) -> np.ndarray:
    """
    Adjust segmentation using ellipsoidal dilation or erosion until the
    estimated target volume is reached.
    """
    voxel_target = int(np.sum(mask_np) * (target_vol / initial_vol))
    voxel_min = int(np.sum(mask_np) * MIN_VOLUME_RATIO)

    kernel = create_ellipsoid(lr, ap, cc)
    current = mask_np.copy()
    current_voxels = np.sum(current)

    for _ in range(max_iters):
        if operation == "dilate" and current_voxels < voxel_target:
            candidate = binary_dilation(current, structure=kernel).astype(np.uint8)
        elif operation == "erode" and current_voxels > voxel_target:
            candidate = binary_erosion(current, structure=kernel).astype(np.uint8)
        else:
            break

        candidate = apply_brain_constraint(candidate, brain_mask)
        cand_count = np.sum(candidate)

        if cand_count == 0 or (operation == "erode" and cand_count < voxel_min):
            break

        current = candidate
        current_voxels = cand_count

    return current.astype(np.uint8)


# ---------------------------------------------------------------------------
# Main patient processing
# ---------------------------------------------------------------------------

def process_case(seg_path: str, output_dir: str):
    """
    Process a single ground-truth segmentation file and produce
    NUM_OBSERVERS simulated observer masks.
    """
    try:
        pid = os.path.basename(seg_path).replace("-seg.nii.gz", "")
        case_dir = os.path.dirname(seg_path)

        # Load ground-truth mask
        seg_img = sitk.ReadImage(seg_path)
        seg_np = (sitk.GetArrayFromImage(seg_img) != 0).astype(np.uint8)

        # Load T1c (for brain mask)
        t1c_candidates = [f for f in os.listdir(case_dir) if f.endswith("-t1c.nii.gz")]
        if not t1c_candidates:
            print(f"[WARN] {pid}: no T1c found, skipping.")
            return

        t1c_path = os.path.join(case_dir, t1c_candidates[0])
        t1c_img = sitk.ReadImage(t1c_path)
        t1c_np = sitk.GetArrayFromImage(t1c_img)

        # Simple brain mask heuristic
        brain_np = (t1c_np > np.percentile(t1c_np[t1c_np > 0], 10)).astype(np.uint8)

        # Volumes
        vox_vol = voxel_volume(seg_img)
        initial_vol = volume(seg_np, vox_vol)

        # Randomly pick a patient profile
        profile_id = np.random.randint(0, COM_MEAN.shape[0])
        com_profile = COM_MEAN[profile_id]

        for obs in range(NUM_OBSERVERS):
            ratio = OBSERVER_RATIOS[obs][profile_id]
            target_vol = initial_vol * ratio
            operation = "dilate" if ratio > 1 else "erode"

            # Directional kernel radii
            if operation == "dilate":
                lr, ap, cc = np.random.normal(com_profile + COM_OFFSET, COM_STD)
            else:
                lr, ap, cc = np.random.normal(com_profile - COM_OFFSET, COM_STD)

            new_mask = adjust_volume(
                seg_np,
                target_vol,
                initial_vol,
                lr, ap, cc,
                brain_np,
                operation,
            )

            # Save output
            out_img = sitk.GetImageFromArray(new_mask)
            out_img.CopyInformation(seg_img)
            out_path = os.path.join(output_dir, f"{pid}_observer_{obs+1}.nii.gz")
            sitk.WriteImage(out_img, out_path)

    except Exception as e:
        print(f"[ERROR] {seg_path}: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simulate observer masks.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to BraTS-MEN dataset root.")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to save simulated observer masks.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers.")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Collect ground-truth segmentations
    seg_files = []
    for root, _, files in os.walk(args.input):
        for f in files:
            if f.endswith("-seg.nii.gz"):
                seg_files.append(os.path.join(root, f))

    print(f"Found {len(seg_files)} cases.")

    # Parallel processing
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_case, seg, args.output): seg for seg in seg_files}

        for fut in as_completed(futures):
            seg = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                print(f"[ERROR] while processing {seg}: {exc}")


if __name__ == "__main__":
    main()
