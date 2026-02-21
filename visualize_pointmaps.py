#!/usr/bin/env python3
"""
Convert .npy depth/pointmap files to PNG visualizations for debugging.

For each depth frame directory, generates:
  - depth_viz.png:    Depth map colored with turbo colormap
  - pointmap_viz.png: Pointmap XYZ channels as RGB (normalized per-channel)

With --objects: for each stereo frame-side, generates per-object pointmap
visualizations by masking the full pointmap with each object's segmentation mask.
  - Output: <frame_side>/objects/object_NNN_pointmap_viz.png

Usage:
    python visualize_pointmaps.py /path/to/droid_processed
    python visualize_pointmaps.py /path/to/droid_processed --force
    python visualize_pointmaps.py /path/to/droid_processed --objects
    python visualize_pointmaps.py /path/to/droid_processed --objects --filter
    python visualize_pointmaps.py /path/to/single/depth/frame_dir
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np


def visualize_depth(depth: np.ndarray) -> np.ndarray:
    """Colorize depth map using turbo colormap. Invalid (<=0) regions shown as black."""
    valid = depth > 0
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    # Normalize valid depth to 0-255
    d_min, d_max = depth[valid].min(), depth[valid].max()
    normalized = np.zeros_like(depth)
    if d_max > d_min:
        normalized[valid] = (depth[valid] - d_min) / (d_max - d_min)

    # Apply turbo colormap (uint8 input expected)
    gray = (normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    # Black out invalid regions
    colored[~valid] = 0

    return colored


def visualize_pointmap(pointmap: np.ndarray) -> np.ndarray:
    """Visualize XYZ pointmap as RGB image. Each channel normalized independently."""
    valid = np.isfinite(pointmap[:, :, 2]) & (pointmap[:, :, 2] > 0)
    viz = np.zeros((*pointmap.shape[:2], 3), dtype=np.uint8)

    if not valid.any():
        return viz

    for c in range(3):
        ch = pointmap[:, :, c].copy()
        ch_valid = ch[valid]
        c_min, c_max = ch_valid.min(), ch_valid.max()
        if c_max > c_min:
            normalized = (ch - c_min) / (c_max - c_min)
            normalized = np.clip(normalized, 0, 1)
            viz[:, :, c] = (normalized * 255).astype(np.uint8)
        else:
            viz[:, :, c] = 128

    viz[~valid] = 0
    return viz


def process_frame_dir(frame_dir: Path, force: bool = False) -> int:
    """Process a single depth frame directory. Returns number of images created."""
    count = 0

    depth_path = frame_dir / "depth.npy"
    pointmap_path = frame_dir / "pointmap.npy"

    if depth_path.exists():
        out = frame_dir / "depth_viz.png"
        if force or not out.exists():
            depth = np.load(depth_path)
            viz = visualize_depth(depth)
            cv2.imwrite(str(out), viz)
            count += 1

    if pointmap_path.exists():
        out = frame_dir / "pointmap_viz.png"
        if force or not out.exists():
            pointmap = np.load(pointmap_path)
            viz = visualize_pointmap(pointmap)
            cv2.imwrite(str(out), viz)
            count += 1

    return count


def visualize_object_pointmaps(
    base: Path, force: bool = False, use_filter: bool = False
) -> int:
    """
    For each stereo frame-side, mask the full pointmap with each object's
    segmentation mask and save per-object pointmap visualizations.

    Directory mapping:
      stereo/<cam>-stereo/{side}_NNNNNN/masks/mask_NNN_*.png  (masks)
      depth/<cam>-stereo/frame_NNNNNN/pointmap.npy            (pointmap)
      stereo/<cam>-stereo/{side}_NNNNNN/objects*/              (output)
    """
    total = 0

    # Walk all trajectories
    for env_dir in sorted(base.iterdir()):
        if not env_dir.is_dir() or env_dir.name.startswith('.'):
            continue
        for outcome_dir in sorted(env_dir.iterdir()):
            if not outcome_dir.is_dir():
                continue
            for traj_dir in sorted(outcome_dir.iterdir()):
                if not traj_dir.is_dir():
                    continue

                stereo_dir = traj_dir / "stereo"
                depth_dir = traj_dir / "depth"
                if not stereo_dir.exists() or not depth_dir.exists():
                    continue

                for cam_dir in sorted(stereo_dir.iterdir()):
                    if not cam_dir.is_dir() or not cam_dir.name.endswith("-stereo"):
                        continue

                    cam_name = cam_dir.name  # e.g. "24400334-stereo"

                    for frame_side_dir in sorted(cam_dir.iterdir()):
                        if not frame_side_dir.is_dir():
                            continue

                        m = re.match(r'^left_(\d+)$', frame_side_dir.name)
                        if not m:
                            continue

                        frame_idx = m.group(1)

                        # Find pointmap
                        pointmap_path = depth_dir / cam_name / f"frame_{frame_idx}" / "pointmap.npy"
                        if not pointmap_path.exists():
                            # Try without cam subdirectory
                            pointmap_path = depth_dir / f"frame_{frame_idx}" / "pointmap.npy"
                        if not pointmap_path.exists():
                            continue

                        masks_dir = frame_side_dir / "masks"
                        if not masks_dir.exists():
                            continue

                        # Process both object variants
                        for variant in ["objects", "objects_pointmap"]:
                            obj_dir = frame_side_dir / variant
                            if not obj_dir.exists():
                                continue

                            # Load filter if requested
                            visible_set = None
                            if use_filter:
                                suffix = "_pointmap" if variant == "objects_pointmap" else ""
                                filter_path = frame_side_dir / f"visible_objects{suffix}.json"
                                if filter_path.exists():
                                    with open(filter_path) as f:
                                        fdata = json.load(f)
                                    visible_set = set(fdata.get("visible_object_indices", []))

                            # Find which object indices exist
                            obj_indices = set()
                            for glb in obj_dir.glob("object_*.glb"):
                                om = re.match(r'object_(\d+)', glb.stem)
                                if om:
                                    obj_indices.add(int(om.group(1)))

                            if not obj_indices:
                                continue

                            # Lazy-load pointmap only when needed
                            pointmap = None

                            for idx in sorted(obj_indices):
                                # Apply filter
                                if visible_set is not None and idx not in visible_set:
                                    continue

                                out_path = obj_dir / f"object_{idx:03d}_pointmap_viz.png"
                                if not force and out_path.exists():
                                    continue

                                # Find corresponding mask
                                mask_matches = list(masks_dir.glob(f"mask_{idx:03d}_*.png"))
                                if not mask_matches:
                                    continue

                                if pointmap is None:
                                    pointmap = np.load(pointmap_path)

                                mask = cv2.imread(str(mask_matches[0]), cv2.IMREAD_GRAYSCALE)
                                if mask is None:
                                    continue

                                # Resize mask if needed
                                if mask.shape[:2] != pointmap.shape[:2]:
                                    mask = cv2.resize(mask, (pointmap.shape[1], pointmap.shape[0]),
                                                      interpolation=cv2.INTER_NEAREST)

                                # Apply mask to pointmap
                                mask_bool = mask > 127
                                masked_pm = pointmap.copy()
                                masked_pm[~mask_bool] = 0

                                viz = visualize_pointmap(masked_pm)
                                cv2.imwrite(str(out_path), viz)
                                total += 1

                            rel = frame_side_dir.relative_to(base)
                            if total > 0 and total % 50 == 0:
                                print(f"  ... {total} object pointmaps created so far")

    return total


def main():
    parser = argparse.ArgumentParser(description="Visualize depth/pointmap .npy files as PNGs")
    parser.add_argument("input_dir", help="droid_processed dir or single depth frame dir")
    parser.add_argument("--force", action="store_true", help="Overwrite existing visualizations")
    parser.add_argument("--objects", action="store_true",
                        help="Generate per-object pointmap visualizations (mask + pointmap)")
    parser.add_argument("--filter", action="store_true",
                        help="Only visualize objects in visible_objects.json (requires --objects)")
    args = parser.parse_args()

    base = Path(args.input_dir)
    if not base.exists():
        print(f"ERROR: {base} not found")
        sys.exit(1)

    if args.objects:
        print(f"Generating per-object pointmap visualizations...")
        if args.filter:
            print("  (filtering by visible_objects.json)")
        n = visualize_object_pointmaps(base, args.force, args.filter)
        print(f"\nDone: {n} object pointmap visualizations created")
        return

    # Check if this is a single frame directory
    if (base / "depth.npy").exists() or (base / "pointmap.npy").exists():
        n = process_frame_dir(base, args.force)
        print(f"Created {n} visualization(s) in {base}")
        return

    # Walk droid_processed tree looking for depth directories
    total = 0
    frame_dirs = sorted(base.rglob("depth.npy"))
    print(f"Found {len(frame_dirs)} depth frames")

    for depth_file in frame_dirs:
        frame_dir = depth_file.parent
        n = process_frame_dir(frame_dir, args.force)
        total += n
        if n > 0:
            print(f"  {frame_dir.relative_to(base)}: {n} image(s)")

    print(f"\nDone: {total} visualizations created")


if __name__ == "__main__":
    main()
