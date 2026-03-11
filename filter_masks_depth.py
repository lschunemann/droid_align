#!/usr/bin/env python3
"""
Filter segmentation masks using depth discontinuities from pointmaps.

For each mask, extracts the Z (depth) values of masked pixels, computes a
robust depth estimate (median + IQR), and removes pixels that deviate
significantly from the object's own depth range. A connected-component pass
then removes isolated floating blobs.

Thresholds are always relative to the object's depth, so filtering adapts
to near and far objects alike — no absolute depth values needed.

Filtered masks are saved to left_*/masks_filtered/ so Stage 2 can use
them with --use-filtered.

Usage:
    # Preview stats without writing
    python filter_masks_depth.py droid_processed_ --dry-run

    # Run with default settings
    python filter_masks_depth.py droid_processed_

    # Looser filtering (keep more)
    python filter_masks_depth.py droid_processed_ --iqr-factor 5 --min-rel-tol 0.25

    # Also save side-by-side viz (original vs filtered)
    python filter_masks_depth.py droid_processed_ --visualize
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


# ── Core filtering ────────────────────────────────────────────────────────────

def filter_mask_by_depth(mask: np.ndarray, pointmap: np.ndarray,
                          iqr_factor: float, min_rel_tol: float,
                          min_component_ratio: float = 0.1,
                          gradient_fill: bool = False,
                          gradient_threshold: float = 0.05) -> np.ndarray:
    """
    Remove background pixels from a segmentation mask using depth.

    Strategy:
      1. Extract Z values of masked pixels.
      2. Compute median Z (Z_med) and IQR.
      3. threshold = max(iqr_factor * IQR,  min_rel_tol * Z_med)
         → adapts to object size and distance; no absolute values.
      4. Remove masked pixels outside [Z_med - threshold, Z_med + threshold].
      5. Keep all connected components >= min_component_ratio of the largest.
      6. (optional) Gradient-fill: flood-fill from centroid blocked by depth
         gradient barriers — removes enclosed background holes that depth
         threshold misses because FoundationStereo interpolated their depth.

    Returns the filtered mask (may equal input if filtering would remove
    everything or there are too few valid depth values).
    """
    if not mask.any():
        return mask

    z_map = pointmap[:, :, 2]

    # Resize pointmap to mask resolution if needed
    if z_map.shape != mask.shape:
        z_map = cv2.resize(z_map, (mask.shape[1], mask.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    z_vals = z_map[mask]
    valid = (z_vals > 0) & np.isfinite(z_vals)
    z_vals = z_vals[valid]

    if len(z_vals) < 20:
        return mask  # too sparse to filter reliably

    z_med = float(np.median(z_vals))
    if z_med <= 0:
        return mask

    z_q25, z_q75 = np.percentile(z_vals, [25, 75])
    z_iqr = float(z_q75 - z_q25)

    threshold = max(iqr_factor * z_iqr, min_rel_tol * z_med)

    depth_ok = (
        (np.abs(z_map - z_med) <= threshold) &
        (z_map > 0) &
        np.isfinite(z_map)
    )
    filtered = mask & depth_ok

    if not filtered.any():
        return mask  # filtering too aggressive — return original

    # Keep all connected components that are at least min_component_ratio
    # of the largest — discards tiny noise but preserves occluded object parts
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        filtered.astype(np.uint8), connectivity=8
    )
    if n_labels > 2:  # 0=background, 1=only component
        areas = stats[1:, cv2.CC_STAT_AREA]  # exclude background label 0
        max_area = float(areas.max())
        keep = np.where(areas >= min_component_ratio * max_area)[0] + 1
        filtered = np.isin(labels, keep)

    # Optional: gradient-connectivity flood-fill to remove enclosed holes
    if gradient_fill:
        filled = _gradient_fill(filtered, pointmap, gradient_threshold)
        if filled.any():
            filtered = filled

    return filtered


def _gradient_fill(mask: np.ndarray, pointmap: np.ndarray,
                   gradient_threshold: float) -> np.ndarray:
    """
    Keep only the connected component of the mask that contains the centroid,
    where connectivity is blocked by relative depth-gradient barriers.

    This removes enclosed background regions (e.g. table visible through a
    cup-handle hole) whose depth was interpolated by FoundationStereo to match
    the object, making them invisible to pure depth-threshold filtering.

    How it works:
      1. Compute the depth gradient magnitude, normalised by median object depth.
      2. Mark pixels with rel_gradient > gradient_threshold as barriers.
      3. Find connected components of (mask & ~barriers) using cv2 (fast C++).
      4. Keep the component that contains the mask centroid.
      5. Dilate back by 1 pixel to recover object-edge pixels that sit on the
         barrier boundary.

    gradient_threshold: lower → stricter (more barriers). Try 0.02–0.10.
    """
    z_map = pointmap[:, :, 2].astype(np.float32)
    if z_map.shape != mask.shape:
        z_map = cv2.resize(z_map, (mask.shape[1], mask.shape[0]),
                           interpolation=cv2.INTER_LINEAR)

    # Light smoothing to reduce stereo noise before gradient
    z_smooth = cv2.GaussianBlur(z_map, (5, 5), 1.0)
    gx = cv2.Sobel(z_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z_smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    z_vals = z_map[mask]
    z_vals = z_vals[(z_vals > 0) & np.isfinite(z_vals)]
    if len(z_vals) < 20:
        return mask
    z_med = float(np.median(z_vals))
    if z_med <= 0:
        return mask

    # Relative gradient: scale-independent threshold
    rel_grad = grad_mag / z_med
    barriers = rel_grad > gradient_threshold

    passable = mask & ~barriers
    if not passable.any():
        return mask  # everything is a barrier — give up

    # Seed = median position of passable mask pixels (robust against outliers)
    ys, xs = np.where(passable)
    cy = int(np.median(ys))
    cx = int(np.median(xs))
    if not passable[cy, cx]:
        dists = (ys - cy) ** 2 + (xs - cx) ** 2
        idx = int(np.argmin(dists))
        cy, cx = int(ys[idx]), int(xs[idx])

    # Connected components of passable region (C++ fast)
    _, labels = cv2.connectedComponents(passable.astype(np.uint8), connectivity=4)
    seed_label = int(labels[cy, cx])
    if seed_label == 0:
        return mask  # seed on non-passable pixel (shouldn't happen)

    core = (labels == seed_label)

    # Dilate by 1 pixel to re-include object-edge pixels on the gradient barrier
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.dilate(core.astype(np.uint8), kernel).astype(bool) & mask

    return result if result.any() else mask


# ── Path helpers ──────────────────────────────────────────────────────────────

def find_pointmap(traj_dir: Path, cam_name: str, frame_idx: str) -> Path | None:
    depth_dir = traj_dir / "depth"
    for p in [
        depth_dir / cam_name / f"frame_{frame_idx}" / "pointmap.npy",
        depth_dir / f"frame_{frame_idx}" / "pointmap.npy",
    ]:
        if p.exists():
            return p
    return None


# ── Visualization helper ──────────────────────────────────────────────────────

def add_label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((22, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return np.vstack([bar, img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)])


def save_viz(image_path: Path, orig_masks: list, filt_masks: list, out_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        return
    h, w = img.shape[:2]
    np.random.seed(42)
    colors = np.random.randint(80, 230, (max(len(orig_masks), 1), 3), dtype=np.uint8)

    def to_hw(m):
        if m.shape != (h, w):
            return cv2.resize(m.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        return m

    def color_overlay(masks):
        v = img.copy().astype(np.float32)
        for i, m in enumerate(masks):
            m = to_hw(m)
            col = colors[i % len(colors)].tolist()
            ov = np.zeros_like(v)
            ov[m] = col
            v = v * (1 - 0.5 * m[:, :, None]) + ov * 0.5 * m[:, :, None]
        return np.clip(v, 0, 255).astype(np.uint8)

    def bw_combined(masks):
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in masks:
            combined = np.maximum(combined, to_hw(m).astype(np.uint8) * 255)
        return combined

    def diff_image(orig_masks, filt_masks):
        """Red = removed pixels, green = kept pixels, dark = not in any mask."""
        removed  = np.zeros((h, w), dtype=bool)
        kept     = np.zeros((h, w), dtype=bool)
        for orig, filt in zip(orig_masks, filt_masks):
            orig = to_hw(orig)
            filt = to_hw(filt)
            removed |= (orig & ~filt)
            kept    |= filt
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[kept]    = (80, 200, 80)
        out[removed] = (60, 60, 220)   # BGR red
        return out

    row1 = np.hstack([
        add_label(color_overlay(orig_masks), "original (color overlay)"),
        add_label(color_overlay(filt_masks), "filtered (color overlay)"),
    ])
    row2 = np.hstack([
        add_label(bw_combined(orig_masks),   "original (B&W masks)"),
        add_label(bw_combined(filt_masks),   "filtered (B&W masks)"),
    ])
    row3 = np.hstack([
        add_label(diff_image(orig_masks, filt_masks), "diff: green=kept  red=removed"),
        add_label(img, "original image"),
    ])
    cv2.imwrite(str(out_path), np.vstack([row1, row2, row3]))


# ── Processing ────────────────────────────────────────────────────────────────

def process_frame(frame_side_dir: Path, traj_dir: Path, cam_name: str,
                  frame_idx: str, iqr_factor: float, min_rel_tol: float,
                  min_component_ratio: float, gradient_fill: bool,
                  gradient_threshold: float, dry_run: bool, visualize: bool,
                  image_path: Path) -> dict:
    masks_dir = frame_side_dir / "masks"
    mask_files = sorted(masks_dir.glob("mask_*.png"))
    if not mask_files:
        return {}

    pointmap_path = find_pointmap(traj_dir, cam_name, frame_idx)
    if pointmap_path is None:
        return {}

    pointmap = np.load(str(pointmap_path))

    results = {"total": len(mask_files), "changed": 0, "removed": 0}
    orig_masks, filt_masks = [], []

    out_dir = frame_side_dir / "masks_filtered"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for mask_path in mask_files:
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue
        if mask_img.shape != pointmap.shape[:2]:
            mask_img = cv2.resize(mask_img,
                                  (pointmap.shape[1], pointmap.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        mask = mask_img > 127

        filtered = filter_mask_by_depth(mask, pointmap, iqr_factor, min_rel_tol,
                                        min_component_ratio,
                                        gradient_fill, gradient_threshold)

        orig_masks.append(mask)
        filt_masks.append(filtered)

        changed = not np.array_equal(mask, filtered)
        if changed:
            results["changed"] += 1
        if not filtered.any():
            results["removed"] += 1

        if not dry_run:
            out_path = out_dir / mask_path.name
            cv2.imwrite(str(out_path), filtered.astype(np.uint8) * 255)

    if visualize and not dry_run and image_path.exists():
        save_viz(image_path, orig_masks, filt_masks,
                 frame_side_dir / "depth_filter_viz.jpg")

    return results


def process_dir(base: Path, iqr_factor: float, min_rel_tol: float,
                min_component_ratio: float, gradient_fill: bool,
                gradient_threshold: float, dry_run: bool, visualize: bool):
    total_frames = total_masks = changed_masks = removed_masks = 0

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
                if not stereo_dir.exists():
                    continue
                for cam_dir in sorted(stereo_dir.iterdir()):
                    if not cam_dir.is_dir() or not cam_dir.name.endswith("-stereo"):
                        continue
                    cam_name = cam_dir.name
                    for frame_side_dir in sorted(cam_dir.iterdir()):
                        if not frame_side_dir.is_dir():
                            continue
                        m = re.match(r'^left_(\d+)$', frame_side_dir.name)
                        if not m:
                            continue
                        frame_idx = m.group(1)
                        image_path = cam_dir / f"left_{frame_idx}.png"

                        r = process_frame(frame_side_dir, traj_dir, cam_name,
                                          frame_idx, iqr_factor, min_rel_tol,
                                          min_component_ratio, gradient_fill,
                                          gradient_threshold, dry_run,
                                          visualize, image_path)
                        if not r:
                            continue

                        total_frames += 1
                        total_masks  += r["total"]
                        changed_masks += r["changed"]
                        removed_masks += r["removed"]

                        rel = frame_side_dir.relative_to(base)
                        print(f"  {rel}: {r['total']} masks, "
                              f"{r['changed']} changed, {r['removed']} removed")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Done.")
    print(f"  Frames:  {total_frames}")
    print(f"  Masks:   {total_masks}")
    print(f"  Changed: {changed_masks}  ({100*changed_masks/max(total_masks,1):.1f}%)")
    print(f"  Removed: {removed_masks}  ({100*removed_masks/max(total_masks,1):.1f}%)")
    if not dry_run:
        print(f"\n  Filtered masks saved to left_*/masks_filtered/")
        print(f"  Run Stage 2 with --use-filtered to use them.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Filter masks by depth to remove background contamination"
    )
    parser.add_argument("input_dir", help="droid_processed_ directory")
    parser.add_argument(
        "--iqr-factor", type=float, default=3.0,
        help="Depth threshold = max(iqr_factor * IQR, min_rel_tol * median). "
             "Lower = stricter filtering. Default: 3.0"
    )
    parser.add_argument(
        "--min-rel-tol", type=float, default=0.15,
        help="Minimum tolerance as fraction of median depth (e.g. 0.15 = 15%%). "
             "Prevents over-filtering thin/flat objects. Default: 0.15"
    )
    parser.add_argument(
        "--min-component-ratio", type=float, default=0.1,
        help="Keep all connected components with area >= this fraction of the largest. "
             "e.g. 0.1 keeps any component that is at least 10%% of the biggest. "
             "Preserves occluded object parts while dropping tiny noise. Default: 0.1"
    )
    parser.add_argument(
        "--gradient-fill", action="store_true",
        help="Also apply gradient-connectivity flood-fill to remove enclosed "
             "background holes (e.g. table visible through cup-handle hole) "
             "that depth-threshold filtering misses because FoundationStereo "
             "interpolated their depth to match the object."
    )
    parser.add_argument(
        "--gradient-threshold", type=float, default=0.05,
        help="Relative depth-gradient threshold for --gradient-fill. "
             "A pixel is a 'barrier' when its gradient / Z_median > this value. "
             "Lower = stricter (more barriers, more aggressive hole removal). "
             "Try 0.02–0.10. Default: 0.05"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing any files"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save depth_filter_viz.jpg (original vs filtered) per frame"
    )
    args = parser.parse_args()

    base = Path(args.input_dir)
    print(f"Filtering masks in {base}")
    print(f"  iqr_factor={args.iqr_factor}  min_rel_tol={args.min_rel_tol}  "
          f"min_component_ratio={args.min_component_ratio}")
    if args.gradient_fill:
        print(f"  gradient_fill=True  gradient_threshold={args.gradient_threshold}")
    if args.dry_run:
        print("  [DRY RUN]\n")

    process_dir(base, args.iqr_factor, args.min_rel_tol, args.min_component_ratio,
                args.gradient_fill, args.gradient_threshold,
                args.dry_run, args.visualize)


if __name__ == "__main__":
    main()
