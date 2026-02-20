#!/usr/bin/env python3
"""
Mesh Reprojection for DROID Processed Trajectories

Reprojects SAM3D mesh output onto camera images using calibration data.
Matches demo_bop.py transform chain exactly.

Usage:
    python reproject_mesh.py /path/to/camera_dir
    python reproject_mesh.py /path/to/camera_dir --mode wireframe --debug
"""

import argparse
import json
import re
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("Warning: trimesh not installed")


# =============================================================================
# PyTorch3D-style transforms (row-vector convention: p @ R.T + t)
# =============================================================================

# Coordinate conversion matrix (flips X and Y)
PYTORCH3D_ROTATION = np.array([
    [-1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0,  1]
], dtype=np.float64)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w,x,y,z] to rotation matrix.
    Matches PyTorch3D's quaternion_to_matrix exactly.
    """
    q = np.asarray(q, dtype=np.float64).flatten()
    q = q / np.linalg.norm(q)
    w, x, y, z = q[0], q[1], q[2], q[3]

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)


def transform_points_pt3d(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Transform points using PyTorch3D convention: p_new = p @ R + t

    PyTorch3D uses row vectors with right-multiplication: [p, 1] @ M
    where M[:3,:3] = R and M[3,:3] = t. So transform_points = p @ R + t.
    """
    return points @ R + t


def build_pose_matrix(scale: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build pose transform matching Transform3d().scale(s).rotate(R).translate(t)

    Returns (R_combined, t) where R_combined = diag(scale) @ R
    For transform_points: p_new = p @ R_combined.T + t
    """
    S = np.diag(scale)
    R_combined = S @ rotation  # Scale then rotate
    return R_combined, translation


def compose_transforms(R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compose two transforms: first apply (R1, t1), then (R2, t2)
    Matches Transform3d.compose() behavior.

    For row vectors: p @ R1 + t1, then result @ R2 + t2
    Combined: p @ (R1 @ R2) + (t1 @ R2 + t2)
    """
    R_combined = R1 @ R2
    t_combined = t1 @ R2 + t2
    return R_combined, t_combined


def transform_mesh_to_opencv(
    vertices: np.ndarray,
    quaternion: np.ndarray,
    translation: np.ndarray,
    scale: np.ndarray,
    debug: bool = False
) -> np.ndarray:
    """
    Transform mesh vertices from local coordinates to OpenCV camera coordinates.
    Matches demo_bop.py transform chain exactly.

    Chain:
    1. Apply Rx(-90) flip (Gaussian alignment)
    2. Apply pose (scale, rotate, translate)
    3. Apply OpenCV conversion
    """
    # Step 1: Pose transform - scale, rotate, translate
    R_pose = quaternion_to_matrix(quaternion)
    R_pose_combined, t_pose = build_pose_matrix(scale, R_pose, translation)

    # Step 2: Gaussian alignment - Rx(-90)
    R_flip = R.from_euler('x', -90, degrees=True).as_matrix()

    # Compose: flip FIRST, then pose (matching tfm_flip.compose(tfm))
    R_after_flip, t_after_flip = compose_transforms(R_flip, np.zeros(3), R_pose_combined, t_pose)

    # Step 3: Transform vertices
    vertices_transformed = transform_points_pt3d(vertices, R_after_flip, t_after_flip)

    # Step 4: OpenCV conversion
    R_final = np.linalg.inv(PYTORCH3D_ROTATION)  # = PYTORCH3D_ROTATION (self-inverse)
    vertices_opencv = transform_points_pt3d(vertices_transformed, R_final, np.zeros(3))

    if debug:
        print(f"  Transform chain:")
        print(f"    R_pose:\n{R_pose}")
        print(f"    Scale: {scale}")
        print(f"    Translation: {translation}")
        print(f"    After flip+pose - X: [{vertices_transformed[:, 0].min():.3f}, {vertices_transformed[:, 0].max():.3f}]")
        print(f"    After flip+pose - Y: [{vertices_transformed[:, 1].min():.3f}, {vertices_transformed[:, 1].max():.3f}]")
        print(f"    After flip+pose - Z: [{vertices_transformed[:, 2].min():.3f}, {vertices_transformed[:, 2].max():.3f}]")
        print(f"    OpenCV - X: [{vertices_opencv[:, 0].min():.3f}, {vertices_opencv[:, 0].max():.3f}]")
        print(f"    OpenCV - Y: [{vertices_opencv[:, 1].min():.3f}, {vertices_opencv[:, 1].max():.3f}]")
        print(f"    OpenCV - Z: [{vertices_opencv[:, 2].min():.3f}, {vertices_opencv[:, 2].max():.3f}]")

    return vertices_opencv


# =============================================================================
# Loading Functions
# =============================================================================

def load_intrinsics(path: Path) -> np.ndarray:
    """Load camera intrinsics from JSON."""
    with open(path) as f:
        data = json.load(f)
    if "K" in data:
        return np.array(data["K"], dtype=np.float64)
    return np.array([
        [data["fx"], 0, data["cx"]],
        [0, data["fy"], data["cy"]],
        [0, 0, 1]
    ], dtype=np.float64)


def load_pose(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SAM3D pose from JSON."""
    with open(path) as f:
        data = json.load(f)
    return (
        np.array(data["rotation"], dtype=np.float64).flatten(),
        np.array(data["translation"], dtype=np.float64).flatten(),
        np.array(data["scale"], dtype=np.float64).flatten()
    )


def find_frame_image(cam_dir: Path) -> Optional[Path]:
    """Find frame image in camera directory."""
    for pattern in ["frame_*.png", "frame_*.jpg", "*.png", "*.jpg"]:
        matches = [m for m in cam_dir.glob(pattern) if "reprojection" not in m.name.lower()]
        if matches:
            return sorted(matches)[0]
    return None


# =============================================================================
# Projection and Rendering
# =============================================================================

def project_to_2d(points_3d: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points (OpenCV coords) to 2D pixels."""
    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    valid = Z > 0.01
    Z_safe = np.where(Z > 0.01, Z, 0.01)

    u = K[0, 0] * X / Z_safe + K[0, 2]
    v = K[1, 1] * Y / Z_safe + K[1, 2]

    return np.stack([u, v], axis=-1), valid


def render_bbox_overlay(
    image: np.ndarray,
    vertices: np.ndarray,
    K: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2
) -> np.ndarray:
    """Render 3D bounding box on image."""
    viz = image.copy()
    h, w = viz.shape[:2]

    # Get bounding box corners
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    corners = np.array([
        [mins[0], mins[1], mins[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], maxs[1], mins[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], maxs[2]],
        [mins[0], maxs[1], maxs[2]],
    ])

    corners_2d, valid = project_to_2d(corners, K)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
    ]

    for i, j in edges:
        if valid[i] and valid[j]:
            pt1 = tuple(corners_2d[i].astype(int))
            pt2 = tuple(corners_2d[j].astype(int))
            if (-w < pt1[0] < 2*w and -h < pt1[1] < 2*h and
                -w < pt2[0] < 2*w and -h < pt2[1] < 2*h):
                cv2.line(viz, pt1, pt2, color, thickness)

    return viz


def render_wireframe_overlay(
    image: np.ndarray,
    mesh: "trimesh.Trimesh",
    vertices_cam: np.ndarray,
    K: np.ndarray,
    color: Tuple[int, int, int]
) -> np.ndarray:
    """Render mesh wireframe on image."""
    viz = image.copy()
    h, w = viz.shape[:2]

    vertices_2d, valid = project_to_2d(vertices_cam, K)
    edges = mesh.edges_unique if hasattr(mesh, 'edges_unique') else mesh.edges

    for i, j in edges:
        if valid[i] and valid[j]:
            pt1 = tuple(vertices_2d[i].astype(int))
            pt2 = tuple(vertices_2d[j].astype(int))
            if (-w < pt1[0] < 2*w and -h < pt1[1] < 2*h and
                -w < pt2[0] < 2*w and -h < pt2[1] < 2*h):
                cv2.line(viz, pt1, pt2, color, 1)

    return viz


# =============================================================================
# Main Processing
# =============================================================================

def load_intrinsics_npy(path: Path) -> np.ndarray:
    """Load camera intrinsics from numpy file."""
    K = np.load(str(path))
    return K.astype(np.float64)


def load_visible_objects(cam_dir: Path, pointmap: bool = False) -> Optional[set]:
    """Load visible_objects.json filter if it exists. Returns set of allowed indices or None."""
    suffix = "_pointmap" if pointmap else ""
    filter_path = cam_dir / f"visible_objects{suffix}.json"
    if not filter_path.exists():
        return None
    try:
        with open(filter_path) as f:
            data = json.load(f)
        return set(data.get("visible_object_indices", []))
    except Exception:
        return None


def process_camera_directory(
    cam_dir: Path,
    mode: str = "bbox",
    output_name: str = "reprojection.png",
    debug: bool = False,
    pointmap: bool = False,
    intrinsics_override: Optional[Path] = None,
    image_override: Optional[Path] = None,
    use_filter: bool = False
) -> Optional[Path]:
    """Process camera directory and create reprojection visualization."""
    if not HAS_TRIMESH:
        print("ERROR: trimesh required")
        return None

    cam_dir = Path(cam_dir)
    objects_dir = cam_dir / ("objects_pointmap" if pointmap else "objects")

    if not objects_dir.exists():
        print(f"ERROR: {objects_dir.name}/ not found in {cam_dir}")
        return None

    # Load intrinsics
    if intrinsics_override:
        intrinsics_path = Path(intrinsics_override)
    else:
        intrinsics_path = cam_dir / "intrinsics.json"

    if not intrinsics_path.exists():
        print(f"ERROR: intrinsics not found at {intrinsics_path}")
        return None

    # Find frame image
    if image_override:
        frame_path = Path(image_override)
    else:
        frame_path = find_frame_image(cam_dir)

    if not frame_path or not frame_path.exists():
        print(f"ERROR: No frame image found")
        return None

    # Load intrinsics (support both JSON and numpy)
    if intrinsics_path.suffix == '.npy':
        K = load_intrinsics_npy(intrinsics_path)
    else:
        K = load_intrinsics(intrinsics_path)

    if debug:
        print(f"Camera: {cam_dir}")
        print(f"Frame: {frame_path.name}")
        print(f"Intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    image = cv2.imread(str(frame_path))
    if image is None:
        print(f"ERROR: Could not load {frame_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    viz = image.copy()

    # Load filter if requested
    visible_set = None
    if use_filter:
        visible_set = load_visible_objects(cam_dir, pointmap)
        if visible_set is not None and debug:
            print(f"  Filter: {len(visible_set)} visible objects")

    # Find meshes
    mesh_files = sorted(objects_dir.glob("*.glb")) or sorted(objects_dir.glob("*.obj"))
    if not mesh_files:
        print(f"WARNING: No meshes in {objects_dir}")
        output_path = cam_dir / output_name
        cv2.imwrite(str(output_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        return output_path

    np.random.seed(42)
    colors = np.random.randint(100, 255, size=(len(mesh_files), 3))

    processed = 0
    filtered = 0
    for i, mesh_file in enumerate(mesh_files):
        pose_file = mesh_file.parent / f"{mesh_file.stem}_pose.json"
        if not pose_file.exists():
            pose_file = mesh_file.with_suffix(".json")
        if not pose_file.exists():
            continue

        # Apply filter: extract object index from filename (object_NNN.glb)
        if visible_set is not None:
            m = re.match(r'object_(\d+)', mesh_file.stem)
            if m:
                obj_idx = int(m.group(1))
                if obj_idx not in visible_set:
                    filtered += 1
                    continue

        try:
            mesh = trimesh.load(str(mesh_file), force='mesh')
            quat, trans, scale = load_pose(pose_file)

            if debug:
                print(f"\n  {mesh_file.name}: {len(mesh.vertices)} vertices")

            # Transform to OpenCV camera coordinates
            vertices_cam = transform_mesh_to_opencv(
                mesh.vertices, quat, trans, scale, debug=debug
            )

            color = tuple(colors[i].tolist())

            if mode == "bbox":
                viz = render_bbox_overlay(viz, vertices_cam, K, color)
            elif mode == "wireframe":
                viz = render_wireframe_overlay(viz, mesh, vertices_cam, K, color)
            elif mode == "vertices":
                pts_2d, valid = project_to_2d(vertices_cam, K)
                for pt, v in zip(pts_2d, valid):
                    if v and 0 <= pt[0] < viz.shape[1] and 0 <= pt[1] < viz.shape[0]:
                        cv2.circle(viz, tuple(pt.astype(int)), 1, color, -1)

            processed += 1

        except Exception as e:
            print(f"  Error with {mesh_file.name}: {e}")
    
    if pointmap:
        output_name = f"reprojection_pointmap_{mode}.png"
        output_path = cam_dir / output_name
    else:
        output_name = f"reprojection_{mode}.png"
        output_path = cam_dir / output_name

    cv2.imwrite(str(output_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
    filter_msg = f" ({filtered} filtered)" if filtered > 0 else ""
    print(f"[OK] {cam_dir.name}: {processed} objects{filter_msg} -> {output_path.name}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Reproject SAM3D meshes onto image")
    parser.add_argument("camera_dir", help="Camera directory path")
    parser.add_argument("--mode", default="bbox", choices=["bbox", "wireframe", "vertices"])
    parser.add_argument("--output", default="reprojection.png")
    parser.add_argument("--pointmap", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--intrinsics", default=None, help="Path to intrinsics file (.json or .npy)")
    parser.add_argument("--image", default=None, help="Path to frame image")
    parser.add_argument("--filter", action="store_true",
                        help="Only reproject objects listed in visible_objects.json")
    args = parser.parse_args()

    result = process_camera_directory(
        Path(args.camera_dir), args.mode, args.output, args.debug, args.pointmap,
        intrinsics_override=Path(args.intrinsics) if args.intrinsics else None,
        image_override=Path(args.image) if args.image else None,
        use_filter=args.filter,
    )
    if result is None:
        exit(1)


if __name__ == "__main__":
    main()
