#!/usr/bin/env python3
"""
SAM 3D Objects: Image to 3D Object Generation

Reconstructs 3D objects (mesh + Gaussian splats) from a single image.
Can use SAM3 for automatic segmentation or load pre-computed masks.

Pipeline:
1. (Optional) SAM3 segmentation to get object masks
2. SAM 3D Objects reconstruction for each masked object
3. Export GLB meshes and Gaussian splat PLY files

Requirements:
- SAM 3D Objects installed with checkpoints (≥32GB VRAM)
- (Optional) SAM3 for automatic segmentation

Usage:
    # With automatic SAM3 segmentation
    python image_to_3d_objects.py --image /path/to/image.jpg --output ./output

    # With pre-computed masks directory
    python image_to_3d_objects.py --image /path/to/image.jpg --masks /path/to/masks/ --output ./output

    # From video frame
    python image_to_3d_objects.py --video /path/to/video.mp4 --frame 0 --output ./output

    # Segmentation only (skip 3D reconstruction)
    python image_to_3d_objects.py --image /path/to/image.jpg --output ./output --segment-only
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# ============================================================================
# Path Configuration - Update these for your setup
# ============================================================================

# SAM3 path (for segmentation)
SAM3_PATH = Path(__file__).parent.parent / "sam3"

# SAM 3D Objects path - check cluster path first, then local
_CLUSTER_SAM3D_PATH = Path("/ivi/xfs/lschune/sam-3d-objects")
_LOCAL_SAM3D_PATH = Path(__file__).resolve().parent.parent.parent / "DreMa" / "third_party" / "sam-3d-objects"
SAM3D_OBJECTS_PATH = _CLUSTER_SAM3D_PATH if _CLUSTER_SAM3D_PATH.exists() else _LOCAL_SAM3D_PATH

# SAM 3D Objects checkpoint configuration
SAM3D_CHECKPOINT_TAG = "hf"

# ============================================================================


def setup_paths():
    """Add required paths to sys.path."""
    if SAM3_PATH.exists():
        sys.path.insert(0, str(SAM3_PATH))
        print(f"Added SAM3 path: {SAM3_PATH}")

    if SAM3D_OBJECTS_PATH.exists():
        # Add notebook directory where inference.py lives
        notebook_path = SAM3D_OBJECTS_PATH / "notebook"
        if notebook_path.exists():
            sys.path.insert(0, str(notebook_path))
            print(f"Added SAM3D Objects path: {notebook_path}")

        # Also add main path for imports
        sys.path.insert(0, str(SAM3D_OBJECTS_PATH))


setup_paths()


# ============================================================================
# SAM3 Segmentation Functions
# ============================================================================

def segment_with_sam3(
    image: Image.Image,
    prompt: str = "object",
    min_score: float = 0.3,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Segment objects in image using SAM3.

    Args:
        image: PIL Image (RGB)
        prompt: Text prompt for detection
        min_score: Minimum confidence threshold

    Returns:
        masks: List of binary masks (H, W)
        scores: Confidence scores per mask
    """
    try:
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        raise ImportError(
            f"SAM3 not available. Install it or provide pre-computed masks.\n"
            f"SAM3 path: {SAM3_PATH}\nError: {e}"
        )

    print("Building SAM3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    print(f"Segmenting with prompt: '{prompt}'")
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=prompt)

    # Extract results
    masks = output["masks"]
    scores = output["scores"]

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    # Ensure binary
    if masks.dtype != bool:
        masks = masks > 0.5

    # Filter by score
    keep = scores >= min_score
    masks = masks[keep]
    scores = scores[keep]

    print(f"Found {len(masks)} objects (score >= {min_score})")

    return list(masks), scores


# ============================================================================
# Mask Loading Functions
# ============================================================================

def load_masks_from_directory(masks_dir: Path) -> Tuple[List[np.ndarray], List[dict]]:
    """Load masks from a directory of PNG files."""
    masks = []
    infos = []

    mask_files = sorted(masks_dir.glob("*.png"))
    if not mask_files:
        mask_files = sorted(masks_dir.glob("*.jpg"))

    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # FIX: Check for any value > 0, not just > 127
            # This handles masks that are [0, 1] AND masks that are [0, 255]
            binary_mask = mask > 0  
            
            # Double check we didn't just load an empty file
            if not np.any(binary_mask):
                print(f"Warning: Mask {mask_file.name} is empty (all zeros). Skipping.")
                continue

            masks.append(binary_mask)
            infos.append({"filename": mask_file.name})

    return masks, infos


def extract_frame_from_video(video_path: str, frame_idx: int = 0) -> np.ndarray:
    """Extract frame from video as RGB numpy array."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {frame_idx}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ============================================================================
# SAM 3D Objects Functions
# ============================================================================

def load_sam3d_inference(checkpoint_tag: str = "hf", compile: bool = False):
    """Load SAM 3D Objects inference model."""
    try:
        from inference import Inference
    except ImportError as e:
        raise ImportError(
            f"SAM 3D Objects not available at {SAM3D_OBJECTS_PATH}\n"
            f"Make sure to clone and install it. Error: {e}"
        )

    config_path = SAM3D_OBJECTS_PATH / "checkpoints" / checkpoint_tag / "pipeline.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"SAM 3D Objects config not found at: {config_path}\n"
            f"Download checkpoints with: hf download facebook/sam-3d-objects"
        )

    print(f"Loading SAM 3D Objects from: {config_path}")
    return Inference(str(config_path), compile=compile)


def reconstruct_3d(
    inference,
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 42,
    pointmap: Optional[np.ndarray] = None,
) -> dict:
    """
    Reconstruct 3D object from image and mask.

    Args:
        inference: SAM 3D Objects Inference instance
        image: RGB image (H, W, 3) uint8
        mask: Binary mask (H, W)
        seed: Random seed for reproducibility
        pointmap: Optional XYZ pointmap (H, W, 3) for depth-guided reconstruction

    Returns:
        Dictionary with 'glb', 'gs', 'rotation', 'translation', 'scale', etc.
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Ensure mask matches image size
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # Resize pointmap if needed
    if pointmap is not None and pointmap.shape[:2] != image.shape[:2]:
        # Resize each channel separately with INTER_NEAREST to preserve values
        pointmap_resized = np.zeros((image.shape[0], image.shape[1], 3), dtype=pointmap.dtype)
        for c in range(3):
            pointmap_resized[:, :, c] = cv2.resize(
                pointmap[:, :, c],
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        pointmap = pointmap_resized

    # SAM 3D Objects expects mask as boolean or 0/1 values (NOT 0/255)
    # The inference.merge_mask_to_rgba() will multiply by 255 internally
    mask_bool = mask > 0

    # Debug: Check mask is not empty
    if not np.any(mask_bool):
        raise ValueError("Mask is empty (all zeros). Cannot reconstruct 3D object.")

    print(f"  Mask stats: shape={mask_bool.shape}, pixels={np.sum(mask_bool)}, "
          f"coverage={100*np.mean(mask_bool):.1f}%")

    if pointmap is not None:
        pointmap = pointmap.copy()
        # Mark invalid regions as NaN instead of 0.
        # depth2xyzmap sets [0,0,0] for invalid depth, but SAM3D would treat
        # these as real points at the camera origin, corrupting scale/shift
        # statistics. SAM3D uses nanmedian/nanmean to properly ignore NaN.
        invalid = pointmap[:, :, 2] <= 0  # Z<=0 means invalid depth
        pointmap[invalid] = float('nan')
        # Replace any remaining non-finite values (inf from stereo edges) with NaN
        pointmap = np.where(np.isfinite(pointmap), pointmap, float('nan'))
        # Convert from OpenCV camera coords (X-right, Y-down, Z-forward) to
        # PyTorch3D camera coords (X-left, Y-up, Z-forward).
        # SAM3D's internal depth model applies this conversion via
        # camera_to_pytorch3d_camera(), but user-provided pointmaps bypass it.
        pointmap[:, :, 0] *= -1  # X: right -> left
        pointmap[:, :, 1] *= -1  # Y: down -> up
        valid_count = int(np.isfinite(pointmap[:, :, 2]).sum())
        print(f"  Using pointmap: shape={pointmap.shape}, valid={valid_count}/{pointmap.shape[0]*pointmap.shape[1]} pixels")
        # Convert numpy pointmap to torch tensor (SAM3D expects torch tensor)
        import torch
        pointmap = torch.from_numpy(pointmap).float()

    # Run inference with RGB image, boolean mask, and optional pointmap
    output = inference(image, mask_bool, seed=seed, pointmap=pointmap)

    return output


# ============================================================================
# Output Functions
# ============================================================================

def save_outputs(
    output: dict,
    output_dir: Path,
    object_name: str,
    save_glb: bool = True,
    save_ply: bool = True,
) -> dict:
    """
    Save 3D reconstruction outputs.

    Returns:
        Dictionary with paths to saved files
    """
    saved = {}

    # Save GLB mesh
    if save_glb and "glb" in output and output["glb"] is not None:
        glb_path = output_dir / f"{object_name}.glb"
        output["glb"].export(str(glb_path))
        saved["glb"] = str(glb_path)
        print(f"  Saved mesh: {glb_path}")

    # Save Gaussian Splat PLY
    if save_ply and "gs" in output and output["gs"] is not None:
        ply_path = output_dir / f"{object_name}.ply"
        output["gs"].save_ply(str(ply_path))
        saved["ply"] = str(ply_path)
        print(f"  Saved gaussian: {ply_path}")

    # Save 4k Gaussian if available
    if save_ply and "gs_4" in output and output["gs_4"] is not None:
        ply_4k_path = output_dir / f"{object_name}_4k.ply"
        output["gs_4"].save_ply(str(ply_4k_path))
        saved["ply_4k"] = str(ply_4k_path)
        print(f"  Saved gaussian (4k): {ply_4k_path}")

    # Save pose information
    pose_path = output_dir / f"{object_name}_pose.json"
    pose_data = {}

    if "rotation" in output:
        rot = output["rotation"]
        if hasattr(rot, "cpu"):
            rot = rot.cpu().numpy()
        pose_data["rotation"] = rot.tolist() if hasattr(rot, "tolist") else list(rot)

    if "translation" in output:
        trans = output["translation"]
        if hasattr(trans, "cpu"):
            trans = trans.cpu().numpy()
        pose_data["translation"] = trans.tolist() if hasattr(trans, "tolist") else list(trans)

    if "scale" in output:
        scale = output["scale"]
        if hasattr(scale, "cpu"):
            scale = scale.cpu().numpy()
        pose_data["scale"] = scale.tolist() if hasattr(scale, "tolist") else list(scale)

    if pose_data:
        with open(pose_path, "w") as f:
            json.dump(pose_data, f, indent=2)
        saved["pose"] = str(pose_path)

    return saved


def create_segmentation_visualization(
    image: np.ndarray,
    masks: List[np.ndarray],
    scores: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Create visualization with colored mask overlays."""
    viz = image.copy()

    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(len(masks), 3))

    for i, mask in enumerate(masks):
        color = colors[i]

        # Resize mask if needed
        if mask.shape[:2] != viz.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (viz.shape[1], viz.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ) > 0

        # Overlay
        overlay = np.zeros_like(viz)
        overlay[mask] = color
        viz = (viz * (1 - alpha * mask[:, :, None]) +
               overlay * alpha * mask[:, :, None]).astype(np.uint8)

        # Label
        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            label = f"{i}"
            if scores is not None:
                label += f": {scores[i]:.2f}"
            cv2.putText(viz, label, (cx - 10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return viz


def render_scene_video(
    outputs: List[dict],
    output_path: Path,
    num_frames: int = 120,
    resolution: int = 512,
):
    """Render 360-degree video of combined scene."""
    try:
        from inference import make_scene, ready_gaussian_for_video_rendering, render_video
        import imageio
    except ImportError as e:
        print(f"Could not render video: {e}")
        return

    # Combine all objects into scene
    scene_gs = make_scene(*outputs)
    scene_gs = ready_gaussian_for_video_rendering(scene_gs)

    # Render
    video_frames = render_video(
        scene_gs,
        resolution=resolution,
        num_frames=num_frames,
    )

    # Save
    imageio.mimwrite(str(output_path), video_frames, fps=30)
    print(f"Saved scene video: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D objects from image using SAM 3D Objects"
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to input image")
    input_group.add_argument("--video", type=str, help="Path to video (use with --frame)")

    parser.add_argument("--frame", type=int, default=0, help="Frame index for video input")
    parser.add_argument("--masks", type=str, help="Path to pre-computed masks directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")

    # SAM3 segmentation options
    parser.add_argument("--prompt", type=str, default="object",
                       help="Text prompt for SAM3 segmentation")
    parser.add_argument("--min-score", type=float, default=0.3,
                       help="Minimum confidence for SAM3")

    # SAM 3D Objects options
    parser.add_argument("--checkpoint", type=str, default="hf",
                       help="SAM 3D Objects checkpoint tag")
    parser.add_argument("--compile", action="store_true",
                       help="Compile model (faster inference, slower startup)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output options
    parser.add_argument("--segment-only", action="store_true",
                       help="Only run segmentation, skip 3D reconstruction")
    parser.add_argument("--no-glb", action="store_true", help="Skip GLB mesh export")
    parser.add_argument("--no-ply", action="store_true", help="Skip Gaussian PLY export")
    parser.add_argument("--render-video", action="store_true",
                       help="Render 360-degree video of scene")
    parser.add_argument("--save-masks", action="store_true",
                       help="Save segmentation masks as PNGs")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    if args.image:
        print(f"Loading image: {args.image}")
        image = np.array(Image.open(args.image).convert("RGB"))
        image_name = Path(args.image).stem
    else:
        print(f"Extracting frame {args.frame} from: {args.video}")
        image = extract_frame_from_video(args.video, args.frame)
        image_name = f"{Path(args.video).stem}_frame{args.frame:06d}"

    # Save input
    input_path = output_dir / f"{image_name}_input.png"
    Image.fromarray(image).save(input_path)
    print(f"Saved input: {input_path}")

    # Get masks
    if args.masks:
        print(f"\nLoading masks from: {args.masks}")
        masks, mask_infos = load_masks_from_directory(Path(args.masks))
        scores = np.ones(len(masks))
    else:
        print(f"\n--- SAM3 Segmentation ---")
        pil_image = Image.fromarray(image)
        masks, scores = segment_with_sam3(pil_image, args.prompt, args.min_score)

    if len(masks) == 0:
        print("No objects found. Try lowering --min-score or different --prompt")
        return

    print(f"\nFound {len(masks)} objects")

    # Save masks if requested
    if args.save_masks:
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_path = masks_dir / f"object_{i:03d}_score_{score:.3f}.png"
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
        print(f"Saved masks to: {masks_dir}")

    # Save segmentation visualization
    viz = create_segmentation_visualization(image, masks, scores)
    viz_path = output_dir / f"{image_name}_segmentation.png"
    cv2.imwrite(str(viz_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization: {viz_path}")

    # Stop here if segment-only
    if args.segment_only:
        summary = {
            "image_name": image_name,
            "num_objects": len(masks),
            "scores": [float(s) for s in scores],
            "prompt": args.prompt,
        }
        with open(output_dir / "segmentation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\nSegmentation complete (--segment-only)")
        return

    # Load SAM 3D Objects
    print(f"\n--- SAM 3D Objects Reconstruction ---")
    inference = load_sam3d_inference(args.checkpoint, args.compile)

    # Process each object
    objects_dir = output_dir / "objects"
    objects_dir.mkdir(exist_ok=True)

    results = []
    successful_outputs = []

    for i, (mask, score) in enumerate(zip(masks, scores)):
        print(f"\nObject {i + 1}/{len(masks)} (score: {score:.3f})")

        object_name = f"object_{i:03d}"

        try:
            output = reconstruct_3d(inference, image, mask, seed=args.seed)

            saved = save_outputs(
                output,
                objects_dir,
                object_name,
                save_glb=not args.no_glb,
                save_ply=not args.no_ply,
            )

            results.append({
                "object_id": i,
                "score": float(score),
                "status": "success",
                "files": saved,
            })

            successful_outputs.append(output)

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "object_id": i,
                "score": float(score),
                "status": "failed",
                "error": str(e),
            })

    # Render scene video if requested
    if args.render_video and len(successful_outputs) > 0:
        print("\n--- Rendering Scene Video ---")
        video_path = output_dir / f"{image_name}_scene.mp4"
        render_scene_video(successful_outputs, video_path)

    # Save summary
    summary = {
        "image_name": image_name,
        "input_image": str(input_path),
        "num_objects": len(masks),
        "num_successful": sum(1 for r in results if r["status"] == "success"),
        "prompt": args.prompt if not args.masks else None,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "objects": results,
    }

    with open(output_dir / "reconstruction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'=' * 50}")
    print("Summary:")
    print(f"  Total objects: {len(masks)}")
    print(f"  Successful: {summary['num_successful']}")
    print(f"  Failed: {len(masks) - summary['num_successful']}")
    print(f"  Output: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
