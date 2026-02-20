#!/usr/bin/env python3
"""
Interactive manual annotation tool for DROID frames using SAM3.

For each left_NNNNNN frame in a trajectory's stereo directories, shows the
image and lets you draw bounding boxes. SAM3 generates a mask from each box.
Accept or discard masks, annotate multiple objects per frame.

Output: masks/mask_NNN_score_X.XXX.png per frame-side directory
(same format as automatic SAM3 segmentation - compatible with SAM3D pipeline)

Usage:
    python annotate_droid.py /path/to/droid_processed/ENV/success/TRAJ

Controls:
    Left-click + drag  Draw bounding box for SAM3
    A                  Accept current mask (saves it)
    R                  Redo - discard current box/mask, draw again
    N                  Next frame (done with this frame)
    S                  Skip frame (no masks saved)
    Q                  Quit

Run with sam3 conda env:
    conda run -n sam3 python annotate_droid.py <traj_dir>
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# SAM3 imports - must be run from Gaia-DreMa root or with sam3 on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Gaia-DreMa root
sys.path.insert(0, str(Path(__file__).parent.parent / "sam3"))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# =============================================================================
# SAM3 model (loaded once)
# =============================================================================

_processor = None

def get_processor(device: str) -> Sam3Processor:
    global _processor
    if _processor is None:
        print("Loading SAM3 model...")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model = build_sam3_image_model()
        _processor = Sam3Processor(model, device=device, confidence_threshold=0.1)
        print("SAM3 ready.")
    return _processor


def run_sam3_box(processor: Sam3Processor, image: np.ndarray, box_xyxy: np.ndarray):
    """
    Run SAM3 with a bounding box prompt.
    box_xyxy: [x0, y0, x1, y1] in pixel coords.
    Returns (mask H×W bool, score float) or (None, None).
    """
    h, w = image.shape[:2]
    pil = Image.fromarray(image)

    with torch.autocast("cuda" if processor.device == "cuda" else "cpu",
                        dtype=torch.bfloat16 if processor.device == "cuda" else torch.float32):
        state = processor.set_image(pil)

        # Convert pixel box → normalized [cx, cy, w_norm, h_norm]
        x0, y0, x1, y1 = box_xyxy
        cx = ((x0 + x1) / 2) / w
        cy = ((y0 + y1) / 2) / h
        bw = abs(x1 - x0) / w
        bh = abs(y1 - y0) / h
        box_norm = [cx, cy, bw, bh]

        state = processor.add_geometric_prompt(box_norm, label=True, state=state)

    masks = state.get("masks")
    scores = state.get("scores")

    if masks is None or len(masks) == 0:
        return None, None

    # Take highest-scoring mask
    scores_np = scores.float().cpu().numpy()
    best = int(scores_np.argmax())
    mask = masks[best].squeeze().float().cpu().numpy() > 0.5
    score = float(scores_np[best])
    return mask, score


# =============================================================================
# Interactive annotation window
# =============================================================================

class AnnotationSession:
    """Handles one frame's interactive annotation."""

    def __init__(self, image: np.ndarray, processor: Sam3Processor, existing_masks: int = 0):
        self.image = image
        self.processor = processor
        self.next_mask_idx = existing_masks  # continue from existing

        self.accepted_masks = []   # list of (mask, score)
        self.current_mask = None
        self.current_score = None
        self.current_box = None    # [x0, y0, x1, y1]

        self._drag_start = None
        self._rect_patch = None
        self._mask_im = None

        self.done = False   # True when user presses N (next frame)
        self.skip = False   # True when user presses S (skip frame)

    def _draw(self):
        self.ax.cla()
        self.ax.imshow(self.image)

        # Draw all accepted masks
        overlay = np.zeros((*self.image.shape[:2], 4), dtype=np.float32)
        np.random.seed(42)
        for i, (m, _) in enumerate(self.accepted_masks):
            color = np.random.rand(3)
            overlay[m, :3] = color
            overlay[m, 3] = 0.45

        if overlay.any():
            self.ax.imshow(overlay)

        # Draw current mask preview
        if self.current_mask is not None:
            preview = np.zeros((*self.image.shape[:2], 4), dtype=np.float32)
            preview[self.current_mask] = [1, 1, 0, 0.5]  # yellow
            self.ax.imshow(preview)

        # Draw current box
        if self.current_box is not None:
            x0, y0, x1, y1 = self.current_box
            rect = patches.Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0), abs(y1 - y0),
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            self.ax.add_patch(rect)

        n_acc = len(self.accepted_masks)
        title = (f"Masks: {n_acc} accepted | "
                 f"[A] accept  [R] redo  [N] next frame  [S] skip  [Q] quit")
        if self.current_score is not None:
            title = f"Score: {self.current_score:.3f} | " + title
        self.ax.set_title(title, fontsize=9)
        self.ax.axis('off')
        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self._drag_start = (event.xdata, event.ydata)

    def _on_release(self, event):
        if event.inaxes != self.ax or event.button != 1 or self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.xdata, event.ydata
        self._drag_start = None

        # Ignore tiny clicks (< 10px)
        if abs(x1 - x0) < 10 or abs(y1 - y0) < 10:
            return

        self.current_box = [x0, y0, x1, y1]
        self.current_mask = None
        self.current_score = None
        self._draw()

        # Run SAM3
        print(f"  Running SAM3 on box [{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]...")
        box = np.array([min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)])
        mask, score = run_sam3_box(self.processor, self.image, box)
        if mask is None:
            print("  No mask returned.")
        else:
            print(f"  Mask score: {score:.3f}")
            self.current_mask = mask
            self.current_score = score
        self._draw()

    def _on_key(self, event):
        key = event.key.lower() if event.key else ''

        if key == 'a':  # Accept
            if self.current_mask is not None:
                self.accepted_masks.append((self.current_mask, self.current_score))
                print(f"  Accepted mask {self.next_mask_idx + len(self.accepted_masks) - 1} "
                      f"(score {self.current_score:.3f})")
                self.current_mask = None
                self.current_score = None
                self.current_box = None
                self._draw()
            else:
                print("  Nothing to accept - draw a box first.")

        elif key == 'r':  # Redo
            self.current_mask = None
            self.current_score = None
            self.current_box = None
            self._draw()

        elif key == 'n':  # Next frame
            self.done = True
            plt.close(self.fig)

        elif key == 's':  # Skip frame
            self.skip = True
            self.done = True
            plt.close(self.fig)

        elif key == 'q':  # Quit
            self.skip = True
            self.done = True
            self._quit = True
            plt.close(self.fig)

    def run(self) -> list:
        """
        Open window, let user annotate. Returns list of (mask, score) pairs.
        Sets self.skip=True if frame should be skipped.
        """
        self._quit = False
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._draw()
        plt.tight_layout()
        plt.show()
        return self.accepted_masks, getattr(self, '_quit', False)


# =============================================================================
# Mask saving
# =============================================================================

def save_masks(masks_scores: list, masks_dir: Path, start_idx: int):
    """Save list of (mask, score) pairs to masks_dir/mask_NNN_score_X.XXX.png"""
    masks_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, (mask, score) in enumerate(masks_scores):
        idx = start_idx + i
        filename = f"mask_{idx:03d}_score_{score:.3f}.png"
        path = masks_dir / filename
        mask_uint8 = (mask.astype(np.uint8)) * 255
        cv2.imwrite(str(path), mask_uint8)
        saved.append(str(path))
        print(f"    Saved: {path.name}")
    return saved


def count_existing_masks(masks_dir: Path) -> int:
    if not masks_dir.exists():
        return 0
    return len(list(masks_dir.glob("mask_*.png")))


# =============================================================================
# Frame discovery
# =============================================================================

def find_frames(traj_dir: Path) -> list:
    """
    Find all left_NNNNNN frame-side directories in a trajectory's stereo dir.
    Returns list of (frame_side_dir, image_path).
    """
    frames = []
    stereo_dir = traj_dir / "stereo"
    if not stereo_dir.exists():
        return frames

    for cam_dir in sorted(stereo_dir.iterdir()):
        if not cam_dir.is_dir() or not cam_dir.name.endswith("-stereo"):
            continue
        # Only external cameras (serial starts with 2 or 3)
        serial = cam_dir.name.replace("-stereo", "")
        if not serial or serial[0] not in ('2', '3'):
            continue

        for frame_side_dir in sorted(cam_dir.iterdir()):
            if not frame_side_dir.is_dir():
                continue
            m = re.match(r'^left_(\d+)$', frame_side_dir.name)
            if not m:
                continue

            # Image is at stereo/<cam>-stereo/left_NNNNNN.png (sibling, not in subdir)
            frame_idx = m.group(1)
            image_path = cam_dir / f"left_{frame_idx}.png"
            if not image_path.exists():
                # Fall back: check inside dir
                image_path = frame_side_dir / "left.png"
            if not image_path.exists():
                continue

            frames.append((frame_side_dir, image_path))

    return frames


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Interactive SAM3 annotation for DROID frames")
    parser.add_argument("traj_dir", help="Path to trajectory directory (env/outcome/traj)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true",
                        help="Re-annotate frames that already have masks")
    args = parser.parse_args()

    traj_dir = Path(args.traj_dir)
    if not traj_dir.exists():
        print(f"ERROR: {traj_dir} not found")
        sys.exit(1)

    frames = find_frames(traj_dir)
    if not frames:
        print(f"No left_* frames found in {traj_dir}/stereo/")
        sys.exit(1)

    print(f"Found {len(frames)} frames to annotate in {traj_dir.name}")
    print("Controls: drag=draw box | A=accept | R=redo | N=next frame | S=skip | Q=quit\n")

    processor = get_processor(args.device)

    total_saved = 0
    quit_requested = False

    for i, (frame_side_dir, image_path) in enumerate(frames):
        rel = frame_side_dir.relative_to(traj_dir)
        masks_dir = frame_side_dir / "masks"
        existing = count_existing_masks(masks_dir)

        if existing > 0 and not args.force:
            print(f"[{i+1}/{len(frames)}] {rel}: skipping ({existing} masks exist, use --force to redo)")
            continue

        print(f"[{i+1}/{len(frames)}] {rel}  (image: {image_path.name})")

        image = np.array(Image.open(image_path).convert("RGB"))

        session = AnnotationSession(image, processor, existing_masks=existing)
        accepted, quit_requested = session.run()

        if quit_requested:
            print("Quitting.")
            break

        if session.skip:
            print(f"  Skipped.")
            continue

        if accepted:
            saved = save_masks(accepted, masks_dir, start_idx=existing)
            total_saved += len(saved)
            print(f"  Saved {len(saved)} mask(s).")
        else:
            print(f"  No masks saved.")

    print(f"\nDone. Total masks saved: {total_saved}")
    print(f"\nNext steps:")
    print(f"  1. rsync masks to cluster:")
    print(f"     rsync -av {traj_dir}/ cluster:/ivi/xfs/lschune/droid_processed/.../")
    print(f"  2. Run SAM3D reconstruction on cluster")


if __name__ == "__main__":
    main()
