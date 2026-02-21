#!/usr/bin/env python3
"""
Interactive web-based annotation tool for DROID frames using SAM3.

Serves a Flask app - open in browser via SSH port forwarding:
    ssh -L 5000:localhost:5000 user@server
    conda run -n sam3 python annotate_droid.py ~/droid_processed_
    # open http://localhost:5000

Controls (in browser):
    Click + drag  Draw bounding box
    Accept        Save current mask, draw another object
    Redo          Discard current mask, redraw
    Next          Done with this frame, move on
    Skip          Skip frame without saving any masks
"""

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "sam3"))

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# =============================================================================
# SAM3
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
    h, w = image.shape[:2]
    pil = Image.fromarray(image)
    dtype = torch.bfloat16 if processor.device == "cuda" else torch.float32
    with torch.autocast(processor.device, dtype=dtype):
        state = processor.set_image(pil)
        x0, y0, x1, y1 = box_xyxy
        box_norm = [((x0+x1)/2)/w, ((y0+y1)/2)/h, abs(x1-x0)/w, abs(y1-y0)/h]
        state = processor.add_geometric_prompt(box_norm, label=True, state=state)
    masks = state.get("masks")
    scores = state.get("scores")
    if masks is None or len(masks) == 0:
        return None, None
    scores_np = scores.float().cpu().numpy()
    best = int(scores_np.argmax())
    mask = masks[best].squeeze().float().cpu().numpy() > 0.5
    return mask, float(scores_np[best])


# =============================================================================
# Frame discovery
# =============================================================================

def find_pointmap(traj_dir: Path, cam_name: str, frame_idx: str) -> Path:
    depth_dir = traj_dir / "depth"
    for p in [
        depth_dir / cam_name / f"frame_{frame_idx}" / "pointmap.npy",
        depth_dir / f"frame_{frame_idx}" / "pointmap.npy",
    ]:
        if p.exists():
            return p
    return None


def find_frames(traj_dir: Path) -> list:
    frames = []
    stereo_dir = traj_dir / "stereo"
    if not stereo_dir.exists():
        return frames
    for cam_dir in sorted(stereo_dir.iterdir()):
        if not cam_dir.is_dir() or not cam_dir.name.endswith("-stereo"):
            continue
        serial = cam_dir.name.replace("-stereo", "")
        if not serial or serial[0] not in ('2', '3'):
            continue
        for frame_side_dir in sorted(cam_dir.iterdir()):
            if not frame_side_dir.is_dir():
                continue
            m = re.match(r'^left_(\d+)$', frame_side_dir.name)
            if not m:
                continue
            frame_idx = m.group(1)
            pointmap_path = find_pointmap(traj_dir, cam_dir.name, frame_idx)
            if pointmap_path is None:
                continue
            image_path = cam_dir / f"left_{frame_idx}.png"
            if not image_path.exists():
                image_path = frame_side_dir / "left.png"
            if not image_path.exists():
                continue
            frames.append((frame_side_dir, image_path, pointmap_path))
    return frames


def get_env(frame_side_dir: Path, base: Path) -> str:
    try:
        return frame_side_dir.relative_to(base).parts[0]
    except Exception:
        return ""


def find_all_frames(base: Path, skip_envs: set = None) -> list:
    all_frames = []
    for env_dir in sorted(base.iterdir()):
        if not env_dir.is_dir() or env_dir.name.startswith('.'):
            continue
        if skip_envs and env_dir.name in skip_envs:
            continue
        for outcome_dir in sorted(env_dir.iterdir()):
            if not outcome_dir.is_dir():
                continue
            for traj_dir in sorted(outcome_dir.iterdir()):
                if not traj_dir.is_dir():
                    continue
                all_frames.extend(find_frames(traj_dir))
    return all_frames


# =============================================================================
# Mask helpers
# =============================================================================

def clear_masks(masks_dir: Path):
    if masks_dir.exists():
        for f in masks_dir.glob("mask_*.png"):
            f.unlink()


def save_masks(masks_scores: list, masks_dir: Path):
    masks_dir.mkdir(parents=True, exist_ok=True)
    for i, (mask, score) in enumerate(masks_scores):
        path = masks_dir / f"mask_{i:03d}_score_{score:.3f}.png"
        cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
        print(f"  Saved: {path.name}")


def image_to_b64(img_rgb: np.ndarray) -> str:
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def overlay_masks(image: np.ndarray, accepted: list, current_mask=None) -> str:
    viz = image.copy().astype(np.float32)
    np.random.seed(42)
    for i, (m, _) in enumerate(accepted):
        color = (np.random.rand(3) * 155 + 100)
        viz[m] = viz[m] * 0.5 + color * 0.5
    if current_mask is not None:
        viz[current_mask] = viz[current_mask] * 0.4 + np.array([255, 255, 0]) * 0.6
    return image_to_b64(np.clip(viz, 0, 255).astype(np.uint8))


# =============================================================================
# Flask app
# =============================================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>DROID Annotator</title>
<style>
  body { margin: 0; background: #1a1a1a; color: #eee; font-family: monospace; display: flex; flex-direction: column; align-items: center; }
  #header { padding: 10px; font-size: 13px; color: #aaa; width: 100%; box-sizing: border-box; }
  #progress { color: #7af; font-weight: bold; }
  #frame-info { color: #fa7; margin-left: 20px; }
  #canvas-wrap { position: relative; cursor: crosshair; }
  #canvas { display: block; }
  #overlay { position: absolute; top: 0; left: 0; pointer-events: none; }
  #controls { padding: 10px; display: flex; gap: 10px; align-items: center; }
  button { padding: 8px 20px; font-size: 14px; border: none; border-radius: 4px; cursor: pointer; font-family: monospace; }
  #btn-accept { background: #2a7; color: #fff; }
  #btn-redo   { background: #a72; color: #fff; }
  #btn-next   { background: #27a; color: #fff; }
  #btn-skip   { background: #555; color: #ccc; }
  #status     { padding: 6px 12px; font-size: 13px; color: #ccc; }
  #score      { color: #af7; font-weight: bold; }
  #masks-list { font-size: 12px; color: #888; margin-left: 10px; }
</style>
</head>
<body>
<div id="header">
  <span id="progress">Loading...</span>
  <span id="frame-info"></span>
</div>
<div id="canvas-wrap">
  <canvas id="canvas"></canvas>
  <canvas id="overlay"></canvas>
</div>
<div id="controls">
  <button id="btn-accept" onclick="accept()">Accept</button>
  <button id="btn-redo"   onclick="redo()">Redo</button>
  <button id="btn-next"   onclick="next()">Next &#8594;</button>
  <button id="btn-skip"   onclick="skip()">Skip</button>
  <span id="status">Draw a box around an object</span>
  <span id="masks-list"></span>
</div>
<script>
const canvas  = document.getElementById('canvas');
const overlay = document.getElementById('overlay');
const ctx     = canvas.getContext('2d');
const octx    = overlay.getContext('2d');

let startX, startY, dragging = false;
let currentBox = null;

function setStatus(msg, score) {
  document.getElementById('status').innerHTML =
    score !== undefined ? `Score: <span id="score">${score.toFixed(3)}</span> &nbsp; ${msg}` : msg;
}

function loadFrame(data) {
  document.getElementById('progress').textContent =
    `Frame ${data.frame_idx + 1} / ${data.total}`;
  document.getElementById('frame-info').textContent =
    (data.env ? `[${data.env}]  ` : '') + data.rel_path;
  document.getElementById('masks-list').textContent =
    data.accepted_count > 0 ? `${data.accepted_count} mask(s) accepted` : '';
  currentBox = null;

  const img = new Image();
  img.onload = () => {
    canvas.width  = img.width;
    canvas.height = img.height;
    overlay.width  = img.width;
    overlay.height = img.height;
    ctx.drawImage(img, 0, 0);
    octx.clearRect(0, 0, overlay.width, overlay.height);
    if (data.overlay_b64) {
      const ov = new Image();
      ov.onload = () => ctx.drawImage(ov, 0, 0);
      ov.src = 'data:image/jpeg;base64,' + data.overlay_b64;
    }
  };
  img.src = 'data:image/jpeg;base64,' + data.image_b64;
  setStatus('Draw a box around an object');
}

canvas.addEventListener('mousedown', e => {
  const r = canvas.getBoundingClientRect();
  startX = e.clientX - r.left;
  startY = e.clientY - r.top;
  dragging = true;
});

canvas.addEventListener('mousemove', e => {
  if (!dragging) return;
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  octx.strokeStyle = '#ff0';
  octx.lineWidth = 2;
  octx.strokeRect(startX, startY, x - startX, y - startY);
});

canvas.addEventListener('mouseup', e => {
  if (!dragging) return;
  dragging = false;
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  if (Math.abs(x - startX) < 10 || Math.abs(y - startY) < 10) return;
  currentBox = [Math.min(startX,x), Math.min(startY,y), Math.max(startX,x), Math.max(startY,y)];
  setStatus('Running SAM3...');
  fetch('/box', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({box: currentBox})})
    .then(r => r.json()).then(data => {
      if (data.error) { setStatus(data.error); return; }
      const img = new Image();
      img.onload = () => { ctx.drawImage(img, 0, 0); octx.clearRect(0,0,overlay.width,overlay.height); };
      img.src = 'data:image/jpeg;base64,' + data.overlay_b64;
      setStatus('Accept or redo', data.score);
    });
});

function accept() {
  fetch('/accept', {method:'POST'}).then(r=>r.json()).then(data => {
    document.getElementById('masks-list').textContent =
      `${data.accepted_count} mask(s) accepted`;
    const img = new Image();
    img.onload = () => { ctx.drawImage(img, 0, 0); octx.clearRect(0,0,overlay.width,overlay.height); };
    img.src = 'data:image/jpeg;base64,' + data.overlay_b64;
    setStatus('Draw another box, or click Next');
  });
}

function redo() {
  fetch('/redo', {method:'POST'}).then(r=>r.json()).then(data => {
    const img = new Image();
    img.onload = () => { ctx.drawImage(img, 0, 0); octx.clearRect(0,0,overlay.width,overlay.height); };
    img.src = 'data:image/jpeg;base64,' + data.overlay_b64;
    setStatus('Draw a box around an object');
  });
}

function next() {
  setStatus('Saving...');
  fetch('/next', {method:'POST'}).then(r=>r.json()).then(data => {
    if (data.done) { setStatus('All frames done!'); return; }
    loadFrame(data);
  });
}

function skip() {
  fetch('/skip', {method:'POST'}).then(r=>r.json()).then(data => {
    if (data.done) { setStatus('All frames done!'); return; }
    loadFrame(data);
  });
}

// Load first frame on page load
fetch('/state').then(r=>r.json()).then(loadFrame);
</script>
</body>
</html>
"""


def create_app(frames: list, root: Path, processor: Sam3Processor) -> Flask:
    app = Flask(__name__)
    app.config['frames'] = frames
    app.config['root'] = root
    app.config['processor'] = processor

    state = {
        'idx': 0,
        'image': None,
        'accepted': [],       # list of (mask, score)
        'current_mask': None,
        'current_score': None,
    }

    def load_frame(idx):
        frame_side_dir, image_path, _ = frames[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        clear_masks(frame_side_dir / "masks")
        # Mark this frame as manually visited (distinguishes from SAM3-generated masks)
        marker = frame_side_dir / "annotated.json"
        marker.write_text(json.dumps({"manually_annotated": True}))
        state['image'] = image
        state['accepted'] = []
        state['current_mask'] = None
        state['current_score'] = None

    def frame_response(done=False):
        if done:
            return jsonify({'done': True})
        idx = state['idx']
        frame_side_dir, _, _ = frames[idx]
        rel = frame_side_dir.relative_to(root)
        env = rel.parts[0] if len(rel.parts) > 0 else ""
        overlay_b64 = overlay_masks(state['image'], state['accepted'], state['current_mask'])
        return jsonify({
            'frame_idx': idx,
            'total': len(frames),
            'rel_path': str(rel),
            'env': env,
            'image_b64': image_to_b64(state['image']),
            'overlay_b64': overlay_b64,
            'accepted_count': len(state['accepted']),
        })

    @app.route('/')
    def index():
        return render_template_string(HTML)

    @app.route('/state')
    def get_state():
        load_frame(state['idx'])
        return frame_response()

    @app.route('/box', methods=['POST'])
    def box():
        data = request.json
        box = np.array(data['box'])
        mask, score = run_sam3_box(processor, state['image'], box)
        if mask is None:
            return jsonify({'error': 'No mask found - try a different box'})
        state['current_mask'] = mask
        state['current_score'] = score
        overlay_b64 = overlay_masks(state['image'], state['accepted'], mask)
        return jsonify({'overlay_b64': overlay_b64, 'score': score})

    @app.route('/accept', methods=['POST'])
    def accept():
        if state['current_mask'] is not None:
            state['accepted'].append((state['current_mask'], state['current_score']))
            state['current_mask'] = None
            state['current_score'] = None
        overlay_b64 = overlay_masks(state['image'], state['accepted'])
        return jsonify({'overlay_b64': overlay_b64, 'accepted_count': len(state['accepted'])})

    @app.route('/redo', methods=['POST'])
    def redo():
        state['current_mask'] = None
        state['current_score'] = None
        overlay_b64 = overlay_masks(state['image'], state['accepted'])
        return jsonify({'overlay_b64': overlay_b64})

    @app.route('/next', methods=['POST'])
    def next_frame():
        # Save accepted masks
        frame_side_dir, _, _ = frames[state['idx']]
        if state['accepted']:
            save_masks(state['accepted'], frame_side_dir / "masks")
            print(f"  Saved {len(state['accepted'])} mask(s) for {frame_side_dir.name}")
        state['idx'] += 1
        if state['idx'] >= len(frames):
            return jsonify({'done': True})
        load_frame(state['idx'])
        return frame_response()

    @app.route('/skip', methods=['POST'])
    def skip():
        print(f"  Skipped {frames[state['idx']][0].name}")
        state['idx'] += 1
        if state['idx'] >= len(frames):
            return jsonify({'done': True})
        load_frame(state['idx'])
        return frame_response()

    return app


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Web-based SAM3 annotation for DROID frames")
    parser.add_argument("input_dir", help="droid_processed directory or single trajectory directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--skip-envs", nargs="+", default=[],
                        help="Environments to skip (e.g. --skip-envs AUTOLab CLVR GuptaLab ILIAD)")
    args = parser.parse_args()

    base = Path(args.input_dir)
    if not base.exists():
        print(f"ERROR: {base} not found")
        sys.exit(1)

    skip_envs = set(args.skip_envs)
    if skip_envs:
        print(f"Skipping environments: {', '.join(sorted(skip_envs))}")

    if (base / "stereo").exists():
        frames = find_frames(base)
    else:
        frames = find_all_frames(base, skip_envs=skip_envs)

    if not frames:
        print(f"No frames with pointmaps found in {base}")
        sys.exit(1)

    print(f"Found {len(frames)} frames with pointmaps")

    processor = get_processor(args.device)

    app = create_app(frames, base, processor)

    print(f"\nOpen in browser: http://localhost:{args.port}")
    print(f"(via SSH: ssh -L {args.port}:localhost:{args.port} user@server)\n")
    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == "__main__":
    main()
