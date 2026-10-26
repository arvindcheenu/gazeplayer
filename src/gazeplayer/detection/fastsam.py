"""
FastSAM annotator
=================
Runs FastSAM segmentation on world.mp4 frames using CPU.
Stores masks in a format aligned with world.csv (frame index = world.csv row).
Requires the ``detect`` extra (``pip install "gazeplayer[detect]"``).

Usage:
    gazeplayer fastsam /path/to/session [--participant NAME]

Output per participant:
    <participant>/fastsam_masks.json  — segmentation metadata + RLE masks
    <participant>/fastsam_masks/      — directory of per-frame mask PNGs (optional)

JSON schema (aligned with world.csv frame indices):
{
  "model": "FastSAM-s",
  "video": "world.mp4",
  "frame_count": 12345,
  "frame_width": 1088,
  "frame_height": 1080,
  "frames": {
    "0": {
      "masks": [
        {
          "mask_id": 0,
          "bbox": [x, y, w, h],
          "area": 15234,
          "label": null,
          "rle": "<run-length-encoded mask>"
        }
      ]
    }
  },
  "annotations": {
    "0": {
      "mask_id": 2,
      "aoi_name": "guitar",
      "frame_idx": 0
    }
  }
}

The "annotations" section is filled in by the player UI when the user
clicks on a mask to assign it an AOI label. This creates the bridge
between segmentation and AOI analysis.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import cv2


def _rle_encode(mask: np.ndarray) -> str:
    """Run-length encode a binary mask to a compact string."""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ','.join(str(x) for x in runs)


def _rle_decode(rle_str: str, shape: Tuple[int, int]) -> np.ndarray:
    """Decode a run-length encoded string back to a binary mask."""
    if not rle_str:
        return np.zeros(shape, dtype=np.uint8)
    runs = [int(x) for x in rle_str.split(',')]
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for i in range(0, len(runs), 2):
        start = runs[i]
        length = runs[i + 1] if i + 1 < len(runs) else 0
        if start < len(mask):
            end = min(start + length, len(mask))
            mask[start:end] = 1
    return mask.reshape(shape)


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get bounding box [x, y, w, h] from binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return (0, 0, 0, 0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1))


def run_fastsam_on_video(
    video_path: Path,
    output_path: Path,
    interval: int = 30,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.9,
    imgsz: int = 640,
    max_frames: Optional[int] = None,
    save_mask_images: bool = False,
    progress_callback=None,
) -> dict:
    """
    Run FastSAM segmentation on a video.

    Parameters
    ----------
    video_path : Path to world.mp4
    output_path : Path for fastsam_masks.json
    interval : Process every Nth frame (default 30 = ~1 per second)
    conf_threshold : Confidence threshold for FastSAM
    iou_threshold : IOU threshold for NMS
    imgsz : Input image size for FastSAM
    max_frames : Cap on frames to process
    save_mask_images : Whether to save mask PNGs alongside JSON
    progress_callback : fn(message, percent)

    Returns
    -------
    dict : The segmentation results
    """
    try:
        from ultralytics import FastSAM
    except ImportError as e:
        raise ImportError(
            "ultralytics is not installed. Install the detection extra:\n"
            '    pip install "gazeplayer[detect]"') from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"  Loading FastSAM model (CPU)...")
    model = FastSAM('FastSAM-s.pt')

    results_data = {
        'model': 'FastSAM-s',
        'video': video_path.name,
        'frame_count': total_frames,
        'fps': fps,
        'frame_width': frame_w,
        'frame_height': frame_h,
        'interval': interval,
        'frames': {},
        'annotations': {},
    }

    mask_dir = None
    if save_mask_images:
        mask_dir = output_path.parent / 'fastsam_masks'
        mask_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    frame_idx = 0

    print(f"  Processing {total_frames} frames (every {interval} frames)...")

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = model(
                frame,
                device='cpu',
                retina_masks=True,
                imgsz=imgsz,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
        except Exception as e:
            print(f"    ⚠ Error at frame {frame_idx}: {e}")
            frame_idx += interval
            continue

        frame_masks = []

        if results and len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()

            for mask_id, mask in enumerate(masks):
                # Resize mask to frame size if needed
                if mask.shape != (frame_h, frame_w):
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8),
                        (frame_w, frame_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    mask_resized = mask.astype(np.uint8)

                bbox = _bbox_from_mask(mask_resized)
                area = int(mask_resized.sum())

                # Skip tiny masks
                if area < 100:
                    continue

                rle = _rle_encode(mask_resized)

                frame_masks.append({
                    'mask_id': mask_id,
                    'bbox': list(bbox),
                    'area': area,
                    'label': None,
                    'rle': rle,
                })

                # Optionally save mask image
                if mask_dir is not None:
                    mask_path = mask_dir / f"frame_{frame_idx:06d}_mask_{mask_id:03d}.png"
                    cv2.imwrite(str(mask_path), mask_resized * 255)

        if frame_masks:
            results_data['frames'][str(frame_idx)] = {'masks': frame_masks}

        processed += 1
        if progress_callback and processed % 10 == 0:
            pct = int(100 * frame_idx / total_frames)
            progress_callback(f"Frame {frame_idx}/{total_frames}", pct)

        if processed % 50 == 0:
            print(f"    Frame {frame_idx}/{total_frames} "
                  f"({100*frame_idx/total_frames:.0f}%) — "
                  f"{len(frame_masks)} masks")

        frame_idx += interval

    cap.release()

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_data, f)

    n_frames = len(results_data['frames'])
    total_masks = sum(
        len(fd['masks'])
        for fd in results_data['frames'].values()
    )
    print(f"  ✓ Done: {n_frames} frames, {total_masks} total masks")
    print(f"  ✓ Saved to {output_path}")

    return results_data


def load_fastsam_data(json_path: Path) -> Optional[dict]:
    """Load previously computed FastSAM data."""
    if not json_path.exists():
        return None
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ Could not load FastSAM data: {e}")
        return None


def save_fastsam_data(data: dict, json_path: Path):
    """Save FastSAM data back to disk (after annotation updates)."""
    with open(json_path, 'w') as f:
        json.dump(data, f)


def get_masks_at_frame(
    fastsam_data: dict,
    frame_idx: int,
) -> List[dict]:
    """Get all masks at a specific frame (exact or nearest processed frame)."""
    frames = fastsam_data.get('frames', {})

    # Exact match
    key = str(frame_idx)
    if key in frames:
        return frames[key].get('masks', [])

    # Find nearest processed frame
    if not frames:
        return []

    processed_indices = sorted(int(k) for k in frames.keys())
    # Find closest
    idx = np.searchsorted(processed_indices, frame_idx)
    if idx >= len(processed_indices):
        idx = len(processed_indices) - 1
    elif idx > 0:
        # Check which is closer
        if (frame_idx - processed_indices[idx-1]) <= (processed_indices[idx] - frame_idx):
            idx = idx - 1

    nearest = str(processed_indices[idx])
    return frames[nearest].get('masks', [])


def get_mask_at_point(
    fastsam_data: dict,
    frame_idx: int,
    click_x: int,
    click_y: int,
) -> Optional[dict]:
    """
    Find the mask that contains a clicked point.
    Returns the smallest mask containing the point (most specific).
    """
    masks = get_masks_at_frame(fastsam_data, frame_idx)
    if not masks:
        return None

    frame_h = fastsam_data.get('frame_height', 1080)
    frame_w = fastsam_data.get('frame_width', 1088)

    candidates = []
    for mask_info in masks:
        bbox = mask_info['bbox']
        bx, by, bw, bh = bbox
        # Quick bbox check
        if bx <= click_x <= bx + bw and by <= click_y <= by + bh:
            # Decode RLE and check exact pixel
            rle = mask_info.get('rle', '')
            if rle:
                mask = _rle_decode(rle, (frame_h, frame_w))
                if click_y < mask.shape[0] and click_x < mask.shape[1]:
                    if mask[click_y, click_x] > 0:
                        candidates.append(mask_info)

    if not candidates:
        return None

    # Return smallest (most specific) mask
    return min(candidates, key=lambda m: m['area'])


def annotate_mask(
    fastsam_data: dict,
    frame_idx: int,
    mask_id: int,
    aoi_name: str,
):
    """
    Annotate a mask with an AOI name. This stores the annotation
    in the fastsam_data dict and can be used to create AOI bounding
    boxes from segmentation masks.
    """
    annotation_key = f"{frame_idx}_{mask_id}"
    if 'annotations' not in fastsam_data:
        fastsam_data['annotations'] = {}

    fastsam_data['annotations'][annotation_key] = {
        'mask_id': mask_id,
        'aoi_name': aoi_name,
        'frame_idx': frame_idx,
    }

    # Also update the label in the frame data
    frame_key = str(frame_idx)
    if frame_key in fastsam_data.get('frames', {}):
        for mask in fastsam_data['frames'][frame_key].get('masks', []):
            if mask['mask_id'] == mask_id:
                mask['label'] = aoi_name
                break


def get_annotations_for_aoi(
    fastsam_data: dict,
    aoi_name: str,
) -> List[dict]:
    """Get all annotations for a specific AOI name."""
    result = []
    for key, ann in fastsam_data.get('annotations', {}).items():
        if ann.get('aoi_name') == aoi_name:
            result.append(ann)
    return sorted(result, key=lambda a: a.get('frame_idx', 0))


def mask_to_bbox(
    fastsam_data: dict,
    frame_idx: int,
    mask_id: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Get the bounding box for a specific mask (for AOI creation)."""
    frame_key = str(frame_idx)
    if frame_key not in fastsam_data.get('frames', {}):
        return None

    for mask in fastsam_data['frames'][frame_key].get('masks', []):
        if mask['mask_id'] == mask_id:
            return tuple(mask['bbox'])
    return None


# Expose decode for use in the player UI
def decode_mask_rle(rle_str: str, frame_height: int, frame_width: int) -> np.ndarray:
    """Public wrapper for RLE decoding."""
    return _rle_decode(rle_str, (frame_height, frame_width))


# =============================================================================
# Session runner (used by `gazeplayer fastsam`)
# =============================================================================

def run_on_session(session_path: Path, participants: Optional[List[str]] = None,
                   interval: int = 30, conf_threshold: float = 0.4,
                   imgsz: int = 640, max_frames: Optional[int] = None,
                   save_mask_images: bool = False):
    """Segment every participant video in a session.

    Writes ``<participant>/fastsam_masks.json`` next to each ``world.mp4``.
    """
    from ..session import discover_participants

    folders = discover_participants(session_path, participants)
    if participants:
        folders = {pid: d for pid, d in folders.items()
                   if pid.lower() in {p.lower() for p in participants}}
    if not folders:
        print(f"⚠ No participant folders found in {session_path}")
        return

    for pid, pdir in folders.items():
        video_path = pdir / 'world.mp4'
        if not video_path.exists():
            print(f"⚠ Skipping {pid}: no world.mp4 in {pdir}")
            continue

        print(f"\n{'='*60}")
        print(f"FastSAM segmentation: {pid} ({video_path})")
        print(f"{'='*60}")

        run_fastsam_on_video(
            video_path=video_path,
            output_path=pdir / 'fastsam_masks.json',
            interval=interval,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            max_frames=max_frames,
            save_mask_images=save_mask_images,
        )

    print(f"\n✅ FastSAM processing complete!")
