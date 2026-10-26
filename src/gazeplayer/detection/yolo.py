"""
YOLO object tracker
===================
Runs YOLO detection + ByteTrack on world.mp4 videos to produce
per-frame bounding boxes for AOI modelling.  Requires the ``detect`` extra
(``pip install "gazeplayer[detect]"``).

Usage:
    gazeplayer yolo /path/to/session [--participant NAME] [--interval 5]

Output:
    <participant>/yolo_detections.json  — per-frame bounding boxes with track IDs

The JSON schema aligns with world.csv: frame index i maps to row i of world.csv.
{
  "model": "yolov8n",
  "video": "world.mp4",
  "frame_count": 12345,
  "fps": 30.0,
  "classes_of_interest": ["person", "guitar", "cell phone", "tv"],
  "frames": {
    "0":  [{"track_id": 1, "class": "person", "confidence": 0.87,
            "bbox": [x, y, w, h], "label_suggestion": "person_1"}],
    "30": [...]
  },
  "tracks": {
    "1": {"class": "person", "first_frame": 0, "last_frame": 12300,
          "suggested_aoi": "person_1"}
  }
}
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2

# COCO classes we care about for the music collaboration scenario
CLASSES_OF_INTEREST = {
    0:  'person',
    # Musical instruments (COCO doesn't have guitar/drums directly,
    # but person detection is primary; instruments are handled by
    # manual AOI or fine-tuned model)
    63: 'laptop',
    67: 'cell phone',
    62: 'tv',       # for monitor
}

# Mapping from YOLO class names to suggested AOI labels
LABEL_MAP = {
    'person':     'person',
    'laptop':     'monitor',
    'cell phone': 'phone',
    'tv':         'monitor',
}


def run_yolo_on_video(
    video_path: Path,
    output_path: Path,
    interval: int = 1,
    conf_threshold: float = 0.35,
    model_name: str = 'yolov8n',
    max_frames: Optional[int] = None,
    progress_callback=None,
) -> dict:
    """
    Run YOLO detection + tracking on a video file.

    Parameters
    ----------
    video_path : Path to world.mp4
    output_path : Path to write yolo_detections.json
    interval : Process every Nth frame (1 = all frames)
    conf_threshold : Minimum confidence
    model_name : YOLO model variant
    max_frames : Cap on frames to process (None = all)
    progress_callback : fn(message, percent) called during processing

    Returns
    -------
    dict : The detection results
    """
    try:
        from ultralytics import YOLO
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

    print(f"  Loading YOLO model: {model_name}...")
    model = YOLO(resolve_model(model_name))

    # We'll use the tracker built into ultralytics
    results_data = {
        'model': model_name,
        'video': video_path.name,
        'frame_count': total_frames,
        'fps': fps,
        'frame_width': frame_w,
        'frame_height': frame_h,
        'interval': interval,
        'confidence_threshold': conf_threshold,
        'classes_of_interest': list(CLASSES_OF_INTEREST.values()),
        'frames': {},
        'tracks': {},
    }

    # Track ID to info
    track_info: Dict[int, dict] = {}
    # Counter for generating suggested AOI names per class
    class_counters: Dict[str, int] = {}

    processed = 0
    frame_idx = 0

    print(f"  Processing {total_frames} frames (every {interval} frame(s))...")

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO with tracking
        # Use ByteTrack tracker built into ultralytics
        # Note: Don't filter classes if using a custom-trained model - let it detect all its classes
        try:
            track_results = model.track(
                frame,
                persist=True,
                conf=conf_threshold,
                verbose=False,
                tracker='bytetrack.yaml',
            )
        except Exception:
            # Fallback: detect without tracking
            track_results = model(
                frame,
                conf=conf_threshold,
                verbose=False,
            )

        frame_detections = []

        if track_results and len(track_results) > 0:
            result = track_results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Get track ID if available
                    track_id = int(box.id[0]) if box.id is not None else -1

                    # Get class name from model (works for both COCO and custom models)
                    class_name = model.names.get(cls_id, 'unknown')
                    # For COCO models, map to AOI-friendly names; for custom models, use as-is
                    aoi_base = LABEL_MAP.get(class_name, class_name)

                    # Build suggested label
                    if track_id >= 0:
                        if track_id not in track_info:
                            # New track - assign a label
                            class_counters[aoi_base] = class_counters.get(aoi_base, 0) + 1
                            count = class_counters[aoi_base]
                            suggested = f"{aoi_base}_{count}" if aoi_base == 'person' else aoi_base
                            track_info[track_id] = {
                                'class': class_name,
                                'first_frame': frame_idx,
                                'last_frame': frame_idx,
                                'suggested_aoi': suggested,
                            }
                        else:
                            track_info[track_id]['last_frame'] = frame_idx
                        suggested = track_info[track_id]['suggested_aoi']
                    else:
                        suggested = aoi_base

                    detection = {
                        'track_id': track_id,
                        'class': class_name,
                        'confidence': round(conf, 3),
                        'bbox': [round(x1), round(y1),
                                 round(x2 - x1), round(y2 - y1)],  # x, y, w, h
                        'label_suggestion': suggested,
                    }
                    frame_detections.append(detection)

        if frame_detections:
            results_data['frames'][str(frame_idx)] = frame_detections

        processed += 1
        if progress_callback and processed % 50 == 0:
            pct = int(100 * frame_idx / total_frames)
            progress_callback(f"Frame {frame_idx}/{total_frames}", pct)

        if processed % 200 == 0:
            print(f"    Processed frame {frame_idx}/{total_frames} "
                  f"({100*frame_idx/total_frames:.0f}%)")

        frame_idx += interval

    cap.release()

    # Store track summaries
    results_data['tracks'] = {
        str(tid): info for tid, info in track_info.items()
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    n_frames_with_det = len(results_data['frames'])
    n_tracks = len(track_info)
    print(f"  ✓ Done: {n_frames_with_det} frames with detections, {n_tracks} tracks")
    print(f"  ✓ Saved to {output_path}")

    return results_data


def load_yolo_detections(json_path: Path) -> Optional[dict]:
    """Load previously computed YOLO detections."""
    if not json_path.exists():
        return None
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ Could not load YOLO detections: {e}")
        return None


def detections_to_aoi_keyframes(
    detections: dict,
    track_id: int,
    total_frames: int,
    exact_mode: bool = True,
) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Convert YOLO detections for a specific track_id into
    AOI keyframes: frame_idx -> (x, y, w, h).
    
    If exact_mode=True (default): Only creates keyframes at detected frames.
    AOI will only be visible at frames where track was detected.
    
    If exact_mode=False: Creates keyframes with interpolation hints.
    """
    # Collect all detected frames for this track
    detected_frames = {}
    for frame_str, frame_dets in detections.get('frames', {}).items():
        frame_idx = int(frame_str)
        for det in frame_dets:
            if det.get('track_id') == track_id:
                x, y, w, h = det['bbox']
                detected_frames[frame_idx] = (float(x), float(y), float(w), float(h))
                break
    
    if not detected_frames:
        return {}
    
    # Return only detected frames as keyframes (exact 1:1 correspondence)
    return detected_frames


def get_detection_at_frame(
    detections: dict,
    frame_idx: int,
) -> List[dict]:
    """Get all detections at a specific frame."""
    return detections.get('frames', {}).get(str(frame_idx), [])


def get_interpolated_bbox(
    keyframes: Dict[int, Tuple[float, float, float, float]],
    frame_idx: int,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Get interpolated bounding box at frame_idx from sparse keyframes.
    Uses linear interpolation between the two nearest keyframes.
    """
    if not keyframes:
        return None

    sorted_frames = sorted(keyframes.keys())

    # Exact match
    if frame_idx in keyframes:
        return keyframes[frame_idx]

    # Before first keyframe
    if frame_idx < sorted_frames[0]:
        return None

    # After last keyframe
    if frame_idx > sorted_frames[-1]:
        return keyframes[sorted_frames[-1]]

    # Find bracketing keyframes
    prev_f = sorted_frames[0]
    next_f = sorted_frames[-1]
    for i, f in enumerate(sorted_frames):
        if f > frame_idx:
            next_f = f
            prev_f = sorted_frames[i - 1]
            break

    if prev_f == next_f:
        return keyframes[prev_f]

    # Interpolate
    t = (frame_idx - prev_f) / (next_f - prev_f)
    p = keyframes[prev_f]
    n = keyframes[next_f]

    return (
        p[0] + t * (n[0] - p[0]),
        p[1] + t * (n[1] - p[1]),
        p[2] + t * (n[2] - p[2]),
        p[3] + t * (n[3] - p[3]),
    )


# =============================================================================
# Session runner (used by `gazeplayer yolo`)
# =============================================================================

def resolve_model(model_name: str) -> str:
    """Accept a bare model name ('yolov8n') or a path to custom weights."""
    if model_name.endswith('.pt') or Path(model_name).exists():
        return model_name
    return f'{model_name}.pt'


def run_on_session(session_path: Path, participants: Optional[List[str]] = None,
                   interval: int = 1, conf_threshold: float = 0.35,
                   model_name: str = 'yolov8n', max_frames: Optional[int] = None):
    """Run detection + tracking for every participant video in a session.

    Writes ``<participant>/yolo_detections.json`` next to each ``world.mp4``.
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
        print(f"Processing {pid}: {video_path}")
        print(f"{'='*60}")

        run_yolo_on_video(
            video_path=video_path,
            output_path=pdir / 'yolo_detections.json',
            interval=interval,
            conf_threshold=conf_threshold,
            model_name=model_name,
            max_frames=max_frames,
        )

    print(f"\n✅ All done!")
