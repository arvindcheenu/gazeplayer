"""Build a YOLO training dataset from AOIs annotated in the player.

Every AOI keyframe in ``session_aois.json`` becomes a labelled bounding box
on the matching ``world.mp4`` frame.  The output is a standard Ultralytics
dataset::

    out/
        dataset.yaml
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt     (class x_center y_center width height, 0-1)

Used by ``gazeplayer training-data``.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

from .config import PlayerConfig
from .session import discover_participants

AOI_FILENAME = 'session_aois.json'


def find_annotated_sessions(root: Path) -> List[Path]:
    """Every folder at or below *root* that contains ``session_aois.json``."""
    root = Path(root)
    return sorted(p.parent for p in root.rglob(AOI_FILENAME))


def parse_class_spec(spec: Sequence[str]) -> Tuple[List[str], Dict[str, int]]:
    """Turn ``['phone', 'drums', 'drumkit=drums']`` into class names and an
    AOI-name -> class-id lookup (keys lower-cased)."""
    names: List[str] = []
    lookup: Dict[str, int] = {}
    aliases = []
    for item in spec:
        if '=' in item:
            alias, target = item.split('=', 1)
            aliases.append((alias.strip().lower(), target.strip().lower()))
            continue
        name = item.strip().lower()
        if name not in lookup:
            lookup[name] = len(names)
            names.append(name)
    for alias, target in aliases:
        if target not in lookup:
            raise ValueError(f"Alias '{alias}={target}': unknown class '{target}'")
        lookup[alias] = lookup[target]
    return names, lookup


def _load_aois(session_dir: Path) -> List[dict]:
    with open(session_dir / AOI_FILENAME) as f:
        return json.load(f).get('aois', [])


def bbox_to_yolo(x: float, y: float, w: float, h: float,
                 img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """Top-left pixel box -> normalised centre box, clamped to 0-1."""
    clamp = lambda v: max(0.0, min(1.0, v))
    return (clamp((x + w / 2) / img_width), clamp((y + h / 2) / img_height),
            clamp(w / img_width), clamp(h / img_height))


def prepare_dataset(sessions: Sequence[Path], output_dir: Path,
                    classes: Optional[Sequence[str]] = None,
                    config: Optional[PlayerConfig] = None,
                    train_ratio: float = 0.8, frame_sample_rate: int = 5,
                    max_frames_per_video: Optional[int] = None,
                    seed: Optional[int] = None) -> Path:
    """Extract frames + labels for *sessions* into *output_dir*.

    *classes* fixes the class order (and supports ``alias=name``); without
    it every AOI name found becomes a class, sorted alphabetically.
    Returns the path of the written ``dataset.yaml``.
    """
    config = config or PlayerConfig()
    output_dir = Path(output_dir)
    sessions = [Path(s) for s in sessions]

    if not classes:
        found = {a['name'].lower() for s in sessions for a in _load_aois(s)}
        classes = sorted(found)
    class_names, lookup = parse_class_spec(classes)
    print(f"Classes: {class_names}")

    # video path -> frame index -> [label rows]
    videos: Dict[Path, Dict[int, List[tuple]]] = defaultdict(lambda: defaultdict(list))
    video_ids: Dict[Path, str] = {}
    for session_dir in sessions:
        folders = {pid.lower(): d for pid, d in
                   discover_participants(session_dir, config.participants).items()}
        label = '_'.join(session_dir.resolve().parts[-2:])
        for aoi in _load_aois(session_dir):
            cid = lookup.get(aoi['name'].lower())
            if cid is None:
                print(f"  {label}: AOI '{aoi['name']}' is not a class, skipping")
                continue
            for pid, frames in aoi.get('participant_keyframes', {}).items():
                folder = folders.get(pid.lower())
                if folder is None or not (folder / 'world.mp4').exists():
                    continue
                video = folder / 'world.mp4'
                video_ids[video] = f"{label}_{folder.name}"
                for fidx, b in frames.items():
                    videos[video][int(fidx)].append(
                        (cid, b['x'], b['y'], b['width'], b['height']))

    samples = []
    for video, frames in videos.items():
        picked = sorted(frames)[::max(1, frame_sample_rate)]
        if max_frames_per_video:
            picked = picked[:max_frames_per_video]
        print(f"  {video_ids[video]}: {len(frames)} annotated frames, using {len(picked)}")
        samples += [(video, fidx, frames[fidx]) for fidx in picked]

    random.Random(seed).shuffle(samples)
    split = int(len(samples) * train_ratio)
    splits = {'train': samples[:split], 'val': samples[split:]}

    counts = {}
    for split_name, split_samples in splits.items():
        (output_dir / 'images' / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split_name).mkdir(parents=True, exist_ok=True)
        by_video = defaultdict(list)
        for video, fidx, rows in split_samples:
            by_video[video].append((fidx, rows))
        written = 0
        for video, items in by_video.items():
            cap = cv2.VideoCapture(str(video))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            for fidx, rows in sorted(items):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                if not ret:
                    continue
                stem = f"{video_ids[video]}_frame{fidx:06d}"
                cv2.imwrite(str(output_dir / 'images' / split_name / f"{stem}.jpg"), frame)
                with open(output_dir / 'labels' / split_name / f"{stem}.txt", 'w') as f:
                    for cid, x, y, bw, bh in rows:
                        xc, yc, nw, nh = bbox_to_yolo(x, y, bw, bh, w, h)
                        f.write(f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
                written += 1
            cap.release()
        counts[split_name] = written

    yaml_path = output_dir / 'dataset.yaml'
    yaml_path.write_text(
        "# YOLO dataset generated by `gazeplayer training-data`\n"
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: {json.dumps(class_names)}\n")

    print(f"\n✓ {counts['train']} train + {counts['val']} val images → {output_dir}")
    print(f"  Train with:  yolo detect train data={yaml_path} model=yolov8n.pt epochs=50 imgsz=640")
    return yaml_path
