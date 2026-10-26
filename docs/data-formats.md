# Data formats

- [Inputs](#inputs) (one folder per participant)
- [Files the player writes](#files-the-player-writes)
- [Metrics CSVs](#metrics-csvs)

All timestamps are integer **nanoseconds** (UTC epoch, as exported by Pupil Cloud / Neon Player). All pixel coordinates are in the participant's **world-video pixel space**, with the origin at the top-left.

## Inputs

A participant folder must contain at least one of `world.mp4`, `world.csv` or `gaze.csv`. Column names must match exactly, including the units in brackets.

| File | Required columns | Used for |
|---|---|---|
| `world.mp4` | — | Scene video. Frame *i* is row *i* of `world.csv`. |
| `world.csv` | `timestamp [ns]` | Frame timestamps; **needed for synchronisation** |
| `gaze.csv` | `timestamp [ns]`, `gaze x [px]`, `gaze y [px]` | Gaze overlay, AOI classification, metrics |
| `fixations.csv` | `start timestamp [ns]`, `end timestamp [ns]`, `fixation x [px]`, `fixation y [px]`, `duration [ms]` | Fixation overlay |
| `saccades.csv` | `start timestamp [ns]`, `end timestamp [ns]` | Loaded and available from the API (not drawn) |
| `imu.csv` | any | Loaded and available from the API (not drawn) |

Extra columns are ignored, so a raw Pupil Cloud "Timeseries Data" export works as it is.

## Files the player writes

### `session_aois.json` (session folder)

```json
{
  "aois": [
    {
      "name": "guitar",
      "color": "#ff5733",
      "visible": true,
      "exact_match": false,
      "participant_keyframes": {
        "alice":  {"120": {"x": 410.0, "y": 380.0, "width": 220.0, "height": 300.0}},
        "carol":  {"118": {"x": 90.0,  "y": 400.0, "width": 180.0, "height": 260.0}}
      },
      "participant_anchor_frames": {"alice": [119]}
    }
  ]
}
```

- Keyframe keys are **frame indices** into that participant's `world.csv` / `world.mp4`.
- A box holds from its keyframe until the next keyframe.
- `exact_match: true` (set for YOLO imports) shows a box only on its own keyframes (±1 frame).
- `participant_anchor_frames` lists keyframes that were pinned automatically while editing (see the [user guide](user-guide.md#editing)).

### `session_events.csv` and `session_notes.json` (session folder)

| Column | Meaning |
|---|---|
| `timestamp [ns]` | When the event happened |
| `name` | Free text; transitions are `gaze_<from>_to_<to>` |
| `type` | `musical_event`, `gaze_event`, `gaze_transition`, `interaction`, `annotation` or custom |
| `participant` | Participant ID or `all` |
| `from_aoi`, `to_aoi` | Filled for `gaze_transition` events |

Notes are stored separately in `session_notes.json` as `{"<timestamp ns>": "note text"}`.

### `yolo_detections.json` (participant folder)

Written by `gazeplayer yolo` and edited by the Tracks tab.

```json
{
  "model": "yolov8n", "video": "world.mp4", "frame_count": 13710, "fps": 30.0,
  "frame_width": 1088, "frame_height": 1080, "interval": 1, "confidence_threshold": 0.35,
  "frames": {
    "0": [{"track_id": 1, "class": "person", "confidence": 0.87,
           "bbox": [100, 200, 150, 300], "label_suggestion": "person_1"}]
  },
  "tracks": {
    "1": {"class": "person", "first_frame": 0, "last_frame": 12300, "suggested_aoi": "person_1"}
  }
}
```

`bbox` is `[x, y, width, height]`. Frames with no detections are omitted.

### `fastsam_masks.json` (participant folder)

Written by `gazeplayer fastsam`. Clicking a mask in the player adds to `annotations`.

```json
{
  "model": "FastSAM-s", "frame_width": 1088, "frame_height": 1080, "interval": 30,
  "frames": {
    "0": {"masks": [{"mask_id": 0, "bbox": [100, 200, 150, 300], "area": 15234,
                     "label": null, "rle": "523,12,1611,14,…"}]}
  },
  "annotations": {"0_2": {"mask_id": 2, "aoi_name": "guitar", "frame_idx": 0}}
}
```

`rle` is a run-length encoding of the row-major flattened mask, as alternating `start,length` pairs. `gazeplayer.detection.fastsam.decode_mask_rle(rle, height, width)` turns it back into a 0/1 array.

## Metrics CSVs

Written by **💾 Export Metrics to CSV** or `gazeplayer metrics`. How the values are computed is described in the [user guide](user-guide.md#metrics).

**`metrics_dwell_time.csv`**: one row per participant × AOI (including `Other`)

| participant | aoi | dwell_time_ms | dwell_percent |
|---|---|---|---|

**`metrics_transitions.csv`**: one row per participant × directed AOI pair

| participant | from_aoi | to_aoi | count |
|---|---|---|---|

**`metrics_summary.csv`**: one row per participant

| participant | total_dwell_time_ms | num_aois_visited | total_transitions | first_fixation_aoi |
|---|---|---|---|---|

### Training dataset (`gazeplayer training-data`)

A standard Ultralytics detection dataset: `dataset.yaml`, `images/{train,val}/*.jpg`, and `labels/{train,val}/*.txt` with one `class x_center y_center width height` line per box, normalised to 0–1. Every AOI keyframe, including anchor keyframes, becomes a label.
