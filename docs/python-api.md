# Python API

`import gazeplayer` doesn't import Qt, so everything on this page works in scripts, notebooks and on headless machines.

## One-call loading

```python
from gazeplayer import load_session

session, aois, tracker = load_session("recordings/song1/attempt1")          # nearest gazeplayer.json
session, aois, tracker = load_session(path, "examples/gazeplayer.json")     # explicit config file
session, aois, tracker = load_session(path, PlayerConfig(participants=["a", "b"]))
```

`load_session` returns:

| Object | Class | Holds |
|---|---|---|
| `session` | `SessionData` | Participants, synchronisation, events |
| `aois` | `AOIManager` | AOIs from `session_aois.json`; gaze classification |
| `tracker` | `GazeAOITracker` | Metrics and transition events |

Status messages go through the standard `logging` module (logger names start with `gazeplayer`). Call `logging.basicConfig(level=logging.INFO)` to see them.

## Session data

```python
session.participants              # {participant_id: ParticipantData}, in display order
p = session.participants["alice"]
p.gaze, p.fixations, p.saccades   # pandas DataFrames (or None if the file is missing)
p.world, p.imu
p.fps, p.total_frames, p.frame_width, p.frame_height
p.start_timestamp, p.end_timestamp          # ns

frame, idx = p.get_frame_at_timestamp(ts_ns)   # RGB numpy array + frame index
p.get_gaze_at_timestamp(ts_ns)                 # (x, y) mean within ±33 ms, or None

sync = session.sync_info
sync.overlap_start_ns, sync.overlap_end_ns, sync.overlap_duration_s
sync.participant_offsets          # seconds each recording started before the window
sync.participant_start_frames     # first frame inside the window
session.get_current_timestamp(12.5)            # session seconds → ns

session.events                    # DataFrame of session_events.csv
session.close()                   # release the video files
```

## AOIs and classification

```python
aois.aois                                   # {name: AOI}
box = aois.aois["guitar"].get_bbox_at_frame("alice", 1200)   # BoundingBox or None
aois.get_all_aois_at_frame("alice", 1200)    # {name: BoundingBox}
aois.classify_gaze_aoi("alice", 1200, gx, gy, p.frame_width, p.frame_height)   # name or "Other"
```

Editing methods (`create_aoi`, `update_aoi`, `update_aoi_isolated`, `delete_keyframe`, `rename_aoi`, `delete_aoi`) **save to `session_aois.json` immediately**.

## Metrics

```python
metrics = tracker.compute_all_metrics()          # {participant: GazeAOIMetrics}
m = tracker.compute_metrics("alice", start_frame, end_frame)   # a frame range

m.dwell_time_ms        # {aoi: ms}
m.transitions          # {(from_aoi, to_aoi): count}
m.transition_sequence  # [(from, to), …] in order
m.first_fixation_aoi

from gazeplayer import export_metrics_csv
export_metrics_csv(metrics, "out/")              # returns the paths written

events = tracker.generate_all_transition_events()   # list of event dicts (not saved)
session.add_events_batch(events)                    # saves to session_events.csv
```

### Example: dwell time per musical section

```python
import pandas as pd
from gazeplayer import load_session

session, aois, tracker = load_session("recordings/song1/attempt1")
sections = session.events[session.events["type"] == "musical_event"].sort_values("timestamp [ns]")

rows = []
for (_, a), (_, b) in zip(sections.iterrows(), sections.iloc[1:].iterrows()):
    for pid, p in session.participants.items():
        ts = p.world["timestamp [ns]"].values
        start, end = ts.searchsorted(a["timestamp [ns]"]), ts.searchsorted(b["timestamp [ns]"])
        m = tracker.compute_metrics(pid, int(start), int(end))
        rows += [{"section": a["name"], "participant": pid, "aoi": k, "ms": v}
                 for k, v in m.dwell_time_ms.items()]
print(pd.DataFrame(rows).pivot_table(index=["section", "participant"], columns="aoi", values="ms"))
```

## Configuration

```python
from gazeplayer import PlayerConfig, load_config, find_config

cfg = load_config("recordings/song1/attempt1")   # nearest file, or defaults
cfg.source                                       # path it came from (None = defaults)
cfg = PlayerConfig.from_file("study.json")
cfg.save("copy.json")
```

## Detection helpers

These need `pip install "gazeplayer[detect]"`:

```python
from gazeplayer.detection.yolo import run_yolo_on_video, run_on_session, load_yolo_detections
from gazeplayer.detection.fastsam import run_fastsam_on_video, decode_mask_rle, get_mask_at_point
from gazeplayer.training import prepare_dataset, find_annotated_sessions
```

`prepare_dataset` and the JSON readers and decoders need only OpenCV. ultralytics is imported only when a model actually runs.

## Embedding the GUI

```python
from gazeplayer.ui import run_gui, MultiViewPlayer
run_gui("recordings/song1/attempt1", config_path=None)   # blocks until the window closes
```
