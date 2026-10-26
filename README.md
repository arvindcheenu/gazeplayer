# 👁 GazePlayer

A desktop player for **multi-participant mobile eye-tracking recordings**. It plays every participant's world-camera video side by side, synchronised on their timestamps, with gaze, fixations and scanpaths drawn on top. In the player you can annotate **areas of interest (AOIs)** per participant, speed that up with **YOLO** tracking or **FastSAM** click-to-segment, add timestamped events, and export **dwell-time and transition metrics**.

It was built for a four-musician rehearsal study recorded with Pupil Labs glasses, but nothing is tied to that study any more. Participants are discovered from the folder structure, and study-specific settings live in an optional `gazeplayer.json`.

```
┌──────────────┬──────────────┐ ┌──────────────┐
│ ALICE → drums│ BOB → Other  │ │ AOIs │ Tracks│
│  (video+gaze)│  (video+gaze)│ │ ● drums       │
├──────────────┼──────────────┤ │ ● guitar      │
│ CAROL        │ DAN → alice  │ │ ● alice ...   │
│              │              │ │ 📊 Metrics    │
└──────────────┴──────────────┘ │ 💾 Export CSV │
 ▶ 0:30.03 / 7:22.27   Draw AOI: 1 2 3 4   Gaze Fix Trail Scan AOIs YOLO
 ═══════════|═══════════════════ timeline with event markers ════════
```

## Install

Requires Python 3.9 or newer.

```bash
pip install git+https://github.com/arvindcheenu/gazeplayer
pip install "gazeplayer[all] @ git+https://github.com/arvindcheenu/gazeplayer"
```

Or from a local clone:

```bash
pip install /path/to/gazeplayer              # player + headless analysis
pip install "/path/to/gazeplayer[detect]"    # + YOLO / FastSAM / model training (pulls in PyTorch)
pip install "/path/to/gazeplayer[all]"       # + detect + optional dark theme
```

For development, use `pip install -e "/path/to/gazeplayer[dev]"` and run `pytest`.

This installs a `gazeplayer` command. Using a virtual environment is recommended:

```bash
python3 -m venv ~/.venvs/gazeplayer
source ~/.venvs/gazeplayer/bin/activate
pip install "/path/to/gazeplayer[all]"
```

> **macOS + external drives:** exFAT/FAT drives fill folders with `._*` metadata files. If the install complains about `._something.py`, run `find /path/to/gazeplayer -name '._*' -delete` first, or install from a copy on your internal disk.

## Quick start

```bash
gazeplayer info recordings/song1/attempt1     # check what was found and how it syncs
gazeplayer recordings/song1/attempt1          # open the player
```

You can also run `gazeplayer` on its own and pick a folder with **📂 Open** (Ctrl+O).

A **session folder** has one sub-folder per participant, each holding a Pupil Labs–style export:

```
attempt1/                 ← the session folder you open
├── alice/
│   ├── world.mp4         scene video
│   ├── world.csv         one timestamp per video frame  (needed for sync)
│   ├── gaze.csv          gaze samples
│   ├── fixations.csv     optional
│   └── saccades.csv, imu.csv, events.csv   optional, loaded if present
├── bob/ …
└── (files the player writes: session_aois.json, session_events.csv, …)
```

Any sub-folder containing `world.mp4`, `world.csv` or `gaze.csv` counts as a participant, named after the folder. The [data formats](docs/data-formats.md) page lists the exact columns.

## A typical workflow

1. **Check the session.** Run `gazeplayer info SESSION`. Participants whose recordings don't overlap in time are marked `sync: NO`; they still play, but from their own start.
2. **Optional pre-processing.** Run `gazeplayer yolo SESSION` to track people and objects, or `gazeplayer fastsam SESSION` to precompute click-to-segment masks.
3. **Annotate AOIs** in the player. You can draw boxes, import YOLO tracks, or click FastSAM masks, then fix boxes per participant and per frame.
4. **Add events** (E) for musical or behavioural moments.
5. **Export metrics** with **💾 Export Metrics to CSV** in the GUI or `gazeplayer metrics SESSION` in a script. **🔄 Generate Transition Events** (or `gazeplayer transitions SESSION`) writes every AOI-to-AOI gaze shift to the event timeline.

## Command line

| Command | What it does |
|---|---|
| `gazeplayer [SESSION]` | Open the player (same as `gazeplayer open`) |
| `gazeplayer info SESSION` | List participants, video sizes, sample counts and sync status |
| `gazeplayer metrics SESSION [-o DIR]` | Write `metrics_dwell_time.csv`, `metrics_transitions.csv`, `metrics_summary.csv` |
| `gazeplayer transitions SESSION` | Regenerate `gaze_transition` events in `session_events.csv` |
| `gazeplayer yolo SESSION` | YOLO + ByteTrack → `<participant>/yolo_detections.json` *(detect extra)* |
| `gazeplayer fastsam SESSION` | FastSAM masks → `<participant>/fastsam_masks.json` *(detect extra)* |
| `gazeplayer training-data PATH… -o DIR` | Turn annotated AOIs into a YOLO training set (training it needs the detect extra) |
| `gazeplayer init-config SESSION` | Write a starter `gazeplayer.json` |

Every command takes `-c/--config FILE` and `-q/--quiet`. The [CLI reference](docs/cli.md) has all the options.

## Configuration (optional)

Without a config file, participants are the data sub-folders in alphabetical order, and the AOI name presets are those participant names. To fix the order and colours, offer your own AOI labels, or set the tie-breaking rules for overlapping AOIs, put a `gazeplayer.json` in the session folder **or any parent folder**:

```json
{
  "participants": ["alice", "bob", "carol", "dan"],
  "participant_colors": {"alice": "#ff6b6b", "bob": "#4ecdc4"},
  "aoi_presets": ["guitar", "drums", "mike", "monitor", "phone"],
  "depth_hierarchy": {"alice": ["alice", "dan", "carol", "bob"]}
}
```

[examples/gazeplayer.json](examples/gazeplayer.json) is a complete four-participant example. The [configuration](docs/configuration.md) page covers every key.

> ⚠ **Metrics depend on the config.** When gaze falls on overlapping AOIs, `depth_hierarchy` decides which one wins, so the same session can give slightly different dwell times with and without the config. Keep the config file alongside the data you analyse.

## Use it from Python

The core package doesn't need Qt, so it runs fine in notebooks and on servers:

```python
from gazeplayer import load_session, export_metrics_csv

session, aois, tracker = load_session("recordings/song1/attempt1")
metrics = tracker.compute_all_metrics()          # {participant: GazeAOIMetrics}
print(metrics["alice"].dwell_time_ms)
export_metrics_csv(metrics, "out/")
```

See the [Python API](docs/python-api.md) page.

## Documentation

- [User guide](docs/user-guide.md): the player window, AOI editing, YOLO and FastSAM workflows, keyboard shortcuts
- [CLI reference](docs/cli.md)
- [Configuration](docs/configuration.md)
- [Data formats](docs/data-formats.md): input CSVs and every file the player writes
- [Python API](docs/python-api.md)
- [Changelog](CHANGELOG.md): what changed from the original `player/` scripts

## License

MIT. See [LICENSE](LICENSE).
