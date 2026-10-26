# CLI reference

```
gazeplayer [SESSION]                 open the player
gazeplayer COMMAND [options] …
gazeplayer --version
```

Every command accepts:

| Option | Meaning |
|---|---|
| `-c, --config FILE` | Use this config instead of the nearest `gazeplayer.json` ([configuration](configuration.md)) |
| `-q, --quiet` | Only print warnings and errors |

`python -m gazeplayer …` works the same as `gazeplayer …`.

---

## `open`: the player

```bash
gazeplayer                                  # empty window; Ctrl+O to choose a session
gazeplayer recordings/song1/attempt1        # 'open' is implied
gazeplayer open -c study.json recordings/song1/attempt1
```

## `info`: inspect a session

```bash
gazeplayer info recordings/song1/attempt1
```

Prints the config in use, one row per participant (folder, video size, frames, fps, gaze and fixation counts, YOLO tracks, whether it's synchronised), the length of the synchronised window, and the AOI and event counts. Exits with status 1 if no participant folders are found.

## `metrics`: export metrics

```bash
gazeplayer metrics SESSION                  # writes into SESSION/
gazeplayer metrics SESSION -o results/song1_attempt1
```

Writes `metrics_dwell_time.csv`, `metrics_transitions.csv` and `metrics_summary.csv`. The output is identical to the GUI's *Export Metrics to CSV*. Exits with status 1 if the session has no AOIs.

To process a whole study:

```bash
for s in recordings/*/*/; do gazeplayer metrics "$s" -q; done
```

## `transitions`: rebuild transition events

```bash
gazeplayer transitions SESSION
```

Deletes every `gaze_transition` row in `session_events.csv` and writes a fresh set based on the current AOIs. Other events are kept.

## `yolo`: detect and track objects

Requires `pip install "gazeplayer[detect]"`.

| Option | Default | |
|---|---|---|
| `-p, --participant NAME` | all | Only this participant; repeat for several |
| `-i, --interval N` | 1 | Process every Nth frame |
| `--conf X` | 0.35 | Minimum detection confidence |
| `--model M` | `yolov8n` | Model name (downloaded automatically) or path to `.pt` weights |
| `--max-frames N` | — | Stop after N frames per video (useful for testing) |

Writes `<participant>/yolo_detections.json`, overwriting any existing file. If `participants` is set in the config, only those participants are processed unless `-p` is given.

## `fastsam`: precompute segmentation masks

Requires `pip install "gazeplayer[detect]"`. Runs on CPU.

| Option | Default | |
|---|---|---|
| `-p, --participant NAME` | all | Only this participant; repeat for several |
| `-i, --interval N` | 30 | Segment every Nth frame |
| `--conf X` | 0.4 | Confidence threshold |
| `--imgsz N` | 640 | Model input size |
| `--max-frames N` | — | Stop after N frames per video |
| `--save-images` | off | Also write each mask as a PNG into `<participant>/fastsam_masks/` |

Writes `<participant>/fastsam_masks.json`.

## `training-data`: export a YOLO dataset from your AOIs

```bash
gazeplayer training-data recordings/ -o datasets/aoi --seed 1
gazeplayer training-data recordings/song1/attempt1 recordings/song2/attempt1 -o datasets/aoi \
    --classes phone monitor guitar drums drumkit=drums
```

| Option | Default | |
|---|---|---|
| `PATH …` | | Session folders, or folders searched recursively for `session_aois.json` |
| `-o, --out DIR` | *required* | Output dataset folder |
| `--classes NAME …` | every AOI name, alphabetical | Class order (= class IDs). `alias=name` maps another AOI name onto an existing class. AOIs that aren't classes are skipped. |
| `-n, --sample-rate N` | 5 | Use every Nth annotated frame per video |
| `--max-frames N` | — | Cap frames per video |
| `--train-ratio X` | 0.8 | Train/validation split |
| `--seed N` | random | Seed for a reproducible split |

Then train with Ultralytics (needs the detect extra):

```bash
yolo detect train data=datasets/aoi/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
gazeplayer yolo SESSION --model runs/detect/train/weights/best.pt
```

## `init-config`: start a config file

```bash
gazeplayer init-config recordings/song1/attempt1                 # writes SESSION/gazeplayer.json
gazeplayer init-config recordings/song1/attempt1 -o recordings/gazeplayer.json
```

Lists the discovered participants with palette colours and uses them as AOI presets. Refuses to overwrite an existing file unless you pass `--force`.

## Exit status

`0` on success, `1` when a command finds nothing to work on, and `2` for invalid arguments.
