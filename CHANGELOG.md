# Changelog

## 4.1.0: standalone package

The original single-file player scripts ("Pupil Player v4") repackaged as an installable, study-independent tool.

### Compatibility

- **All file formats are unchanged.** Existing `session_aois.json`, `session_events.csv`, `session_notes.json`, `yolo_detections.json` and `fastsam_masks.json` open as they are.
- With the original study's config (the settings that used to be hard-coded), metrics are **byte-identical** to the original player's CSV exports. This was checked on all six pilot sessions.
- Without a config, participant order is alphabetical and depth tie-breaking follows that order, so metrics for overlapping AOIs can differ slightly. See [configuration](docs/configuration.md#depth_hierarchy).
- `session_aois.json` no longer stores empty keyframe maps for participants with no keyframes. Older readers are unaffected.

### Added

- `pip install` support with a `gazeplayer` command. Optional extras: `detect` (ultralytics), `theme` (pyqtdarktheme), `dev`.
- `gazeplayer.json` config with lookup in the session folder and its parents. It covers participants, colours, AOI presets, Tracks-tab quick labels, depth hierarchy, proximity threshold, gaze sample duration and event types.
- Any number of participants: the video grid, draw buttons and dialogs adapt to whoever is in the folder.
- Headless commands: `info`, `metrics`, `transitions`, `init-config`, `training-data`.
- **📂 Open** button / Ctrl+O, so the player can be started without a path.
- Headless Python API (`gazeplayer.load_session`, `export_metrics_csv`, …) that doesn't import Qt.
- Test suite (`pytest`) using a synthetic session, including offscreen GUI tests.
- Documentation in `docs/`.

### Fixed

- **YOLO import dialog crashed when you pressed OK** (`ValueError: too many values to unpack`). It also showed the preset buttons and OK/Cancel twice.
- The **T** (trail) and **M** (Tracks tab) shortcuts were documented or advertised in the status bar but never bound.
- `yolo --model path/to/best.pt` used to look for `best.pt.pt`. A path to weights now works.
- Detection modules no longer call `sys.exit()` when ultralytics is missing; they raise an `ImportError` explaining how to install it.

### Changed

- The 3,800-line `multi_view_player.py` is split into `config`, `session`, `aoi`, `analysis`, `detection/`, `training` and `ui/` modules. The UI code moved over unchanged apart from the configuration hooks.
- Core modules log through `logging` instead of `print`. The CLI shows the same messages; `-q` hides everything but warnings.
- `prepare_yolo_training.py` became `gazeplayer training-data`. Classes come from the AOI names (or `--classes` with `alias=name`) instead of a hard-coded map, sessions can be any folders, and `--seed` makes the split reproducible.
- When no folder matches the configured participants, folders keep their own names instead of being mapped into the four fixed slots.

### Removed

- `Makefile`: replaced by `pip install` and the `gazeplayer` command.
- `diagnose.py`: replaced by `gazeplayer info`.
- `collect_videos.py` and `test_single_frame.py`: specific to the pilot study.
- README instructions for `run_custom_yolo.py`, a script that didn't exist.
- The README claim that AOI boxes are linearly interpolated. They hold their last keyframe, and the docs now say so.
