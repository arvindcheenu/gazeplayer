# User guide

- [Opening a session](#opening-a-session)
- [The window](#the-window)
- [Playback and synchronisation](#playback-and-synchronisation)
- [Gaze overlays](#gaze-overlays)
- [Areas of interest (AOIs)](#areas-of-interest-aois)
- [YOLO-assisted AOIs](#yolo-assisted-aois)
- [FastSAM click-to-segment](#fastsam-click-to-segment)
- [Events](#events)
- [Metrics](#metrics)
- [Keyboard and mouse reference](#keyboard-and-mouse-reference)
- [Troubleshooting](#troubleshooting)

---

## Opening a session

```bash
gazeplayer path/to/session          # open directly
gazeplayer                          # open empty, then 📂 Open / Ctrl+O
gazeplayer -c study.json path/…     # use a specific config file
```

When a session loads, the terminal prints what was found for each participant (video size, frame counts, sample counts) and how the recordings were synchronised. Read this output first when something looks wrong. `gazeplayer info SESSION` prints the same information as a table without opening the window.

## The window

| Area | Contents |
|---|---|
| **Video grid** | One tile per participant, laid out in a near-square grid. The coloured header shows the participant ID and, during playback, the AOI their gaze is on (`ALICE → guitar`). Participants named in the config but missing from the folder get a "no data" tile. |
| **Control bar** | 📂 Open, play/pause, the clock (`current / synchronised duration`), one **Draw AOI** button per participant, and the overlay toggles *Gaze · Fix · Trail · Scan · AOIs · YOLO*. |
| **Timeline** | Click or drag to seek. Vertical ticks are events, coloured by participant; gold ticks are gaze transitions. |
| **AOIs tab** | AOI list, tool buttons, metrics and event buttons. |
| **Tracks tab** | The YOLO track manager. |
| **Status bar** | Hints and the result of the last action. |

## Playback and synchronisation

The player reads the nanosecond timestamps in every participant's `world.csv` and finds the **largest group of participants whose recordings overlap in time**. The session clock runs over that overlap: 0:00 is the moment the last of them started recording, and the end is when the first of them stopped. At any session time, each synchronised participant shows the video frame nearest that wall-clock moment.

A participant who doesn't overlap the group at all is **unsynchronised**. That participant's video plays from its own first frame at the session's frame rate, and live AOI classification is off for that tile. The terminal and `gazeplayer info` flag these participants.

If a participant has a video but no `world.csv`, timestamps are generated from the video frame rate, so that participant can't be truly synchronised with the others.

| Action | How |
|---|---|
| Play / pause | **Space** or ▶ |
| Step one frame | **← / →** |
| Jump to start / end | **Home / End** |
| Seek | click or drag the timeline |

## Gaze overlays

| Toggle | Key | Shows |
|---|---|---|
| Gaze | G | Crosshair at the mean gaze position within ±33 ms of the current frame, in the participant's colour |
| Fix | F | Circles for fixations that started in the last 0.5 s; radius grows with duration |
| Trail | T | The last 30 gaze positions, fading out |
| Scan | S | Scanpath of the last 200 gaze points, coloured by AOI, with **gold** segments where gaze moved from one AOI to another. Seeking clears it. |
| AOIs | B | AOI boxes. ◆ means this frame has a keyframe and ◇ means the box is held from an earlier keyframe. A dashed outline means the box is outside the frame (OOB). |
| YOLO | Y | Raw YOLO detections labelled `T<track> <class> <confidence>` |

## Areas of interest (AOIs)

An AOI is a named rectangle, such as `guitar` or `alice`, with its **own keyframes in every participant's video**. The same guitar sits in a different place in each person's view, so each participant's boxes are edited separately.

### How keyframes behave

- At any frame, an AOI shows the box from its **most recent keyframe** for that participant, and holds it until the next keyframe. Boxes are not interpolated between keyframes.
- Before an AOI's first keyframe for a participant, the AOI doesn't exist in that participant's view.
- AOIs imported from YOLO are **exact-match**: they only appear on frames where the track was detected (±1 frame), so they vanish when the object isn't detected.

### Creating an AOI by drawing

1. Pause on a frame where the object is visible.
2. Click a numbered **Draw AOI** button (or **+ Draw**, which draws on the first participant) and drag a rectangle on that participant's video.
3. Pick a preset name or choose `custom…`.

The new box is placed **in every participant's video at the current moment**, all at the same pixel coordinates. Then move or resize it in each tile where it's wrong.

### Editing

- **Move**: drag inside a box. **Resize**: drag a corner or edge.
- Each edit creates a keyframe for that participant at the current frame. It also pins the previous frame with an *anchor* keyframe, so frames before the edit keep their old box. Later edits clear leftover anchors up to the next real keyframe, so a correction carries forward until the next real detection or drawn keyframe.
- **Right-click a box** to get:
  - *Delete keyframe at frame N*: removes this frame's keyframe. The neighbouring frames are pinned so the surrounding video looks the same.
  - *Pin current box as keyframe*: turns a held box into a real keyframe without moving it.
- **Delete** removes the keyframe at the current frame for the selected AOI.
- **Right-click an AOI in the sidebar list** to rename it or delete it from all participants. **🗑 Delete All AOIs** clears everything after you confirm.

Every change is saved to `session_aois.json` straight away.

### How gaze is assigned to an AOI

At each gaze sample, the AOIs present on that frame are checked:

1. An AOI is a **candidate** if the gaze point is inside it or within `gaze_aoi_proximity_px` (default 40 px) of its edge. AOIs that are entirely outside the video frame are ignored.
2. No candidates gives **Other**.
3. If *every* AOI on the frame is a candidate (and there is more than one), the result is also **Other**. Gaze that is ambiguous between everything isn't attributed to anything.
4. A single candidate wins.
5. Among several candidates, the one most covered by the others wins, which favours the smaller, more specific box. Ties go to the AOI listed first in the observer's `depth_hierarchy` (see [configuration](configuration.md#depth_hierarchy)), then to the AOI nearest the gaze point.

## YOLO-assisted AOIs

Requires `pip install "gazeplayer[detect]"`.

```bash
gazeplayer yolo SESSION                       # all participants, every frame
gazeplayer yolo SESSION -p alice -i 5          # one participant, every 5th frame
gazeplayer yolo SESSION --model runs/detect/my_model/weights/best.pt
```

This writes `yolo_detections.json` next to each `world.mp4`. Reopen the session afterwards (or start the player after it finishes). With the stock COCO model, people get suggested names `person_1`, `person_2`, …, `tv`/`laptop` become `monitor`, and `cell phone` becomes `phone`. A custom model's class names are used as they are.

### Cleaning tracks: the Tracks tab (M)

Trackers fragment one object into many short tracks. Clean them up before importing:

- There is one tab per participant, and tracks are grouped by class. Grey rows have fewer than 20 detections and are probably noise.
- **Selecting a track** highlights it in the video and jumps to its first detection. ↑/↓ move within a class group.
- **Rename**: select tracks (or a class header), type a class in the *Class* box and press **Enter**. The change applies immediately. Double-clicking the *Class* cell edits it in place, and that edit is staged.
- **Quick labels**: with tracks selected, press **1–9** to stage a rename to the label shown at the top of the tab (configurable with `quick_labels`). **Backspace** toggles a deletion mark. Staged changes are yellow or struck through until you click **✓ Apply**.
- **🔗 Merge** joins the selected tracks into the one with the lowest ID.
- **🗑 Delete** removes the selected tracks or class group after you confirm.
- **📦 Import as AOIs** turns every remaining track into AOI keyframes. Each track goes into the AOI named after its suggested name. That is the new class for tracks you renamed, and `person_1`, `person_2`, … for untouched person tracks. Tracks with the same name are combined, which is why renaming fragments to one label first works well.

> ⚠ Rename, merge, delete and Apply all **overwrite `yolo_detections.json`**. Copy the file first if you may want the raw detections back.

### Importing selected tracks: 🔍 YOLO

The **🔍 YOLO** button in the AOIs tab opens a table of every track in every participant. Untick tracks you don't want (short tracks start unticked) and edit the AOI name per row. The preset buttons fill in the first ticked row that doesn't already have a preset name. Tracks with the same AOI name are combined into one AOI.

## FastSAM click-to-segment

Requires `pip install "gazeplayer[detect]"`.

```bash
gazeplayer fastsam SESSION              # segment every 30th frame (CPU)
gazeplayer fastsam SESSION -i 15 -p carol
```

In the player, click **✂ SAM** to toggle segment mode, then click on an object in any video. The smallest mask under the cursor on the nearest segmented frame is used, and you're asked for a name. The mask's bounding box then becomes an AOI keyframe for that participant at that frame; if the AOI doesn't exist yet, it's created for every participant, as with drawing. The label is also stored in `fastsam_masks.json`.

## Events

Press **E** (or **+ Add Event**) to add an event at the current time, with a name, type, participant (or `all`) and notes. Events are saved to `session_events.csv`, notes to `session_notes.json`, and both show as ticks on the timeline.

**🔄 Generate Transition Events** scans *every* gaze sample of every participant and adds a `gaze_transition` event (`gaze_<from>_to_<to>`) each time gaze moves from one named AOI to another. *Other* is skipped, so a jump from `guitar` to *Other* to `drums` counts as `guitar → drums`. Running it again replaces earlier transition events. **🗑 Clear Transition Events** removes them. The CLI command `gazeplayer transitions SESSION` does the same.

## Metrics

**📊 Compute Metrics** shows a dwell-time table and a transition table per participant. **💾 Export Metrics to CSV** asks for a folder and writes three files, described in [data formats](data-formats.md#metrics-csvs). `gazeplayer metrics SESSION -o DIR` writes the same files without the GUI.

How they're computed:

- Each participant's **whole recording** is analysed, not just the synchronised window.
- For long recordings, gaze is subsampled to about 10,000 samples. Each analysed sample counts as `gaze_sample_ms × step` milliseconds of dwell. The default of 5 ms matches 200 Hz gaze; set `gaze_sample_ms` to match your tracker.
- *Other* counts toward dwell time but not toward transitions.
- `first_fixation_aoi` is the AOI of the **first gaze sample**, which may be *Other*. It is not based on fixation detection.

## Keyboard and mouse reference

| Key | Action |
|---|---|
| Space | Play / pause |
| ← / → | Previous / next frame |
| Home / End | Start / end |
| Ctrl+O | Open a session folder |
| E | Add event at current time |
| G / F / T / S / B / Y | Toggle gaze / fixations / trail / scanpath / AOI boxes / YOLO boxes |
| M | Show the Tracks tab (and the YOLO overlay) |
| Delete | Delete the selected AOI's keyframe at this frame |
| Esc | Cancel drawing |
| 1–9 *(Tracks tab)* | Stage rename of selected tracks to quick label *n* |
| Backspace *(Tracks tab)* | Toggle deletion mark on selected tracks |

| Mouse | Action |
|---|---|
| Drag after **Draw AOI** | Draw a new AOI |
| Drag inside / on edge of a box | Move / resize (creates a keyframe) |
| Right-click a box | Delete or pin keyframe |
| Right-click an AOI in the list | Rename / delete AOI |
| Click / drag timeline | Seek |

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| A tile says **No video** | OpenCV can't decode `world.mp4`. Check the terminal; re-encoding to H.264 usually fixes it. |
| A participant is marked **UNSYNCED** | Their `world.csv` timestamps don't overlap the others'. Check that the recordings really were simultaneous and that the device clocks were synced. |
| Duration shows 0:00 | No timestamp overlap and no readable video; see the terminal output. |
| AOIs from an earlier session don't appear | Keyframes are stored under participant IDs. If the IDs changed (for example a config renamed `Alice` to `alice`), they no longer match. Keep IDs stable or edit `session_aois.json`. |
| Metrics differ from an earlier run | Check that the same `gazeplayer.json` was used (`gazeplayer info` prints which one). `depth_hierarchy`, `gaze_aoi_proximity_px` and `gaze_sample_ms` all change results. |
| `ultralytics is not installed` | `pip install "gazeplayer[detect]"` |
| `qt.qpa.plugin: Could not load the Qt platform plugin` | Reinstall PyQt6 in the active environment. On a headless Linux server, use the CLI commands instead of the GUI. |
