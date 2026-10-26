# Configuration

Every setting is optional, and with no config at all GazePlayer works out of the box. A config only fixes study-specific choices: participant order and colours, AOI name presets, and the rules for attributing gaze to overlapping AOIs.

## Where the config is found

1. `--config FILE` (`-c`) on the command line, if given.
2. Otherwise the nearest **`gazeplayer.json`**, looking in the session folder and then each parent folder in turn.
3. Otherwise the built-in defaults.

Put one file at the top of your recordings tree to cover every session below it:

```
recordings/
├── gazeplayer.json          ← applies to every session below
├── song1/attempt1/ …
└── song2/attempt1/ …
```

`gazeplayer info SESSION` prints which config file was used. `gazeplayer init-config SESSION` writes a starter file listing the participants it discovered.

## Keys

```json
{
  "participants": ["alice", "bob", "carol", "dan"],
  "participant_colors": {"alice": "#ff6b6b", "bob": "#4ecdc4", "carol": "#ffe66d", "dan": "#95e1d3"},
  "aoi_presets": ["person_1", "person_2", "guitar", "drums", "mike", "monitor", "phone"],
  "quick_labels": ["alice", "bob", "carol", "dan", "monitor", "phone"],
  "depth_hierarchy": {
    "alice":    ["alice", "dan", "carol", "bob"],
    "bob": ["bob", "carol", "dan", "alice"]
  },
  "gaze_aoi_proximity_px": 40,
  "gaze_sample_ms": 5,
  "event_types": ["musical_event", "gaze_event", "gaze_transition", "interaction", "annotation"]
}
```

| Key | Default | Meaning |
|---|---|---|
| `participants` | every sub-folder with data, alphabetical | Participant IDs **in display order**. Each ID is matched case-insensitively to a sub-folder. IDs with no folder get a "no data" tile. If none match, all data folders are used. |
| `participant_colors` | built-in palette | Hex colour per participant ID, used for the tile header, gaze marker, draw button and timeline ticks. |
| `aoi_presets` | the participant IDs | Names offered when you create an AOI. You can always type a custom name. The first 7 also appear as buttons in the YOLO import dialog. |
| `quick_labels` | participant IDs, then `aoi_presets` | Up to 9 labels bound to keys **1–9** in the Tracks tab. |
| `depth_hierarchy` | each observer uses the `participants` order | See [below](#depth_hierarchy). |
| `gaze_aoi_proximity_px` | `40` | Gaze within this many pixels of an AOI's edge counts as on that AOI. |
| `gaze_sample_ms` | `5` | Milliseconds of dwell credited per gaze sample (5 = 200 Hz). Set it to `1000 / your gaze rate`. |
| `event_types` | as above | Choices in the Add Event dialog; the field also accepts free text. |

Unknown keys are ignored with a warning. Keys starting with `_` (such as `"_comment"`) are ignored silently, so you can annotate the file.

### Participant IDs are stored in your data

AOI keyframes in `session_aois.json` and the `participant` column of `session_events.csv` use participant IDs. If you add or change `participants` after annotating, keep the same spelling, or keyframes recorded under the old IDs will no longer show.

## `depth_hierarchy`

Gaze often lands where several AOIs overlap, for example a guitar AOI in front of the person playing it. After preferring the most specific box (see the [user guide](user-guide.md#how-gaze-is-assigned-to-an-aoi)), remaining ties are broken by **the observer's depth order**: for each observer (participant ID), a list of AOI names from nearest to furthest. The AOI listed earliest wins. AOIs not in the list rank after all listed ones.

In the original study each musician's AOI was named after them, and the order encodes who stands in front of whom from each player's position:

```json
"depth_hierarchy": {
  "alice": ["alice", "dan", "carol", "bob"],
  "carol": ["carol", "bob", "dan", "alice"]
}
```

Names are compared case-insensitively. Observers missing from the mapping fall back to the `participants` list order, or to discovery order if there's no `participants` list.

> Because this setting changes metrics, keep the config that produced a set of results under version control together with the analysis.

## A complete example

[examples/gazeplayer.json](../examples/gazeplayer.json) is a full config for a four-person band, with every key set. It has the same structure as the config used in the original study, but the participant IDs are anonymised. Copy it to the root of your recordings folder and edit it:

```bash
cp examples/gazeplayer.json /path/to/recordings/gazeplayer.json
```
