"""Command-line entry point: ``gazeplayer``.

``gazeplayer SESSION`` opens the GUI; the sub-commands below run headless.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .config import CONFIG_FILENAME, PlayerConfig, load_config

COMMANDS = ("open", "info", "metrics", "transitions", "yolo", "fastsam",
            "training-data", "init-config")


def _existing_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"not a folder: {value}")
    return path


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", type=Path, metavar="FILE",
                        help=f"study config (default: nearest {CONFIG_FILENAME} "
                             "in the session folder or its parents)")
    common.add_argument("-q", "--quiet", action="store_true",
                        help="only print warnings and errors")

    parser = argparse.ArgumentParser(
        prog="gazeplayer",
        description="Synchronised multi-participant eye-tracking player and AOI toolkit.",
        epilog="Run 'gazeplayer SESSION' to open the player on a session folder.\n"
               "Run 'gazeplayer COMMAND -h' for help on a command.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("open", parents=[common], help="open the player GUI (default)")
    p.add_argument("session", nargs="?", type=_existing_dir,
                   help="session folder (optional; use Ctrl+O in the app)")

    p = sub.add_parser("info", parents=[common],
                       help="check a session folder and print what was found")
    p.add_argument("session", type=_existing_dir)

    p = sub.add_parser("metrics", parents=[common],
                       help="export dwell-time / transition metrics to CSV")
    p.add_argument("session", type=_existing_dir)
    p.add_argument("-o", "--out", type=Path,
                   help="output folder (default: the session folder)")

    p = sub.add_parser("transitions", parents=[common],
                       help="regenerate gaze-transition events in session_events.csv")
    p.add_argument("session", type=_existing_dir)

    p = sub.add_parser("yolo", parents=[common],
                       help="run YOLO detection + tracking (needs the 'detect' extra)")
    p.add_argument("session", type=_existing_dir)
    p.add_argument("-p", "--participant", action="append", metavar="NAME",
                   help="only this participant (repeatable; default: all)")
    p.add_argument("-i", "--interval", type=int, default=1,
                   help="process every Nth frame (default: 1)")
    p.add_argument("--conf", type=float, default=0.35,
                   help="confidence threshold (default: 0.35)")
    p.add_argument("--model", default="yolov8n",
                   help="model name or path to .pt weights (default: yolov8n)")
    p.add_argument("--max-frames", type=int, help="cap frames per video")

    p = sub.add_parser("fastsam", parents=[common],
                       help="precompute FastSAM masks (needs the 'detect' extra)")
    p.add_argument("session", type=_existing_dir)
    p.add_argument("-p", "--participant", action="append", metavar="NAME",
                   help="only this participant (repeatable; default: all)")
    p.add_argument("-i", "--interval", type=int, default=30,
                   help="process every Nth frame (default: 30)")
    p.add_argument("--conf", type=float, default=0.4,
                   help="confidence threshold (default: 0.4)")
    p.add_argument("--imgsz", type=int, default=640, help="model input size (default: 640)")
    p.add_argument("--max-frames", type=int, help="cap frames per video")
    p.add_argument("--save-images", action="store_true", help="also write mask PNGs")

    p = sub.add_parser("training-data", parents=[common],
                       help="build a YOLO dataset from annotated AOIs")
    p.add_argument("paths", nargs="+", type=_existing_dir, metavar="PATH",
                   help="session folders, or folders to search for annotated sessions")
    p.add_argument("-o", "--out", type=Path, required=True, help="dataset output folder")
    p.add_argument("--classes", nargs="+", metavar="NAME",
                   help="class order; 'alias=name' maps another AOI name onto a class "
                        "(default: every AOI name, alphabetical)")
    p.add_argument("-n", "--sample-rate", type=int, default=5,
                   help="use every Nth annotated frame (default: 5)")
    p.add_argument("--max-frames", type=int, help="cap frames per video")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="fraction used for training (default: 0.8)")
    p.add_argument("--seed", type=int, help="shuffle seed for a reproducible split")

    p = sub.add_parser("init-config", parents=[common],
                       help=f"write a starter {CONFIG_FILENAME} for a session")
    p.add_argument("session", type=_existing_dir)
    p.add_argument("-o", "--out", type=Path,
                   help=f"where to write (default: SESSION/{CONFIG_FILENAME})")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")

    return parser


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_open(args) -> int:
    try:
        from .ui import run_gui
    except ImportError as e:
        print(f"error: the GUI needs PyQt6 ({e})", file=sys.stderr)
        return 1
    return run_gui(str(args.session) if args.session else None,
                   str(args.config) if args.config else None)


def _load(args):
    from . import load_session
    return load_session(args.session, args.config)


def cmd_info(args) -> int:
    session, aois, _ = _load(args)
    cfg = session.config
    print(f"\nSession: {session.session_dir.resolve()}")
    print(f"Config:  {cfg.source or '(none — defaults)'}")
    if not session.participants:
        print("\n✗ No participant folders found (need gaze.csv, world.csv or world.mp4).")
        return 1

    sync = session.sync_info
    rows = [("participant", "folder", "video", "frames", "fps", "gaze", "fix", "yolo", "sync")]
    for pid, p in session.participants.items():
        synced = (sync is not None and p.world is not None and
                  p.start_timestamp <= sync.overlap_end_ns and
                  p.end_timestamp >= sync.overlap_start_ns)
        rows.append((
            pid, p.data_dir.name,
            f"{p.frame_width}x{p.frame_height}" if p.video_cap else "—",
            str(p.total_frames), f"{p.fps:.1f}",
            str(len(p.gaze)) if p.gaze is not None else "—",
            str(len(p.fixations)) if p.fixations is not None else "—",
            str(len(p.yolo_data.get('tracks', {}))) if p.yolo_data else "—",
            "yes" if synced else "NO",
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    print()
    for n, r in enumerate(rows):
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
        if n == 0:
            print("  " + "  ".join("─" * w for w in widths))
    if cfg.participants:
        missing = [p for p in cfg.participants if p not in session.participants]
        if missing:
            print(f"\n⚠ Configured but not found: {', '.join(missing)}")
    if sync:
        print(f"\nSynchronised window: {sync.overlap_duration_s:.2f}s")
    n_events = 0 if session.events is None else len(session.events)
    print(f"AOIs: {len(aois.aois)}   Events: {n_events}")
    session.close()
    return 0


def cmd_metrics(args) -> int:
    from .analysis import export_metrics_csv
    session, aois, tracker = _load(args)
    if not aois.aois:
        print("✗ No AOIs in this session (session_aois.json missing or empty).", file=sys.stderr)
        return 1
    metrics = tracker.compute_all_metrics()
    out = args.out or session.session_dir
    for path in export_metrics_csv(metrics, out):
        print(f"✓ {path}")
    session.close()
    return 0


def cmd_transitions(args) -> int:
    session, aois, tracker = _load(args)
    if not aois.aois:
        print("✗ No AOIs in this session.", file=sys.stderr)
        return 1
    session.clear_transition_events()
    events = tracker.generate_all_transition_events(
        lambda msg, pct: print(f"\r  {pct:3d}% {msg:<40}", end="", flush=True))
    print()
    if events:
        session.add_events_batch(events)
    print(f"✓ {len(events)} transition events written to {session.events_path}")
    session.close()
    return 0


def cmd_yolo(args) -> int:
    from .detection.yolo import run_on_session
    cfg = load_config(args.session, args.config)
    run_on_session(args.session, args.participant or cfg.participants,
                   interval=args.interval, conf_threshold=args.conf,
                   model_name=args.model, max_frames=args.max_frames)
    return 0


def cmd_fastsam(args) -> int:
    from .detection.fastsam import run_on_session
    cfg = load_config(args.session, args.config)
    run_on_session(args.session, args.participant or cfg.participants,
                   interval=args.interval, conf_threshold=args.conf,
                   imgsz=args.imgsz, max_frames=args.max_frames,
                   save_mask_images=args.save_images)
    return 0


def cmd_training_data(args) -> int:
    from .training import AOI_FILENAME, find_annotated_sessions, prepare_dataset
    sessions = []
    for path in args.paths:
        sessions += [path] if (path / AOI_FILENAME).exists() else find_annotated_sessions(path)
    if not sessions:
        print(f"✗ No {AOI_FILENAME} found under the given paths.", file=sys.stderr)
        return 1
    print(f"Sessions: {[str(s) for s in sessions]}")
    cfg = load_config(sessions[0], args.config)
    prepare_dataset(sessions, args.out, classes=args.classes, config=cfg,
                    train_ratio=args.train_ratio, frame_sample_rate=args.sample_rate,
                    max_frames_per_video=args.max_frames, seed=args.seed)
    return 0


def cmd_init_config(args) -> int:
    from .session import discover_participants
    out = args.out or args.session / CONFIG_FILENAME
    if out.exists() and not args.force:
        print(f"✗ {out} exists (use --force to overwrite)", file=sys.stderr)
        return 1
    pids = list(discover_participants(args.session))
    cfg = PlayerConfig(participants=pids,
                       participant_colors={p: PlayerConfig().color_for(p, i)
                                           for i, p in enumerate(pids)},
                       aoi_presets=list(pids))
    cfg.save(out)
    print(f"✓ Wrote {out} with participants {pids}")
    return 0


HANDLERS = {
    "open": cmd_open, "info": cmd_info, "metrics": cmd_metrics,
    "transitions": cmd_transitions, "yolo": cmd_yolo, "fastsam": cmd_fastsam,
    "training-data": cmd_training_data, "init-config": cmd_init_config,
}


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `gazeplayer`, `gazeplayer SESSION` and `gazeplayer -c X SESSION` mean `open`.
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help", "--version")):
        argv.insert(0, "open")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(message)s", stream=sys.stdout)
    sys.exit(HANDLERS[args.command](args))
