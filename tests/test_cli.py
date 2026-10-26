import json

import pytest

from gazeplayer.cli import build_parser, main
from gazeplayer.config import CONFIG_FILENAME


def run(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    return exc.value.code


def test_bare_session_means_open(session_dir, monkeypatch):
    from gazeplayer import cli
    seen = []
    monkeypatch.setitem(cli.HANDLERS, "open", lambda args: seen.append(args.session) or 0)
    assert run([str(session_dir)]) == 0
    assert run(["-q", str(session_dir)]) == 0
    assert seen == [session_dir, session_dir]


def test_parser_knows_all_commands():
    p = build_parser()
    for cmd in ("info", "metrics", "transitions", "yolo", "fastsam", "init-config"):
        assert p.parse_args([cmd, "."]).command == cmd


def test_info(session_with_aois, capsys):
    assert run(["info", str(session_with_aois), "-q"]) == 0
    out = capsys.readouterr().out
    assert "Alice" in out and "bob" in out and "AOIs: 2" in out


def test_metrics(session_with_aois, tmp_path):
    assert run(["metrics", str(session_with_aois), "-o", str(tmp_path / "m"), "-q"]) == 0
    assert (tmp_path / "m" / "metrics_summary.csv").exists()


def test_metrics_without_aois_fails(session_dir):
    assert run(["metrics", str(session_dir), "-q"]) == 1


def test_transitions_written_to_events(session_with_aois):
    assert run(["transitions", str(session_with_aois), "-q"]) == 0
    text = (session_with_aois / "session_events.csv").read_text()
    assert text.count("gaze_transition") == 2


def test_init_config(session_dir):
    assert run(["init-config", str(session_dir), "-q"]) == 0
    cfg = json.loads((session_dir / CONFIG_FILENAME).read_text())
    assert cfg["participants"] == ["Alice", "bob"]
    assert run(["init-config", str(session_dir), "-q"]) == 1     # no overwrite
    assert run(["init-config", str(session_dir), "--force", "-q"]) == 0
