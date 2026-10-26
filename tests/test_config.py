import json

from gazeplayer.config import CONFIG_FILENAME, PlayerConfig, find_config, load_config


def test_defaults_without_file(tmp_path):
    cfg = load_config(tmp_path)
    assert cfg.participants is None
    assert cfg.source is None
    assert cfg.gaze_aoi_proximity_px == 40
    assert cfg.gaze_sample_ms == 5


def test_found_in_parent_folder(tmp_path):
    session = tmp_path / "recordings" / "song" / "attempt"
    session.mkdir(parents=True)
    (tmp_path / "recordings" / CONFIG_FILENAME).write_text(json.dumps({"participants": ["a"]}))
    assert find_config(session) == (tmp_path / "recordings" / CONFIG_FILENAME).resolve()
    assert load_config(session).participants == ["a"]


def test_explicit_path_wins(tmp_path):
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({"participants": ["near"]}))
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"participants": ["explicit"]}))
    assert load_config(tmp_path, other).participants == ["explicit"]


def test_unknown_keys_are_ignored():
    cfg = PlayerConfig.from_dict({"participants": ["a"], "bogus": 1, "_comment": "x"})
    assert cfg.participants == ["a"]


def test_fallbacks_follow_participants():
    cfg = PlayerConfig()
    assert cfg.presets_for(["a", "b"]) == ["a", "b"]
    assert cfg.quick_labels_for(["a", "b"]) == ["a", "b"]
    assert cfg.depth_order_for("a", ["A", "b"]) == ["a", "b"]
    assert cfg.color_for("a", 0) != cfg.color_for("b", 1)


def test_explicit_values():
    cfg = PlayerConfig(participants=["x", "y"], aoi_presets=["guitar"],
                       quick_labels=[str(i) for i in range(12)],
                       depth_hierarchy={"x": ["Y", "X"]}, participant_colors={"x": "#123456"})
    assert cfg.presets_for(["x"]) == ["guitar"]
    assert cfg.quick_labels_for(["x"]) == [str(i) for i in range(9)]  # keys 1-9 only
    assert cfg.depth_order_for("x", []) == ["y", "x"]
    assert cfg.depth_order_for("y", []) == ["x", "y"]
    assert cfg.color_for("x") == "#123456"


def test_round_trip(tmp_path):
    cfg = PlayerConfig(participants=["a"], depth_hierarchy={"a": ["a"]})
    cfg.save(tmp_path / "c.json")
    again = PlayerConfig.from_file(tmp_path / "c.json")
    assert again.to_dict() == cfg.to_dict()
