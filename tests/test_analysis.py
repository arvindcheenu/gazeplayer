import pandas as pd

from gazeplayer import PlayerConfig, export_metrics_csv, load_session


def test_dwell_and_transitions(session_with_aois):
    session, aois, tracker = load_session(session_with_aois)
    metrics = tracker.compute_all_metrics()
    assert set(metrics) == {"Alice", "bob"}

    m = metrics["Alice"]
    n_gaze = len(session.participants["Alice"].gaze)
    assert m.first_fixation_aoi == "left"
    assert m.dwell_time_ms["left"] == m.dwell_time_ms["right"]
    assert sum(m.dwell_time_ms.values()) == 5 * n_gaze
    assert m.transitions == {("left", "right"): 1}
    session.close()


def test_sample_duration_is_configurable(session_with_aois):
    _, _, t5 = load_session(session_with_aois)
    _, _, t4 = load_session(session_with_aois, PlayerConfig(gaze_sample_ms=4))
    d5 = t5.compute_all_metrics()["bob"].dwell_time_ms["left"]
    d4 = t4.compute_all_metrics()["bob"].dwell_time_ms["left"]
    assert d4 == d5 * 4 / 5


def test_transition_events(session_with_aois):
    session, _, tracker = load_session(session_with_aois)
    events = tracker.generate_all_transition_events()
    assert [(e["participant"], e["name"]) for e in events] == [
        ("Alice", "gaze_left_to_right"), ("bob", "gaze_left_to_right")]
    session.close()


def test_export_csv(session_with_aois, tmp_path):
    _, _, tracker = load_session(session_with_aois)
    written = export_metrics_csv(tracker.compute_all_metrics(), tmp_path / "out")
    assert [p.name for p in written] == [
        "metrics_dwell_time.csv", "metrics_transitions.csv", "metrics_summary.csv"]
    dwell = pd.read_csv(written[0])
    assert set(dwell["aoi"]) == {"left", "right"}
    assert dwell.groupby("participant")["dwell_percent"].sum().round(6).eq(100).all()
