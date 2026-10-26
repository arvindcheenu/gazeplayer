from gazeplayer import PlayerConfig, SessionData, discover_participants


def test_discovers_folders_with_data(session_dir):
    found = discover_participants(session_dir)
    assert list(found) == ["Alice", "bob"]          # 'notes/' has no data


def test_configured_names_match_case_insensitively_and_keep_order(session_dir):
    found = discover_participants(session_dir, ["BOB", "alice", "carol"])
    assert list(found) == ["BOB", "alice"]
    assert found["alice"].name == "Alice"


def test_unmatched_config_falls_back_to_discovery(session_dir):
    assert list(discover_participants(session_dir, ["zed"])) == ["Alice", "bob"]


def test_load_and_sync(session_dir):
    s = SessionData(session_dir)
    assert set(s.participants) == {"Alice", "bob"}
    alice = s.participants["Alice"]
    assert alice.total_frames == 60
    assert (alice.frame_width, alice.frame_height) == (160, 120)

    sync = s.sync_info
    # bob starts 0.5 s later, so the shared window starts at bob's first frame
    assert sync.overlap_start_ns == s.participants["bob"].start_timestamp
    assert abs(sync.participant_offsets["Alice"] + 0.5) < 1e-6
    assert sync.participant_offsets["bob"] == 0
    assert sync.participant_start_frames["bob"] == 0
    assert sync.participant_start_frames["Alice"] in (15, 16)
    s.close()


def test_frame_and_gaze_lookup(session_dir):
    s = SessionData(session_dir, PlayerConfig(participants=["bob"]))
    assert list(s.participants) == ["bob"]
    bob = s.participants["bob"]
    frame, idx = bob.get_frame_at_timestamp(bob.start_timestamp)
    assert idx == 0 and frame.shape == (120, 160, 3)
    assert bob.get_gaze_at_timestamp(bob.start_timestamp) == (20.0, 20.0)
    assert bob.get_current_fixation(bob.start_timestamp) is None
    s.close()


def test_events_round_trip(session_dir):
    s = SessionData(session_dir)
    ts = s.sync_info.overlap_start_ns + 1_000_000
    s.add_event({"timestamp [ns]": ts, "name": "downbeat", "type": "musical_event",
                 "participant": "all", "notes": "bar 1"})
    s.close()
    again = SessionData(session_dir)
    assert list(again.events["name"]) == ["downbeat"]
    assert again.event_notes[ts] == "bar 1"
    again.close()
