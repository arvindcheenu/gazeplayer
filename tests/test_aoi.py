from gazeplayer import AOIManager, BoundingBox, PlayerConfig


def test_bbox_geometry():
    b = BoundingBox(0, 0, 10, 10)
    assert b.contains(5, 5) and not b.contains(11, 5)
    assert b.distance_to(13, 14) == 5.0
    assert b.overlap_fraction(BoundingBox(5, 0, 10, 10)) == 0.5
    assert not BoundingBox(200, 0, 5, 5).is_in_bounds(100, 100)


def test_keyframe_holds_until_next(tmp_path):
    m = AOIManager(tmp_path)
    m.create_aoi("guitar", {"p": 10}, BoundingBox(0, 0, 10, 10))
    m.update_aoi("guitar", "p", 20, BoundingBox(50, 0, 10, 10))
    aoi = m.aois["guitar"]
    assert aoi.get_bbox_at_frame("p", 5) is None
    assert aoi.get_bbox_at_frame("p", 15).x == 0
    assert aoi.get_bbox_at_frame("p", 25).x == 50
    assert aoi.get_bbox_at_frame("other", 15) is None


def test_isolated_edit_pins_previous_frame(tmp_path):
    m = AOIManager(tmp_path)
    m.create_aoi("a", {"p": 0}, BoundingBox(0, 0, 10, 10))
    m.update_aoi_isolated("a", "p", 30, BoundingBox(5, 5, 10, 10))
    aoi = m.aois["a"]
    assert aoi.is_anchor_frame("p", 29)
    assert not aoi.is_anchor_frame("p", 30)
    m.delete_keyframe("a", "p", 30)
    assert aoi.get_bbox_at_frame("p", 30).x == 0


def test_persistence(tmp_path):
    m = AOIManager(tmp_path)
    m.create_aoi("a", {"p": 0}, BoundingBox(1, 2, 3, 4))
    m.rename_aoi("a", "b")
    again = AOIManager(tmp_path)
    assert list(again.aois) == ["b"]
    assert again.aois["b"].get_bbox_at_frame("p", 0).to_dict() == \
        {"x": 1, "y": 2, "width": 3, "height": 4}


def _two_boxes(tmp_path, cfg=None):
    m = AOIManager(tmp_path, cfg)
    m.create_aoi("x", {"p": 0}, BoundingBox(0, 0, 100, 100))
    m.create_aoi("y", {"p": 0}, BoundingBox(300, 300, 100, 100))
    return m


def test_classify_inside_near_and_far(tmp_path):
    m = _two_boxes(tmp_path)
    assert m.classify_gaze_aoi("p", 0, 50, 50, 1000, 1000) == "x"
    assert m.classify_gaze_aoi("p", 0, 130, 50, 1000, 1000) == "x"     # 30 px away
    assert m.classify_gaze_aoi("p", 0, 200, 50, 1000, 1000) == "Other"


def test_proximity_is_configurable(tmp_path):
    m = _two_boxes(tmp_path, PlayerConfig(gaze_aoi_proximity_px=10))
    assert m.classify_gaze_aoi("p", 0, 130, 50, 1000, 1000) == "Other"


def test_point_matching_every_aoi_is_other(tmp_path):
    m = AOIManager(tmp_path)
    m.create_aoi("x", {"p": 0}, BoundingBox(0, 0, 100, 100))
    m.create_aoi("y", {"p": 0}, BoundingBox(50, 0, 100, 100))
    assert m.classify_gaze_aoi("p", 0, 75, 50, 1000, 1000) == "Other"


def test_depth_hierarchy_breaks_ties(tmp_path):
    def mgr(order, sub):
        m = AOIManager(tmp_path / sub, PlayerConfig(depth_hierarchy={"p": order}))
        # identical boxes -> identical overlap; a third AOI elsewhere so that
        # "all AOIs matched" does not apply
        m.create_aoi("near", {"p": 0}, BoundingBox(0, 0, 100, 100))
        m.create_aoi("far", {"p": 0}, BoundingBox(0, 0, 100, 100))
        m.create_aoi("elsewhere", {"p": 0}, BoundingBox(800, 800, 10, 10))
        return m
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    assert mgr(["near", "far"], "a").classify_gaze_aoi("p", 0, 50, 50, 1000, 1000) == "near"
    assert mgr(["far", "near"], "b").classify_gaze_aoi("p", 0, 50, 50, 1000, 1000) == "far"
