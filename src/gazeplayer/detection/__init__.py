"""Optional pre-processing with Ultralytics models (``pip install "gazeplayer[detect]"``).

* :mod:`gazeplayer.detection.yolo` – YOLO + ByteTrack → ``yolo_detections.json``
* :mod:`gazeplayer.detection.fastsam` – FastSAM masks → ``fastsam_masks.json``

Both modules import ``ultralytics`` lazily, so the player runs without it.
"""
