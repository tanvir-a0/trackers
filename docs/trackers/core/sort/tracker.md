# SORT

[![arXiv](https://img.shields.io/badge/arXiv-1602.00763-b31b1b.svg)](https://arxiv.org/abs/1602.00763)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](...)

## Overview

SORT (Simple Online and Realtime Tracking) is a lean, tracking-by-detection method that combines a Kalman filter for motion prediction with the Hungarian algorithm for data association. It uses object detections—commonly from a high-performing CNN-based detector—as its input, updating each tracked object’s bounding box based on linear velocity estimates. Because SORT relies on minimal appearance modeling (only bounding box geometry is used), it is extremely fast and can run comfortably at hundreds of frames per second. This speed and simplicity make it well suited for real-time applications in robotics or surveillance, where rapid, approximate solutions are essential. However, its reliance on frame-to-frame matching makes SORT susceptible to ID switches and less robust during long occlusions, since there is no built-in re-identification module.

## Examples

=== "ultralytics"

    ```python
    import supervision as sv
    from trackers import SORTTracker
    from ultralytics import YOLO

    model = YOLO("yolo11m.pt")
    tracker = SORTTracker()
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        detections = model(frame)
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update(detections)
        detections = detections[detections.tracker_id != -1]

        labels = [str(t) for t in detections.tracker_id]
        return annotator.annotate(frame, detections, labels)

    sv.process_video(
        source_path=<SOURCE_VIDEO_PATH>,
        target_path=<TARGET_VIDEO_PATH>,
        callback=callback,
    )
    ```

## Usage

::: trackers.core.sort.tracker.SORTTracker
