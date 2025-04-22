# DeepSORT

[![arXiv](https://img.shields.io/badge/arXiv-1703.07402-b31b1b.svg)](https://arxiv.org/abs/1703.07402)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](...)

## Overview

DeepSORT extends the original [SORT](../sort/tracker.md) algorithm by integrating appearance information through a deep association metric. While maintaining the core Kalman filtering and Hungarian algorithm components from SORT, DeepSORT adds a convolutional neural network (CNN) trained on large-scale person re-identification datasets to extract appearance features from detected objects. This integration allows the tracker to maintain object identities through longer periods of occlusion, effectively reducing identity switches by approximately 45% compared to the original SORT. DeepSORT operates with a dual-metric approach, combining motion information (Mahalanobis distance) with appearance similarity (cosine distance in feature space) to improve data association decisions. It also introduces a matching cascade that prioritizes recently seen tracks, enhancing robustness during occlusions. Most of the computational complexity is offloaded to an offline pre-training stage, allowing the online tracking component to run efficiently at approximately 20Hz, making it suitable for real-time applications while achieving competitive tracking performance with significantly improved identity preservation.


## Examples

=== "ultralytics"

    ```python
    import supervision as sv
    from trackers import DeepSORTFeatureExtractor, DeepSORTTracker
    from ultralytics import YOLO

    model = YOLO("yolo11m.pt")
    feature_extractor = DeepSORTFeatureExtractor.from_timm(
        model_name="mobilenetv4_conv_small.e1200_r224_in1k"
    )
    tracker = DeepSORTTracker(feature_extractor=feature_extractor)
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        detections = model(frame)
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update(detections, frame)
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

::: trackers.core.deepsort.tracker.DeepSORTTracker
