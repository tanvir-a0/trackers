<div align="center">
  <img src="assets/logo-trackers-violet.svg" alt="Trackers Logo" width="200" height="200">
</div>

# Installation

You can install `trackers` in a [**Python>=3.9**](https://www.python.org/) environment.

!!! example "Basic Installation"

    === "pip"
        ```bash
        pip install trackers
        ```

    === "poetry"
        ```bash
        poetry add trackers
        ```

    === "uv"
        ```bash
        uv pip install trackers
        ```

!!! example "Hardware Acceleration"

    === "CPU"
        ```bash
        pip install "trackers[cpu]"
        ```

    === "CUDA 11.8"
        ```bash
        pip install "trackers[cu118]"
        ```

    === "CUDA 12.4"
        ```bash
        pip install "trackers[cu124]"
        ```

    === "CUDA 12.6"
        ```bash
        pip install "trackers[cu126]"
        ```

    === "ROCm 6.1"
        ```bash
        pip install "trackers[rocm61]"
        ```

    === "ROCm 6.2.4"
        ```bash
        pip install "trackers[rocm624]"
        ```

# Quickstart

=== "inference"

    ```python hl_lines="2 5 12"
    import supervision as sv
    from trackers import SORTTracker
    from inference import get_model

    tracker = SORTTracker()
    model = get_model(model_id="yolov11m-640")
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update(detections)
        return annotator.annotate(frame, detections, labels=detections.tracker_id)

    sv.process_video(
        source_path="input.mp4",
        target_path="output.mp4",
        callback=callback,
    )
    ```

=== "RF-DETR"

    ```python hl_lines="2 5 11"
    import supervision as sv
    from trackers import SORTTracker
    from rfdetr import RFDETRBase

    tracker = SORTTracker()
    model = RFDETRBase()
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        detections = model.predict(frame)
        detections = tracker.update(detections)
        return annotator.annotate(frame, detections, labels=detections.tracker_id)

    sv.process_video(
        source_path="input.mp4",
        target_path="output.mp4",
        callback=callback,
    )
    ```

=== "ultralytics"

    ```python hl_lines="2 5 12"
    import supervision as sv
    from trackers import SORTTracker
    from ultralytics import YOLO

    tracker = SORTTracker()
    model = YOLO("yolo11m.pt")
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update(detections)
        return annotator.annotate(frame, detections, labels=detections.tracker_id)

    sv.process_video(
        source_path="input.mp4",
        target_path="output.mp4",
        callback=callback,
    )
    ```

=== "transformers"

    ```python hl_lines="3 6 28"
    import torch
    import supervision as sv
    from trackers import SORTTracker
    from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

    tracker = SORTTracker()
    image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
    model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        inputs = image_processor(images=frame, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        h, w, _ = frame.shape
        results = image_processor.post_process_object_detection(
            outputs, 
            target_sizes=torch.tensor([(h, w)]), 
            threshold=0.5
        )[0]
        
        detections = sv.Detections.from_transformers(
            transformers_results=results,
            id2label=model.config.id2label
        )
        
        detections = tracker.update(detections)
        return annotator.annotate(frame, detections, labels=detections.tracker_id)

    sv.process_video(
        source_path="input.mp4",
        target_path="output.mp4",
        callback=callback,
    )
    ```
