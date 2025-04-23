<div align="center">
    <h1 align="center">trackers</h1>
    <img width="200" src="docs/assets/logo-trackers-violet.svg" alt="trackers logo">

[![version](https://badge.fury.io/py/trackers.svg)](https://badge.fury.io/py/trackers)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/roboflow/trackers/blob/main/LICENSE.md)
[![python-version](https://img.shields.io/pypi/pyversions/trackers)](https://badge.fury.io/py/trackers)

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VT_FYIe3kborhWrfKKBqqfR0EjQeQNiO?usp=sharing)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)
</div>

## Hello

`trackers` is a unified library offering clean room re-implementations of leading multi-object tracking algorithms. Its modular design allows you to easily swap trackers and integrate them with object detectors from various libraries like `ultralytics`, `inference`, `mmdetection`, or `transformers`.

## Installation

Pip install the `trackers` package in a [**Python>=3.9**](https://www.python.org/) environment.

```bash
pip install trackers
```

<details>
<summary>Install from source</summary>

<br>

By installing `trackers` from source, you can explore the most recent features and enhancements that have not yet been officially released. Please note that these updates are still in development and may not be as stable as the latest published release.

```bash
pip install git+https://github.com/roboflow/trackers.git
```

</details>

## Quickstart

With a modular design, `trackers` lets you combine object detectors from different libraries (such as `ultralytics`, `inference`, `mmdetection`, or `transformers`) with the tracker of your choice. Here's how you can use `SORTTracker` with various detectors:

```python
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

```python
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

https://github.com/user-attachments/assets/910490b3-32a0-4b7f-8b84-5b50aa83e004

## License

The code is released under the [Apache 2.0 license](https://github.com/roboflow/trackers/blob/main/LICENSE).

## Contribution

We welcome all contributions—whether it’s reporting issues, suggesting features, or submitting pull requests. Please read our [contributor guidelines](https://github.com/roboflow/trackers/blob/main/CONTRIBUTING.md) to learn about our processes and best practices.
