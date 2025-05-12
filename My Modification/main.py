import os
import sys

#print("You need youtuble link (For Exaple \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\") and output file name (For exaple \"output\") to run the program.")

if len(sys.argv) != 3:
    print("You need youtuble link (For Exaple \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\") and output file name (For exaple \"output\") to run the program.")
    exit() 
else:
    param1 = sys.argv[1]
    param2 = sys.argv[2]
    print(f"Parameter 1: {param1}")
    print(f"Parameter 2: {param2}")


from inference import get_model
from trackers import SORTTracker

model = get_model("yolov8m-640")
tracker = SORTTracker()

import supervision as sv

color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

box_annotator = sv.BoxAnnotator(
    color=color,
    color_lookup=sv.ColorLookup.TRACK)

trace_annotator = sv.TraceAnnotator(
    color=color,
    color_lookup=sv.ColorLookup.TRACK,
    thickness=2,
    trace_length=100)

label_annotator = sv.LabelAnnotator(
    color=color,
    color_lookup=sv.ColorLookup.TRACK,
    text_color=sv.Color.BLACK,
    text_scale=0.8)

def track_video(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH):
    CONFIDENCE_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.3

    frame_samples = []

    def callback(frame, i):
        result = model.infer(frame, confidence=CONFIDENCE_THRESHOLD)[0]
        detections = sv.Detections.from_inference(result).with_nms(threshold=NMS_THRESHOLD)
        detections = tracker.update(detections)

        annotated_image = frame.copy()
        annotated_image = box_annotator.annotate(annotated_image, detections)
        annotated_image = trace_annotator.annotate(annotated_image, detections)
        annotated_image = label_annotator.annotate(annotated_image, detections, detections.tracker_id)

        if i % 30 == 0 and i != 0:
            frame_samples.append(annotated_image)

        return annotated_image

    tracker.reset()

    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=callback,
        show_progress=True,
    )
    #sv.plot_images_grid(images=frame_samples[:4], grid_size=(2, 2))


def download_and_convert_video(link, output_name):
    import yt_dlp

    url = link

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': f'{output_name}.mp4',  # Output filename
        'merge_output_format': 'mp4',  # Force output format
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print("Download and conversion to MP4 complete.")
    return f"{output_name}.mp4"


def download_and_convert_video(link, output_name = "temp"):
    import yt_dlp

    url = link

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': f'{output_name}.mp4',  # Output filename
        'merge_output_format': 'mp4',  # Force output format
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print("Download and conversion to MP4 complete.")
    return f"{output_name}.mp4"

def download_convert_and_track(link, output_name = "temp_output"):
    name = "temp"
    downalded_file = download_and_convert_video(
        link = link, 
        output_name="temp"
    )
    track_video("temp.mp4", f"{output_name}.mp4")
    os.remove("temp.mp4")


download_convert_and_track(link = sys.argv[1], output_name = sys.argv[2])
