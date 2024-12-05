import os
import cv2
import torch
import argparse
from ultralytics import YOLO
from sort.sort import Sort
import numpy as np
from collections import deque
import tqdm
from onnxruntime.quantization import quantize_dynamic, QuantType
import tempfile


# Function to calculate speed based on displacement between frames
def calculate_speed(displacement_pixels, time_interval_s, scale):
    speed_m_s = (displacement_pixels * scale) / time_interval_s
    return speed_m_s * 3.6  # Convert speed to km/h

# Command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Jet Ski tracking and speed estimation.")
    parser.add_argument(
        "--video",
        type=str,
        default="C:\\project_detect_buke\\CVAT2YOLO-main\\my_dataset_yolo\\test_jetski.mp4",
        help="Path to the input video file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tracked_output.mp4",
        help="Path for saving the output video."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=r"C:\\project_detect_buke\\CVAT2YOLO-main\\my_dataset_yolo\\runs\\detect\\train11\\weights\\best.pt",
        help="Path to the YOLO model weights."
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="C:\\project_detect_buke\\CVAT2YOLO-main\\my_dataset_yolo\\runs\\detect\\train11\\weights\\best.onnx",
        help="Path to save the exported ONNX model."
    )
    parser.add_argument(
        "--quantized_path",
        type=str,
        default="quantized_model.onnx",
        help="Path to save the quantized ONNX model."
    )
    return parser.parse_args()

# Main function to process the video
def process_video(args):
    # Disable caching and set temporary directory for Ultralytics
    os.environ['YOLO_CACHE'] = '0'
    os.environ['TMPDIR'] = tempfile.gettempdir()

    # Check if the model file exists
    if not os.path.isfile(args.model_path):
        print(f"Model file not found at path: {args.model_path}")
        exit(1)

    JET_SKI_LENGTH = 3.0  # Length of Jet Ski in meters
    SMOOTHING_WINDOW = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the YOLO model
    model = YOLO(args.model_path)
    model.to(device)

    # Export the model to ONNX format
    print("Exporting the model to ONNX format...")
    onnx_model_path = model.export(format="onnx", imgsz=640, device=device, half=False)
    print(f"Model successfully exported to {onnx_model_path}")

    # Check if the ONNX model file exists
    if not os.path.isfile(onnx_model_path):
        print(f"ONNX model not found at path: {onnx_model_path}")
        exit(1)

    # Quantize the ONNX model
    print("Quantizing the model...")
    quantize_dynamic(
        onnx_model_path,
        args.quantized_path,
        weight_type=QuantType.QInt8
    )
    print(f"Quantized model saved at {args.quantized_path}")

    # Initialize the tracker
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # Open the video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Failed to open video file: {args.video}")
        exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video FPS: {fps}, Resolution: {frame_width}x{frame_height}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    scale = None  # Scale to be calculated
    object_data = {}  # Dictionary to store object data (tracks)

    # Process the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm.tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=640, conf=0.4, device=device)
        detections = []

        if results[0].boxes.xyxy is not None and len(results[0].boxes.xyxy) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1  # Width of bounding box (in pixels)
                if width <= 0:
                    continue  # Skip incorrect boxes
                detections.append([x1, y1, x2, y2, conf])  # Format for SORT: [x1, y1, x2, y2, score]

                # Calibrate scale based on known Jet Ski length
                if scale is None:
                    scale = JET_SKI_LENGTH / width  # Determine scale (meters per pixel)

        # Process detections
        if len(detections) > 0:
            detections = np.array(detections)
            tracked_objects = tracker.update(detections)
        else:
            tracked_objects = []

        # Calculate speed
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, track)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Object center

            # Check if track ID is in object data
            if track_id not in object_data:
                object_data[track_id] = {
                    "last_position": (cx, cy),
                    "speeds": deque(maxlen=SMOOTHING_WINDOW),
                }

            # Calculate displacement and speed
            last_cx, last_cy = object_data[track_id]["last_position"]
            displacement_pixels = np.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)
            object_data[track_id]["last_position"] = (cx, cy)

            if scale is not None and fps > 0:
                speed_kmh = calculate_speed(displacement_pixels, 1 / fps, scale)
                # Save speed to deque for smoothing
                object_data[track_id]["speeds"].append(speed_kmh)
                smoothed_speed = np.mean(object_data[track_id]["speeds"])  # Smoothed speed
            else:
                smoothed_speed = 0.0

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID: {track_id} Speed: {smoothed_speed:.2f} km/h",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

        out.write(frame)
        cv2.imshow("Jet Ski Tracking", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")

if __name__ == "__main__":
    args = parse_arguments()
    process_video(args)
