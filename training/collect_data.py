
import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Project paths
ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "pose_landmarker.task"
DATA = ROOT / "data" / "landmarks"

# Recording settings
FRAMES_PER_CLIP = 30

parser = argparse.ArgumentParser()
parser.add_argument("--action", required=True)
parser.add_argument("--clips", type=int, default=5)
args = parser.parse_args()

action = args.action.lower().replace(" ", "_")
save_dir = DATA / action
save_dir.mkdir(parents=True, exist_ok=True)

if not MODEL.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL}"
    )

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL)
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Cannot open webcam")

existing = [
    int(p.stem.split("_")[-1])
    for p in save_dir.glob("clip_*.npy")
    if p.stem.split("_")[-1].isdigit()
]
next_clip = max(existing, default=-1) + 1

start = time.monotonic()
last_timestamp = -1
saved = 0
sequence = []
recording = False

print("Press SPACE to start each recording.")
print("Press Q to quit.")

with vision.PoseLandmarker.create_from_options(
    options
) as tracker:

    while saved < args.clips:
        success, frame = camera.read()

        if not success:
            print("Camera error")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(
            (time.monotonic() - start) * 1000
        )
        timestamp = max(
            timestamp, last_timestamp + 1
        )
        last_timestamp = timestamp

        result = tracker.detect_for_video(
            image, timestamp
        )

        if result.pose_landmarks:
            pose = result.pose_landmarks[0]
            height, width = frame.shape[:2]

            landmarks = []

            for point in pose:
                x = int(point.x * width)
                y = int(point.y * height)

                cv2.circle(
                    frame, (x, y), 4,
                    (0, 255, 0), -1
                )

                landmarks.extend([
                    point.x,
                    point.y,
                    point.z,
                    point.visibility
                ])

            if recording:
                sequence.append(landmarks)

        if recording and len(sequence) >= FRAMES_PER_CLIP:
            filename = (
                save_dir /
                f"clip_{next_clip + saved:04d}.npy"
            )

            np.save(
                filename,
                np.array(sequence, dtype=np.float32)
            )

            print(f"Saved: {filename}")
            saved += 1
            sequence = []
            recording = False

        status = (
            "RECORDING"
            if recording
            else "Press SPACE"
        )

        cv2.putText(
            frame,
            f"Action: {action}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"Clip: {saved + 1}/{args.clips}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"{status}: {len(sequence)}/30",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )

        cv2.imshow(
            "AI Movement Data Collection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32 and not recording:
            sequence = []
            recording = True

camera.release()
cv2.destroyAllWindows()
print(f"Finished! Saved {saved} clips.")