'''
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    success, frame = cap.read()

    if not success:
        print("Cannot read webcam")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )

    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        left_wrist = landmarks[
            mp_pose.PoseLandmark.LEFT_WRIST
        ]

        right_wrist = landmarks[
            mp_pose.PoseLandmark.RIGHT_WRIST
        ]

        left_shoulder = landmarks[
            mp_pose.PoseLandmark.LEFT_SHOULDER
        ]

        right_shoulder = landmarks[
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]

        movement = "Standing"

        if (
            left_wrist.y < left_shoulder.y
            and right_wrist.y < right_shoulder.y
        ):
            movement = "Both Hands Raised"

        elif left_wrist.y < left_shoulder.y:
            movement = "Left Hand Raised"

        elif right_wrist.y < right_shoulder.y:
            movement = "Right Hand Raised"

        cv2.putText(
            frame,
            movement,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "AI Full Body Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
pose.close()
cv2.destroyAllWindows()
'''


import cv2
import mediapipe as mp
import time
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Locate your downloaded model
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "pose_landmarker.task"

if not MODEL.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL}"
    )

# Configure body tracking

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL),
        delegate=python.BaseOptions.Delegate.CPU
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

# Connections between body joints
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (29, 31),
    (28, 30), (30, 32)
]

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Cannot open webcam")

start = time.monotonic()
last_timestamp = -1

with vision.PoseLandmarker.create_from_options(
    options
) as tracker:

    while True:
        success, frame = camera.read()

        if not success:
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

        height, width = frame.shape[:2]

        for pose in result.pose_landmarks:
            points = []

            for landmark in pose:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append((x, y))

                cv2.circle(
                    frame, (x, y), 4,
                    (0, 255, 0), -1
                )

            for a, b in CONNECTIONS:
                cv2.line(
                    frame,
                    points[a],
                    points[b],
                    (255, 0, 0),
                    2
                )

        cv2.imshow(
            "AI Full Body Recognition",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()