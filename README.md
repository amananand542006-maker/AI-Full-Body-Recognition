# AI Full-Body Recognition

A Python computer-vision prototype that tracks a person from a webcam with the MediaPipe Pose Landmarker and OpenCV. It can display body landmarks in real time and collect fixed-length landmark sequences for future action-recognition training.

## Current Status

The real-time pose viewer and landmark data collector are implemented. The action-recognition model, feature extractor, hand tracker, drawing helpers, evaluation script, and Streamlit dashboard are currently placeholders and are documented here so the project can be extended without guessing about the intended workflow.

## Features

- Real-time webcam pose tracking with MediaPipe Tasks.
- CPU inference using the bundled `models/pose_landmarker.task` model.
- Mirrored camera preview with landmark points and body connections.
- Fixed-length sequence collection for labeled actions.
- NumPy landmark files stored under `data/landmarks/<action>/`.
- VS Code recommendations for Python, Pylance, and Jupyter.

## Requirements

- macOS, Windows, or Linux.
- Python 3.11 or 3.12. Python 3.12 is the documented environment.
- A webcam and permission for Python/VS Code to access it.
- A GUI session for OpenCV windows. Headless servers cannot run the webcam UI.

## Setup

From the project root:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv312\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

The repository includes the required pose model at `models/pose_landmarker.task`; no model download is needed for the current scripts.

## Run Real-Time Pose Tracking

```bash
source .venv312/bin/activate
python app.py
```

The camera window displays the tracked landmarks. Press `q` to quit. If the camera cannot open, check macOS **System Settings > Privacy & Security > Camera** and grant access to the terminal or VS Code application.

## Collect Training Data

The collector records 30 frames per clip. Start it with an action label and desired clip count:

```bash
python training/collect_data.py --action standing --clips 5
```

Press `SPACE` to begin each clip and `q` to stop. Each completed clip is saved as a `.npy` array in `data/landmarks/standing/` with shape `(30, 33 * 4)`: normalized `x`, `y`, `z`, and visibility for each of MediaPipe's 33 pose landmarks.

Use one directory per action, for example:

```bash
python training/collect_data.py --action waving --clips 30
python training/collect_data.py --action sitting --clips 30
```

## Project Layout

```text
.
├── app.py                         # Live webcam pose viewer
├── requirements.txt               # Pinned Python dependencies
├── dashboard/streamlit_app.py    # Reserved dashboard entry point
├── data/landmarks/                # Collected landmark sequences
├── models/pose_landmarker.task    # MediaPipe pose model
├── src/                           # Reusable modules planned for extraction
└── training/
    ├── collect_data.py            # Webcam sequence collection
    ├── train.py                   # Reserved model-training entry point
    └── evaluate.py                # Reserved evaluation entry point
```

## Data Format

Each collected file is a NumPy float32 array. Every row represents one video frame. Values are stored in landmark order, with four values per landmark:

```text
[x, y, z, visibility] * 33
```

The `x` and `y` coordinates are normalized to the image dimensions by MediaPipe. The current collector saves only frames where a pose is detected, so a clip may take longer than 30 camera frames to complete.

## VS Code Extensions

The workspace recommends:

- **Python** (`ms-python.python`) for interpreter and environment support.
- **Pylance** (`ms-python.vscode-pylance`) for type checking and IntelliSense.
- **Jupyter** (`ms-toolsai.jupyter`) for inspecting `.npy` data and experiments.

Open the Extensions view and choose **Install Workspace Recommended Extensions** when prompted. The recommendations are stored in `.vscode/extensions.json`.

## Troubleshooting

### `source .venv/bin/activate` fails

Use the documented environment name:

```bash
source .venv312/bin/activate
```

The activation command must be run from the project root, or use the full path to `.venv312/bin/activate`.

### MediaPipe import or version errors

Confirm the active interpreter and reinstall the pinned dependencies:

```bash
which python
python --version
python -m pip install -r requirements.txt
```

### `Model not found`

Run the command from this repository or keep `models/pose_landmarker.task` at the documented path. The scripts resolve paths relative to the project files, not the shell's current directory.

### Camera opens but no landmarks appear

Improve lighting, move fully into view, and keep the whole body inside the frame. The app is configured for one pose and CPU inference.

## Development Notes

The next implementation milestone is to move the repeated tracking logic from `app.py` and `training/collect_data.py` into `src/pose_tracker.py`, then implement feature extraction and a sequence classifier in `src/action_recognizer.py`. Training and dashboard code should be added after the data format and action-label contract are finalized.

## License

No license has been specified yet.