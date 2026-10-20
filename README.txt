SMART PARKING AI — CAMERA + PICTURE CLASSIFIER
==============================================

FILES
-----
capture_dataset.py  Captures labelled photos from your webcam.
train_model.py      Trains an EMPTY/OCCUPIED image classifier.
live_camera.py      Opens the webcam and predicts the slot status.

1) INSTALL PYTHON
Use Python 3.10 or 3.11 for a compatible TensorFlow setup.
During installation, enable "Add Python to PATH" on Windows.

2) OPEN A TERMINAL IN THIS FOLDER
Install packages:
    python -m pip install --upgrade pip
    python -m pip install tensorflow opencv-python numpy

3) COLLECT YOUR OWN PHOTOS
Run:
    python capture_dataset.py

Frame ONE parking space consistently.
Press 1 when it is genuinely empty.
Press 2 when it is genuinely occupied.
Press q to exit.

Collect plenty of varied, correctly labelled examples for BOTH classes,
including different lighting and vehicles. Do not save hundreds of nearly
identical frames. Keep the camera angle and region similar to real use.

4) TRAIN THE MODEL
Run:
    python train_model.py

The script creates parking_model.keras. It needs an internet connection
the first time to obtain MobileNetV2's pretrained ImageNet weights.

5) START LIVE DETECTION
Run:
    python live_camera.py

Press q in the video window to quit.
If the webcam does not open, edit cv2.VideoCapture(0) to use 1 instead.

IMPORTANT NOTES
---------------
- This is a starter prototype, not a guaranteed production detector.
- It classifies the camera view as a whole. It is intended for ONE fixed
  parking-space view at a time.
- It does not automatically identify several separate slots in one wide view.
- Model quality depends on your labelled photos and camera setup.
- Keep some images from separate sessions/conditions for a fair test.
- Low confidence displays CHECK instead of making a confident claim.
- No Arduino/LED integration is included yet; this first version runs on PC.
