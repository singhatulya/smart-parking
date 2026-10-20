"""
Capture labelled images for Smart Parking AI.
Usage:
  python capture_dataset.py
Keys:
  1 = capture EMPTY example
  2 = capture OCCUPIED example
  q = quit
Place the camera so it sees the SAME parking area you will use during prediction.
"""
import cv2
from pathlib import Path
from datetime import datetime

ROOT = Path("dataset")
EMPTY = ROOT / "empty"
OCCUPIED = ROOT / "occupied"
EMPTY.mkdir(parents=True, exist_ok=True)
OCCUPIED.mkdir(parents=True, exist_ok=True)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise SystemExit("Could not open webcam. Try changing VideoCapture(0) to VideoCapture(1).")

print("Press 1 to save EMPTY, 2 to save OCCUPIED, q to quit.")
print("Collect varied examples in different lighting and with different vehicles/objects.")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Could not read from camera.")
        break

    view = frame.copy()
    cv2.putText(view, "1: EMPTY  |  2: OCCUPIED  |  q: QUIT",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.imshow("Smart Parking - Dataset Capture", view)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key in (ord("1"), ord("2")):
        folder = EMPTY if key == ord("1") else OCCUPIED
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = folder / f"{folder.name}_{stamp}.jpg"
        if cv2.imwrite(str(path), frame):
            print(f"Saved: {path}")

camera.release()
cv2.destroyAllWindows()
