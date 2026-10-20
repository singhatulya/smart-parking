"""
Live Smart Parking AI camera.
First train parking_model.keras with train_model.py.
Press q to quit.
"""
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "parking_model.keras"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.75

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as exc:
    raise SystemExit(
        f"Could not load {MODEL_PATH}. Run train_model.py successfully first. Details: {exc}"
    )

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise SystemExit("Could not open webcam. Try changing VideoCapture(0) to VideoCapture(1).")

print("Live prediction running. Press q in the camera window to exit.")
print("Keep the parking slot framed consistently with the training images.")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Could not read camera frame.")
        break

    # OpenCV frames are BGR; Keras image models expect RGB.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)

    occupied_probability = float(model.predict(batch, verbose=0)[0][0])
    confidence = max(occupied_probability, 1.0 - occupied_probability)

    if confidence < CONFIDENCE_THRESHOLD:
        label = "CHECK - LOW CONFIDENCE"
    elif occupied_probability >= 0.5:
        label = "OCCUPIED"
    else:
        label = "EMPTY"

    # Use neutral display text; predictions are estimates, not guaranteed truth.
    cv2.rectangle(frame, (10, 10), (470, 90), (25, 25, 25), -1)
    cv2.putText(frame, label, (22, 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Confidence: {confidence:.0%}", (22, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Smart Parking AI - Live Camera", frame)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
