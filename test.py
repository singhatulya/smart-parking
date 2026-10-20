
import cv2
import time

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows camera

if not camera.isOpened():
    print("❌ Camera could not open.")
    raise SystemExit

print("Camera opened. Waiting for frames...")

time.sleep(2)

for i in range(30):
    success, frame = camera.read()

    if success:
        print("✅ Frame captured:", frame.shape)
        cv2.imshow("Camera Test - Press Q to exit", frame)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break
    else:
        print(f"❌ Frame {i + 1} failed")
        time.sleep(0.2)

camera.release()
cv2.destroyAllWindows()
