from ultralytics import YOLO
import cv2

model = YOLO('waste_classifier.pt')

results = model(source=0, show=True, conf=0.4, save=True)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break