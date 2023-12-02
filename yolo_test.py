#from ultralytics import YOLO
import ultralytics
import numpy as np
import cv2
import serial
import time

ser = serial.Serial('/dev/ttyACM0', 9600)
time.sleep(2)

model = ultralytics.YOLO('waste_classifier.pt')

warmupFrame = np.zeros((360, 640, 3), dtype=np.uint8)
model.predict(source=warmupFrame, verbose=False)
print(f"Model warmed up")

state = 1
last_state = 1

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    results = model(frame, verbose=False)
    
    boxes = []
    confidences = []
    classids = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            class_id = box.cls[0].item()
            prob = round(box.conf[0].item(), 2)
            if prob > 0.3:
                # Append to list
                boxes.append([x1, y1, x2-x1, y2-y1])
                confidences.append(float(prob))
                classids.append(class_id)

    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        class_name = model.model.names[classids[i]]
        if str(class_name) == "plastic":
            state = 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        elif str(class_name) == "can":
            state = 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, str(confidences[i]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, str(class_name), (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        xmenor = x
        ymenor = y
        xmayor = x + w
        ymayor = y + h

    if state != last_state:
        if state == 1:
            print("plastic")
            ser.write(b'1')
        elif state == 2:
            print("can")
            ser.write(b'2')
        last_state = state
    
    #results.show()
    cv2.imshow("result", frame)
    if cv2.waitKey(1) == ord('q'):
        break