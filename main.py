from ultralytics import YOLO
import cv2
import cvzone
import time
from Utils import ocr
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', default="street.MOV")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_path)  # For Video
model = YOLO("LP_detector.pt")
char_model = YOLO('Char_detector.pt')


charNames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
             'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
             'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
classNames = ["license_plate"]


prev_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    new_frame_time = time.time()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            crop_img = img[y1:y1 + h, x1:x1 + w]            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            lp = ocr(char_model, crop_img, charNames)

            if lp != "unknown":
                cvzone.putTextRect(img, f'{lp}', (max(0, x1), max(35, y1)), scale=1.5, thickness=1)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, str(int(fps)), (40,70), font, 2, (130,235,34), 2, cv2.LINE_AA)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
