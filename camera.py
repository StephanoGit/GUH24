import cv2
import threading
import time
from ultralytics import YOLO
import face_recognition as face_recognition
import torch, numpy as np, os

class WebcamRecorder(threading.Thread):
    def __init__(self, video_source=0):
        super().__init__()
        self.video_source = video_source
        self.capture = cv2.VideoCapture(self.video_source)
        self.frame = None
        self.is_running = True
        self.model = YOLO("yolo11n.pt")

        # load the known faces
        known_faces_dir = "known_faces_dir"
        known_faces = [face_recognition.load_image_file(f"{known_faces_dir}/{f}") for f in os.listdir(known_faces_dir)]
        self.known_faces = [face_recognition.face_encodings(face)[0] for face in known_faces]

        # Check if the camera opened successfully
        if not self.capture.isOpened():
            raise ValueError(f"Unable to open video source {self.video_source}")

    def run(self):
        try:
            while self.is_running:
                ret, frame = self.capture.read()
                if not ret:
                    print("Warning: Frame capture failed.")
                    continue  # Skip this iteration
                self.frame = frame
                time.sleep(0.01)  # Adjust sleep time if needed
        finally:
            self.capture.release()

    def stop(self):
        """Stop the video recording."""
        self.is_running = False
        self.capture.release()

    def extract_boxes(self, results):
        detections = {}
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates
            confidence = box.conf.item()  # Extract confidence
            label = int(box.cls.item())  # Extract class label index
            label_name = results[0].names[label].split()[-1]

            if label not in detections:
                detections[label_name] = []

            detections[label_name].append([x1, y1, x2, y2, confidence])
        return detections

    def run_inference(self, frame):
        if frame is None:
            return {}

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))  # Resize as needed for model
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize to [0,1]
        results = self.model(img)
        detections = {}
        boxes = results[0].boxes  # boxes is a Boxes object from Ultralytics

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w_ratio = frame.shape[1] / 640
            h_ratio = frame.shape[0] / 640
            x1 = int(x1 * w_ratio)
            y1 = int(y1 * h_ratio)
            x2 = int(x2 * w_ratio)
            y2 = int(y2 * h_ratio)
            confidence = box.conf.item()
            label = int(box.cls.item())

            try:
                label_name = results[0].names[label].split()[-1]
            except AttributeError:
                label_name = "Unknown"

            if label_name.lower() == "person":
                person_subframe = frame[y1:y2, x1:x2]
                person_encoding = face_recognition.face_encodings(np.array(person_subframe))

                if len(person_encoding) > 0:
                    face_result = face_recognition.compare_faces(self.known_faces, person_encoding[0])
                    if face_result:
                        label_name = "KNOWN PERSON"

            if label not in detections:
                detections[label_name] = []
            detections[label_name].append([x1, y1, x2, y2, confidence])

        return detections

    def get_frame(self):
        frame = self.frame
        detections = self.run_inference(frame)
        for det_class in detections:
            for bndbox in detections[det_class]:
                x1, y1, x2, y2, confidence = bndbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{det_class} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
        return frame, detections
