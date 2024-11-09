import os

from ultralytics import YOLO
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import torch
import GUH24.face_recognition.model.face_recognition as face_recognition

def extract_boxes(results):
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




class CameraApp(QMainWindow):
    def __init__(self, know_faces):
        super().__init__()

        self.known_faces = know_faces

        self.model = YOLO("yolo11n.pt")
        self.cap = cv2.VideoCapture(0)
        self.setWindowTitle("Camera Inference App")
        self.video_label = QLabel(self)
        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_camera)
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)


    def start_camera(self):
        """Start or stop the camera feed."""
        if not self.timer.isActive():
            self.cap.open(0)
            self.timer.start(30)
            self.start_button.setText("Stop")
        else:
            self.timer.stop()
            self.cap.release()
            self.video_label.clear()
            self.start_button.setText("Start")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            detections = self.run_inference(frame)

            # Draw detections on the frame
            for det_class in detections:
                for bndbox in detections[det_class]:
                    x1, y1, x2, y2, confidence = bndbox

                    # w_ratio = frame.shape[1] / 640
                    # h_ratio = frame.shape[0] / 640
                    # x1 = int(x1 * w_ratio)
                    # y1 = int(y1 * h_ratio)
                    # x2 = int(x2 * w_ratio)
                    # y2 = int(y2 * h_ratio)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{det_class} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def run_inference(self, frame):
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
                        label_name = "BANIKA"


            if label not in detections:
                detections[label_name] = []
            detections[label_name].append([x1, y1, x2, y2, confidence])

        return detections


if __name__ == "__main__":


    banica_faces_dir = "/Users/banika/Desktop/GreatUniHack/GUH24/me_pictures"
    banica_faces = [face_recognition.load_image_file(f"{banica_faces_dir}/{f}") for f in os.listdir(banica_faces_dir)]
    banica_face_encodings = [face_recognition.face_encodings(face)[0] for face in banica_faces]


    app = QApplication(sys.argv)
    window = CameraApp(know_faces=banica_face_encodings)
    window.show()
    sys.exit(app.exec_())




