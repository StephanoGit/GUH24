from ultralytics import YOLO
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import torch

def extract_boxes(results):
    """
    Extract bounding boxes and labels from the results of the Ultralytics YOLO model.

    Args:
        results: The output from the model, which contains the Boxes object in results[0].

    Returns:
        A list of bounding boxes and labels in the format (x1, y1, x2, y2, label).
    """
    detections = {}

    # Access the boxes from results[0]
    boxes = results[0].boxes  # boxes is a Boxes object from Ultralytics

    # Iterate over each detected box
    for box in boxes:
        # box.xyxy[0] gives [x1, y1, x2, y2] coordinates
        # box.conf gives the confidence
        # box.cls gives the class label index
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates
        confidence = box.conf.item()  # Extract confidence
        label = int(box.cls.item())  # Extract class label index
        label_name = results[0].names[label].split()[-1]

        if label not in detections:
            detections[label_name] = []

        detections[label_name].append([x1, y1, x2, y2, confidence])

    return detections

# model = YOLO("yolo11n.pt")
# img = cv2.imread("/Users/banika/Desktop/GreatUniHack/bus.jpg")
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (640, 640))
# img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
# # Run inference on a single image
# results = model(img)
#
# print(extract_boxes(results))
#
#
# # Display results
# results[0].show()
#


class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the model
        self.model = YOLO("yolo11n.pt")


        # Set up the camera
        self.cap = cv2.VideoCapture(0)

        # Set up the GUI
        self.setWindowTitle("Camera Inference App")

        # Video display label
        self.video_label = QLabel(self)

        # Start/Stop button
        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_camera)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Timer for the camera feed
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
        """Capture frame, run inference, and update the display."""
        ret, frame = self.cap.read()
        if ret:
            # Run inference

            # make a square image from poportions
            # if frame.shape[0] > frame.shape[1]:
            #     frame = frame[:frame.shape[1], :]
            # else:
            #     frame = frame[:, :frame.shape[0]]

            detections = self.run_inference(frame)

            # Draw detections on the frame
            for det_class in detections:
                for bndbox in detections[det_class]:
                    x1, y1, x2, y2, confidence = bndbox

                    # redo size of bdnbox for img size
                    w_ratio = frame.shape[1] / 640
                    h_ratio = frame.shape[0] / 640
                    x1 = int(x1 * w_ratio)
                    y1 = int(y1 * h_ratio)
                    x2 = int(x2 * w_ratio)
                    y2 = int(y2 * h_ratio)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{det_class} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert to QImage and display
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

        # Iterate over each detected box
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            label = int(box.cls.item())
            label_name = results[0].names[label].split()[-1]

            if label not in detections:
                detections[label_name] = []
            detections[label_name].append([x1, y1, x2, y2, confidence])

        return detections


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())




