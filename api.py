import camera
import GUH24.face_recognition.model.face_recognition as face_recognition
from ultralytics import YOLO
import cv2, os

class API:
    def __init__(self):
        self.recorder = camera.WebcamRecorder(video_source=0)


    def start_camera(self):
        self.recorder.start()

        try:
            while True:
                frame, detections = self.recorder.get_frame()
                if frame is not None:
                    cv2.imshow("Webcam", frame)
                    print(detections.keys())

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Stop the recording and release resources
            self.recorder.stop()
            self.recorder.join()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    api = API()
    api.start_camera()

