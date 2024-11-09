import camera
import GUH24.face_recognition.model.face_recognition as face_recognition
from ultralytics import YOLO
import cv2
import os

from openai_helper import get_object_description, speak, conversation_history as conversation_history


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
                    # print(detections.keys())

                    for detected_object in detections.keys():
                        print(f"Detected: {detected_object}")
                        description, _ = get_object_description(
                            detected_object, conversation_history)
                        print(
                            f"Description of {detected_object}: {description}")
                        speak(description)

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
