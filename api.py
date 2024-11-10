import cv2
from camera import WebcamRecorder
from openai_helper import get_object_description, speak, conversation_history

class API:
    def __init__(self):
        self.recorder = WebcamRecorder(video_source=1)  # Adjust the index if needed

    def start_camera_and_speak(self):
        """Start the camera to capture frames, detect objects, and use GPT to describe them."""
        self.recorder.start()

        try:
            while True:
                frame, detections = self.recorder.get_frame()
                if frame is not None:
                    cv2.imshow("Webcam with Detection", frame)

                    # Process and speak detected objects
                    for detected_object in detections.keys():
                        print(f"Detected: {detected_object}")

                        # Get a description of the detected object using GPT
                        description, _ = get_object_description(detected_object, conversation_history)
                        print(f"Description of {detected_object}: {description}")

                        # Speak the description
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
    api.start_camera_and_speak()  # Start the camera and process detected objects
