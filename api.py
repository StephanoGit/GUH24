import time

import cv2
# import pyttsx3
from threading import Thread
from camera import WebcamRecorder
from openai_helper import OpenAIHelper
from face_animation import AnimatedFaceApp

import warnings
warnings.filterwarnings("ignore")

class API:
    stop_commands = ['stop', 'exit', 'quit', 'end', 'terminate', 'goodbye']

    def __init__(self):
        self.recorder = WebcamRecorder(video_source=0)  # Adjust the index if needed
        self.openai_helper = OpenAIHelper()
        self.face_app = AnimatedFaceApp()


    def animate_text(self, recorder, face_app, openai_helper):
        while True:
                response = None

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                try:
                    user_input = openai_helper.listen()
                    print(f"User: {user_input}")
                    if user_input:
                        # Check for various stop phrases
                        if any(command in user_input.lower() for command in self.stop_commands):
                            print("Ending the conversation.")
                            break

                        else:
                            if 'what' in user_input.lower() and 'this' in user_input.lower():
                                frame, detections = recorder.get_frame()
                                no_person_detection = {k: v for k, v in detections.items() if k not in ['person', 'KNOWN PERSON']}

                                if frame is not None:
                                    for detected_object in no_person_detection.keys():
                                        response = openai_helper.get_object_description(detected_object)
                                        print(f"Detected: {detected_object}")

                            elif 'who' in user_input.lower() and 'this' in user_input.lower():
                                frame, detections = recorder.get_frame()
                                for detected_object in detections.keys():
                                    if detected_object == 'person' or detected_object == 'KNOWN PERSON':
                                        if "KNOWN PERSON" in detected_object:
                                            response = "I see one my daddies."
                                        else:
                                            response = "Stranger danger "
                                            response = response * 2

                            else:
                                response = openai_helper.get_conversation(user_input)


                        if response:
                            face_app.animate_text(response)
                            l = 0.33 * len(response.split()) + 1
                            print(l)
                            time.sleep(l)






                except KeyboardInterrupt:
                    print("Program terminated.")
                    break






    def start_camera_and_speak(self):
        """Start the camera to capture frames, detect objects, and use GPT to describe them."""
        self.recorder.start()

        try:
            thread = Thread(target=self.animate_text, args=(self.recorder, self.face_app, self.openai_helper))
            thread.start()
            self.face_app.run()
            # self.animate_text(self.recorder, self.face_app, self.openai_helper)
        finally:
            self.recorder.stop()
            self.recorder.join()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    api = API()
    api.start_camera_and_speak()  # Start the camera and process detected objects
