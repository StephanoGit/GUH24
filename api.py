import time

import cv2
# import pyttsx3
from threading import Thread
from camera import WebcamRecorder
from openai_helper import OpenAIHelper
from face_animation import AnimatedFaceApp
# aaa
import serial

import warnings
warnings.filterwarnings("ignore")

class API:
    stop_commands = ['stop', 'exit', 'quit', 'end', 'terminate', 'goodbye']

    def __init__(self):
        self.recorder = WebcamRecorder(video_source=0)  # Adjust the index if needed
        self.openai_helper = OpenAIHelper()
        self.face_app = AnimatedFaceApp()



    def innit_arduino(self, port  = '/dev/ttyACM0'):
        ser = serial.Serial(port, 9600, timeout=1)
        return ser


    def send_to_arduino(self, ser, command, speed = 100):
        cmd = f"{command},{speed}\n"
        ser.write(cmd.encode())
        time.sleep(0.1)




    def animate_text(self, recorder, face_app, openai_helper):
        ser = self.innit_arduino()

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

                            elif 'forward' in user_input.lower():
                                self.send_to_arduino(ser, 'F', 100)
                                response = "Moving forward"

                            elif 'backward' in user_input.lower():
                                self.send_to_arduino(ser, 'B', 100)
                                response = "Moving backward"

                            elif 'left' in user_input.lower():
                                self.send_to_arduino(ser, 'L', 100)
                                response = "Moving left"

                            elif 'right' in user_input.lower():
                                self.send_to_arduino(ser, 'R', 100)
                                response = "Moving right"

                            elif 'follow' in user_input.lower():
                                self.send_to_arduino(ser, 'F', 100)
                                response = "Following Someone"

                                for _ in range(10):
                                    frame, detections = recorder.get_frame()
                                    if 'person' in detections.keys() or 'KNOWN PERSON' in detections.keys():
                                        for detected_object in detections.keys():
                                            if detected_object == 'person' or detected_object == 'KNOWN PERSON':
                                                person_bndbox = detections[detected_object][0]
                                                x1, y1, x2, y2, _ = person_bndbox
                                                width = x2 - x1
                                                height = y2 - y1
                                                center_x = x1 + width // 2
                                                frame_center_x = frame.shape[1] // 2

                                                if abs(center_x - frame_center_x) > 30:  # Tolerance of 30 pixels
                                                    if center_x < frame_center_x:
                                                        self.send_to_arduino(ser, 'l', 100)  # Move left
                                                    else:
                                                        self.send_to_arduino(ser, 'r', 100)  # Move right
                                                else:
                                                    # Person is roughly centered
                                                    if height < 200:  # Assume height threshold for distance
                                                        self.send_to_arduino(ser, 'f', 100)  # Move forward
                                                    else:
                                                        self.send_to_arduino(ser, 's', 0)  # Stop if close enough
                                                        print("Person reached. Stopping.")
                                                        break  # Exit follow mode
                                    else:
                                        response = "No target sir"

                            elif 'boogie' in user_input.lower() or 'dance' in user_input.lower():
                                dance_moves = [
                                    ('L', 100),
                                    ('R', 100),
                                    ('F', 150),
                                    ('B', 150),
                                    ('R', 120),
                                    ('R', 120),
                                    ('L', 100),
                                    ('R', 100),
                                    ('F', 150),
                                    ('B', 150),
                                    ('L', 120),
                                    ('R', 120)
                                ]

                                response = 'Never gonna give you up   Never gonna let you down   Never gonna run around and desert you   Never gonna make you cry   Never gonna say goodbye   Never gonna tell a lie and hurt you'
                                face_app.animate_text(response)

                                for move, speed in dance_moves:
                                    self.send_to_arduino(ser, move, speed)
                                    time.sleep(1)

                                response = None

                            else:
                                response = openai_helper.get_conversation(user_input)


                        if response:
                            face_app.animate_text(response)
                            l = 0.33 * len(response.split()) + 1
                            print(l)
                            time.sleep(l)


                except KeyboardInterrupt:
                    print("Program terminated.")
                    ser.close()
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
