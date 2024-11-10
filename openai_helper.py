import openai
from openai import OpenAIError
import pyttsx3
import speech_recognition as sr



class OpenAIHelper:
    def __init__(self):
        """
        Initialize the AnimatedFaceApp with OpenAI API key.
        """
        # Initialize OpenAI API key
        openai.api_key = 'sk-svcacct-cNpEjHaToqE8f1_oq5mtOav-MW58kAAPPnY2lzO3W3FdX1lTM4-B88AF-DU36xuVT3BlbkFJiYStWmktQpsH4HTId447QBQGCh4jmzqdiyzaD-Lk-hudqrQbinAEXD8tlcRAO9kA'

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()

        # Initialize conversation history
        self.conversation_history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        # Define stop commands
        self.stop_commands = ['stop', 'exit', 'quit', 'end', 'terminate', 'goodbye']

    def get_conversation(self, description):
        """
        Get a description from GPT based on the object name.
        """
        prompt = "Respond to " + description + " in 10 words max"
        self.conversation_history.append({"role": "user", "content": prompt})

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.conversation_history,
                max_tokens=150
            )
            assistant_reply = response['choices'][0]['message']['content'].strip()
            self.conversation_history.append({"role": "assistant", "content": assistant_reply})
            return str(assistant_reply)
        except OpenAIError as e:
            print(f"An error occurred with the OpenAI API: {e}")
            return "I'm sorry, I couldn't process your request."

    def get_object_description(self, object_name):
        """
        Get a description from GPT based on the object name.
        """
        prompt = f"{object_name}. What is it used for in 10 words max?"
        self.conversation_history.append({"role": "user", "content": prompt})

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.conversation_history,
                max_tokens=150
            )
            assistant_reply = response['choices'][0]['message']['content'].strip()
            self.conversation_history.append({"role": "assistant", "content": assistant_reply})
            return str(assistant_reply)
        except OpenAIError as e:
            print(f"An error occurred with the OpenAI API: {e}")
            return "I'm sorry, I couldn't process your request."

    def speak(self, text):
        """
        Split long text into smaller chunks and speak them.
        """
        max_length = 150
        chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
        for chunk in chunks:
            self.engine.say(chunk)
            self.engine.runAndWait()

    def listen(self):
        """
        Capture and return spoken input as text.
        """
        with sr.Microphone() as source:
            print(sr.Microphone.device_index)
            print("Listening for your question...")
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(source)
                print("Processing speech...")  # Feedback to the user
                speech_text = self.recognizer.recognize_google(audio)
                print(f"You said: {speech_text}")
                return speech_text
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand that. Please try again.")
                return None
            except sr.RequestError as e:
                print(f"Speech recognition service error: {e}")
                return None
            except KeyboardInterrupt:
                print("\nProgram interrupted by user.")
                return "stop"

    def run(self, detected_object = "-9"):
        """
        Run the initial interaction and continuous interaction loop.
        """
        # Initial interactio

        if detected_object != "-9":
            description = self.get_object_description(detected_object)
            print(f"Description of {detected_object}: {description}")


            # Continuous interaction loop
            while True:
                try:
                    user_input = self.listen()
                    if user_input:
                        # Check for various stop phrases
                        if any(command in user_input.lower() for command in self.stop_commands):
                            print("Ending the conversation.")
                            break

                        # Get response from GPT
                        response = self.get_object_description(user_input)

                        print(f"Assistant: {response}")
                        # self.speak(response)
                except KeyboardInterrupt:
                    print("\nManual interruption detected. Ending the conversation.")
                    break

        print("Program terminated.")


# Usage Example
if __name__ == '__main__':
    app = OpenAIHelper()
    app.run()