import openai
import pyttsx3
import speech_recognition as sr

sr.AudioFile.FLAC_CONVERTER = "/opt/homebrew/bin/flac"

# Replace with a method for securely handling the API key
openai.api_key = 'sk-svcacct-cNpEjHaToqE8f1_oq5mtOav-MW58kAAPPnY2lzO3W3FdX1lTM4-B88AF-DU36xuVT3BlbkFJiYStWmktQpsH4HTId447QBQGCh4jmzqdiyzaD-Lk-hudqrQbinAEXD8tlcRAO9kA'

engine = pyttsx3.init()

def get_object_description(object_name, conversation_history):
    """Get a description from GPT based on the object name."""
    prompt = f"{object_name}. What is it used for in 10 words max?"
    conversation_history.append({"role": "user", "content": prompt})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            max_tokens=150
        )
        assistant_reply = response['choices'][0]['message']['content'].strip()
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply, conversation_history
    except openai.error.OpenAIError as e:
        print(f"An error occurred with the OpenAI API: {e}")
        return "I'm sorry, I couldn't process your request.", conversation_history

def speak(text):
    """Split long text into smaller chunks and speak them."""
    max_length = 150
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    for chunk in chunks:
        engine.say(chunk)
        engine.runAndWait()

conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

recognizer = sr.Recognizer()

def listen():
    """Capture and return spoken input as text."""
    with sr.Microphone() as source:
        print("Listening for your question...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source)
            print("Processing speech...")  # Feedback to the user
            speech_text = recognizer.recognize_google(audio)
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

# Initial interaction
detected_object = "cup"
description, conversation_history = get_object_description(detected_object, conversation_history)
print(f"Description of {detected_object}: {description}")
speak(description)

# Continuous interaction loop
stop_commands = ['stop', 'exit', 'quit', 'end', 'terminate', 'goodbye']

while True:
    try:
        user_input = listen()
        if user_input:
            # Check for various stop phrases
            if any(command in user_input.lower() for command in stop_commands):
                print("Ending the conversation.")
                break

            # Get response from GPT
            response, conversation_history = get_object_description(user_input, conversation_history)

            print(f"Assistant: {response}")
            speak(response)
    except KeyboardInterrupt:
        print("\nManual interruption detected. Ending the conversation.")
        break

print("Program terminated.")
