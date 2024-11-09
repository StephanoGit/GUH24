import openai
import pyttsx3
import speech_recognition as sr

sr.AudioFile.FLAC_CONVERTER = "/opt/homebrew/bin/flac"

openai.api_key = ''

engine = pyttsx3.init()


def get_object_description(object_name, conversation_history):
    # Generate the prompt based on the object name
    prompt = f"{object_name}. What is it used for in 10 words max?"

    # Append user's query to the conversation history
    conversation_history.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the updated chat model
        messages=conversation_history,
        max_tokens=150
    )

    # Extract and append the assistant's reply to the conversation history
    assistant_reply = response['choices'][0]['message']['content'].strip()
    conversation_history.append(
        {"role": "assistant", "content": assistant_reply})

    return assistant_reply, conversation_history

# Function for text-to-speech


def speak(text):
    engine.say(text)
    engine.runAndWait()


conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

recognizer = sr.Recognizer()


def listen():
    with sr.Microphone() as source:
        print("Listening for your question...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            speech_text = recognizer.recognize_google(audio)
            print(f"You said: {speech_text}")
            return speech_text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that. Please try again.")
            return None
        except sr.RequestError:
            print("Sorry, there was an issue with the speech recognition service.")
            return None


detected_object = "cup"  # Example object detected by the camera


description, conversation_history = get_object_description(
    detected_object, conversation_history)
print(f"Description of {detected_object}: {description}")
speak(description)

while True:
    user_input = listen()  # Listen for user speech
    if user_input:
        if user_input.lower() in ['exit', 'quit', 'stop']:
            print("Ending the conversation.")
            break

        conversation_history.append({"role": "user", "content": user_input})

        response, conversation_history = get_object_description(
            user_input, conversation_history)

        print(f"Assistant: {response}")
        speak(response)
