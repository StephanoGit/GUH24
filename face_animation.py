class AnimatedFaceApp:
    def __init__(self):
        import pygame
        import numpy as np
        from gtts import gTTS
        import io
        from pydub import AudioSegment
        from threading import Thread

        self.pygame = pygame
        self.np = np
        self.gTTS = gTTS
        self.io = io
        self.AudioSegment = AudioSegment
        self.Thread = Thread

        # Initialize Pygame
        self.pygame.init()
        self.clock = self.pygame.time.Clock()
        self.screen = self.pygame.display.set_mode((640, 480))
        self.pygame.display.set_caption("Animated Face")

        # Create face and mouth surfaces
        self.face_surface = self.create_face_surface()
        self.mouth_open, self.mouth_closed = self.create_mouth_surfaces()

        # State variables
        self.STATE_IDLE = 0
        self.STATE_ANIMATING = 1
        self.state = self.STATE_IDLE
        self.running = True
        self.text_to_animate = None
        self.frame_rate = 24
        self.amplitudes = []
        self.total_frames = 0
        self.frame = 0
        self.audio_thread = None

    def create_face_surface(self):
        face_surface = self.pygame.Surface((640, 480), self.pygame.SRCALPHA)
        # Draw face (circle)
        self.pygame.draw.circle(face_surface, (255, 224, 189), (320, 240), 200)  # Face color
        # Draw eyes
        self.pygame.draw.circle(face_surface, (0, 0, 0), (260, 200), 20)  # Left eye
        self.pygame.draw.circle(face_surface, (0, 0, 0), (380, 200), 20)  # Right eye
        return face_surface

    def create_mouth_surfaces(self):
        mouth_open = self.pygame.Surface((640, 480), self.pygame.SRCALPHA)
        self.pygame.draw.ellipse(mouth_open, (150, 0, 0), [270, 320, 100, 50])  # Open mouth

        mouth_closed = self.pygame.Surface((640, 480), self.pygame.SRCALPHA)
        self.pygame.draw.line(mouth_closed, (0, 0, 0), (270, 345), (370, 345), 5)  # Closed mouth

        return mouth_open, mouth_closed

    def text_to_speech_in_memory(self, text):
        tts = self.gTTS(text=text, lang='en')
        audio_buffer = self.io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer

    def extract_audio_features_from_memory(self, audio_buffer, frame_rate=12):
        audio = self.AudioSegment.from_file(audio_buffer, format="mp3")
        samples = self.np.array(audio.get_array_of_samples())
        # Handle stereo audio
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels))
            samples = samples.mean(axis=1)
        # Normalize samples
        samples = samples / self.np.max(self.np.abs(samples))
        # Calculate amplitude envelope
        duration = audio.duration_seconds
        total_frames = int(duration * frame_rate)
        frame_length = max(1, len(samples) // total_frames)
        amplitudes = [
            self.np.max(self.np.abs(samples[i*frame_length:(i+1)*frame_length]))
            for i in range(total_frames)
        ]
        return amplitudes, duration

    def play_audio_from_memory(self, audio_buffer):
        audio_buffer.seek(0)
        # Convert MP3 buffer to WAV buffer
        audio_segment = self.AudioSegment.from_file(audio_buffer, format="mp3")
        wav_io = self.io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        # Initialize mixer with default parameters
        self.pygame.mixer.init()
        sound = self.pygame.mixer.Sound(file=wav_io)
        sound.play()
        # Wait until playback finishes
        while self.pygame.mixer.get_busy():
            self.pygame.time.Clock().tick(10)

    def animate_text(self, text):
        if not text:
            return

        if self.state == self.STATE_IDLE:
            # Prepare to animate
            self.text_to_animate = text
            self.state = self.STATE_ANIMATING
        else:
            # If already animating, you can choose to queue the text or ignore it
            print("Currently animating. Please wait.")

        return True

    def run(self):
        while self.running:
            if self.state == self.STATE_IDLE:
                # Display idle face
                for event in self.pygame.event.get():
                    if event.type == self.pygame.QUIT:
                        self.running = False

                self.screen.fill((255, 255, 255))
                self.screen.blit(self.face_surface, (0, 0))
                self.screen.blit(self.mouth_closed, (0, 0))
                self.pygame.display.flip()
                self.clock.tick(30)

            elif self.state == self.STATE_ANIMATING:
                if self.text_to_animate:
                    # Convert text to speech in memory
                    audio_buffer = self.text_to_speech_in_memory(self.text_to_animate)
                    # Extract audio features from memory
                    self.amplitudes, duration = self.extract_audio_features_from_memory(audio_buffer, self.frame_rate)
                    # Start audio playback in a separate thread
                    self.audio_thread = self.Thread(target=self.play_audio_from_memory, args=(audio_buffer,))
                    self.audio_thread.start()
                    # Set up animation variables
                    self.total_frames = len(self.amplitudes)
                    self.frame = 0
                    self.text_to_animate = None

                # Animate the face
                if self.frame < self.total_frames:
                    for event in self.pygame.event.get():
                        if event.type == self.pygame.QUIT:
                            self.running = False

                    self.screen.fill((255, 255, 255))
                    self.screen.blit(self.face_surface, (0, 0))

                    # Determine mouth state based on amplitude
                    if self.amplitudes[self.frame] > 0.15:  # Adjust threshold as needed
                        self.screen.blit(self.mouth_open, (0, 0))
                    else:
                        self.screen.blit(self.mouth_closed, (0, 0))

                    self.pygame.display.flip()
                    self.clock.tick(self.frame_rate)
                    self.frame += 1
                else:
                    # Wait for audio thread to finish
                    if self.audio_thread:
                        self.audio_thread.join()
                        self.audio_thread = None
                    # Return to idle state
                    self.state = self.STATE_IDLE

        # Quit Pygame
        self.pygame.quit()




if __name__ == '__main__':
    from threading import Thread

    app = AnimatedFaceApp()

    # Function to get text input from the user
    def input_thread(app):
        while app.running:
            text = input("Enter text to animate (or 'quit' to exit): ")
            if text.lower() == 'quit':
                app.running = False
                break
            app.animate_text(text)

    # Start the input thread
    thread = Thread(target=input_thread, args=(app,))
    thread.start()

    # Run the app (this will block until the app is closed)
    app.run()

